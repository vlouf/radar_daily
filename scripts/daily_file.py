"""
CPOL Level 2 main production line.

@title: CPOL_LEVEL2
@author: Valentin Louf <valentin.louf@monash.edu>
@institution: Bureau of Meteorology
@date: 15/09/2017
@version: 0.4

.. autosummary::
    :toctree: generated/

    processing_line
    multiproc_buffer
    main
"""
# Python Standard Library
import gc
import os
import re
import sys
import time
import signal
import argparse
import datetime
import warnings
import traceback

from multiprocessing import Pool

# Other libraries
import crayons
import pyart
import netCDF4
import pandas as pd
import numpy as np

# Custom modules
from processing_codes.utils import *
from processing_codes.io import read_radar, write_ncfile
from processing_codes import calc

warnings.simplefilter('ignore')


class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]
        

def get_moment(radar, moment_name, fillvalue=-9999):
    """
    Parameters:
    ===========
        input_file: str
            Input file name.
        moment_name: str
            Radar field name.

    Returns:
    ========
        moment_data: ndarray[DIM_LEN, DIM_LEN]
            Moment data fill value is FILLVALUE.
    """
    mymoment = radar.fields[moment_name]

    z = radar.z['data']
    if 'reflectivity' in moment_name:
        moment_data = np.squeeze(mymoment['data'][z == 2500, :, :].filled(fillvalue))
    else:
        moment_data = np.squeeze(mymoment['data'][0, :, :].filled(fillvalue))

    #  Check if maximum range is 70 km or 145 km.
    if np.max(radar.x['data']) > 135000:
        x = radar.x['data']
        y = radar.y['data']

        # NaNing data outside of radar horizon.
        [X, Y] = np.meshgrid(x, y)
        moment_data[(X**2 + Y**2) > 140000**2] = np.NaN

    return moment_data


def processing_line(input_dir, output_dir, year, month, day):
    """
    Transform gridded level 1b files created with the cpol_processing code to
    level 2 files. Take as input not a file, but a directory containing all
    gridded files for one day.

    Parameters:
    ===========
    input_dir: str
        Input directory containing radar gridded files.
    output_dir: str
        Output directory where to save data.
    """
    file_exist = np.zeros((144,), dtype=int)
    # Moments to extract.
    goodkeys = ['corrected_differential_reflectivity',
                'radar_echo_classification', 'D0',
                'NW', 'reflectivity', 'radar_estimated_rain_rate']

    # New moments to compute
    newkeys = ["steiner_echo_classification", "thurai_echo_classification"]  #, "0dB_echo_top_height",
               #"10dB_echo_top_height", "17dB_echo_top_height", "40dB_echo_top_height", "cloud_top_height"]

    # Creating a 24h time array with a 10 min resolution.
    stt = datetime.datetime(year, month, day, 0, 0, 0)  # Start time.
    edt = stt + datetime.timedelta(hours=23, minutes=50)  # End time.
    drange = pd.date_range(stt, edt, freq='10min')

    # Creating netCDF4 compliant time array.
    proc_time = dict()
    proc_time['data'] = np.zeros((144,))
    proc_time['units'] = TIME_UNIT
    for cnt, myrange in enumerate(drange):
        proc_time['data'][cnt] = netCDF4.date2num(myrange, TIME_UNIT)

    # Initialize empty storage
    MOMENT = dict()
    for mykey in goodkeys:
        MOMENT[mykey] = {"data": np.zeros((144, DIM_LEN, DIM_LEN)) + np.NaN}
    for mykey in newkeys:
        MOMENT[mykey] = {"data": np.zeros((144, DIM_LEN, DIM_LEN)) + np.NaN}

    # Get file list of radars data.
    flist = get_flist(input_dir, drange, RES)
    if flist is None:
        return None

    # Check if all files are None.
    if all(onefile is None for onefile in flist):
        print(crayons.yellow("No file found."))
        return None

    # Extract data.
    for cnt, input_file in enumerate(flist):
        if input_file is None:
            continue

        radar = read_radar(input_file)
        file_exist[cnt] = 1

        # Parsing existing keys.
        for mykey in goodkeys:
            mymoment = get_moment(radar, mykey, fillvalue=FILLVALUE)
            if mykey in ['D0', 'NW', 'radar_estimated_rain_rate']:
                mymoment[mymoment == FILLVALUE] = 0
                mymoment = np.ma.masked_where(np.isnan(mymoment), mymoment)
            else:
                mymoment = np.ma.masked_where(np.isnan(mymoment) | (mymoment == FILLVALUE), mymoment)

            MOMENT[mykey]['data'][cnt, :, :] = mymoment
            # Include metadata
            if len(MOMENT[mykey].keys()) == 1:
                moment_meta = get_moment_meta(radar, mykey)
                for key, val in moment_meta.items():
                    MOMENT[mykey][key] = val

        # Extractin radar data needed for echo top height. out of the loop.
        x = radar.x['data']
        y = radar.y['data']
        z = radar.z['data']
        refl = radar.fields['reflectivity']['data']

        # Parsing newkeys.
        for mykey in newkeys:
            if mykey == "steiner_echo_classification":
                eclass = calc.steiner_classification(radar)
                MOMENT[mykey]['data'][cnt, :, :] = eclass['data']
                # Checking if moment metadata already exist.
                if len(MOMENT[mykey].keys()) == 1:
                    for key, val in eclass.items():
                        if key == "data":
                            continue
                        else:
                            MOMENT[mykey][key] = val

            elif mykey == "thurai_echo_classification":
                eclass = calc.thurai_echo_classification(radar)
                MOMENT[mykey]['data'][cnt, :, :] = eclass['data']
                # Checking if moment metadata already exist.
                if len(MOMENT[mykey].keys()) == 1:
                    for key, val in eclass.items():
                        if key == "data":
                            continue
                        else:
                            MOMENT[mykey][key] = val

            elif "echo_top_height" in mykey:
                # Get the dB threshold from string.
                threshold = int(re.sub("dB_echo_top_height", "", mykey))
                # Extract required data
                stein = MOMENT["steiner_echo_classification"]['data'][cnt, :, :]
                # Computing the echo top height.
                echotop = calc.echo_top_height(x, y, z, refl, stein, threshold=threshold)
                MOMENT[mykey]['data'][cnt, :, :] = echotop['data']
                # Checking if moment metadata already exist.
                if len(MOMENT[mykey].keys()) == 1:
                    for key, val in echotop.items():
                        if key == "data":
                            continue
                        else:
                            MOMENT[mykey][key] = val

            elif mykey == "cloud_top_height":
                cth = calc.cloud_top_height(radar)
                MOMENT[mykey]['data'][cnt, :, :] = cth['data']
                # Checking if moment metadata already exist.
                if len(MOMENT[mykey].keys()) == 1:
                    for key, val in cth.items():
                        if key == "data":
                            continue
                        else:
                            MOMENT[mykey][key] = val

    print(crayons.blue("Processing finished for {}.".format(stt.strftime("%Y%m%d"))))

    # Get general metadata.
    main_meta = get_radar_meta(radar)
    # Extract latitude/longitude at level 5 (2.5 km of altitude)
    longitude, latitude = radar.get_point_longitude_latitude(level=5)

    # Writing originals moments.
    for mykey, mymoment in MOMENT.items():
        # generating output filename.
        outfilename = generate_outfilename(output_dir, mykey, stt)
        if os.path.isfile(outfilename):
            print(crayons.yellow("{} already exists. Doing nothing.".format(outfilename)))
            continue

        # Writing level 2 for this date and moment.
        try:
            write_ncfile(outfilename, proc_time, XDIM, XDIM, latitude, longitude,
                         mymoment, mykey, main_meta, file_exist)
        except RuntimeError:
            print(crayons.red(f"RuntimeError for {outfilename}"))
            write_ncfile(outfilename, proc_time, XDIM, XDIM, latitude, longitude,
                         mymoment, mykey, main_meta, file_exist)
            print(crayons.green(f"RuntimeError for {outfilename} passed."))
        print(crayons.green("{} written.".format(outfilename)))

    print("Process for {} done.".format(stt.strftime("%Y%m%d")))

    return None


def multiproc_buffer(input_dir, output_dir, dtime):
    """
    Buffer function that is just here to handle errors during multiprocessing.
    Input arguments are the same as processing_line input arguments.
    """

    print("Looking at {}.".format(dtime.isoformat()))
    year = dtime.year
    month = dtime.month
    day = dtime.day

    # Chronometer the processing time and kills it if taking more than 600s.
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(240)

    try:
        tic = time.time()
        processing_line(input_dir, output_dir, year, month, day)
        toc = time.time()
        print(crayons.green("Processed in {}s.".format(toc - tic)))
    except TimeoutException:
        # Treatment time was too long.
        print("Too much time taken to treat {}, killing process.".format(dtime.strftime("%Y%m%d")))
        return None  # Go to next iteration.
    except Exception:
        print(crayons.red("!!! ERROR !!!", bold=True))
        traceback.print_exc()
        print(crayons.red("!!! ERROR !!!", bold=True))
        return None
    else:
        signal.alarm(0)

    # Collect memory garbage.
    gc.collect()

    return None


def main():
    # Get a copy of global variables
    input_dir = INDIR
    output_dir = OUTDIR
    stt = START_DATE
    edt = END_DATE
    # Initialize empty argument list for multiprocessing call
    args_list = []

    # Generate argument list for multiprocessing call.
    for dtime in pd.date_range(stt, edt):
        args_list.append((input_dir, output_dir, dtime))

    # Multiprocessing starts here.
    # Here multiprocessing will treat several days at the same time.
    for myargs in chunks(args_list, NCPU * 2):
        with Pool(NCPU) as pool:
            pool.starmap(multiproc_buffer, myargs)

    return None


if __name__ == "__main__":
    # Global variables that are to be set through argument parser in the future.
    RES = 2500

    # Parse arguments
    parser_description = "Leveling treatment of CPOL data from level 1a to level 1b."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument('-j', '--cpu', dest='ncpu', default=32, type=int, help='Number of process')
    parser.add_argument('-s', '--start-date', dest='start_date', default=None, type=str, help='Starting date.')
    parser.add_argument('-e', '--end-date', dest='end_date', default=None, type=str, help='Ending date.')
    parser.add_argument('-i', '--indir', dest='indir', default="/g/data2/rr5/vhl548/NEW_CPOL_level_1b/",
                        type=str, help='Input directory.')
    parser.add_argument('-o', '--output', dest='outdir', default="/g/data2/rr5/vhl548/NEW_CPOL_level_2/",
                        type=str, help='Output directory.')

    args = parser.parse_args()
    NCPU = args.ncpu
    START_DATE = args.start_date
    END_DATE = args.end_date
    OUTDIR = args.outdir
    INPUT_DIR = args.indir

    if not (START_DATE and END_DATE):
        parser.error("Starting and ending date required.")

    # Checking that dates are recognize.
    try:
        START_DATE = datetime.datetime.strptime(START_DATE, "%Y%m%d")
        END_DATE = datetime.datetime.strptime(END_DATE, "%Y%m%d")
    except Exception:
        print("Did not understand the date format. Must be YYYYMMDD.")
        sys.exit()

    # Other variables definition.
    FILLVALUE = -9999

    TIME_UNIT = "seconds since 1970-01-01 00:00"

    if RES == 2500:
        DIM_LEN = 117
        XDIM = np.arange(-145000, 145000 + 1, 2500)
        INDIR = os.path.join(INPUT_DIR, "GRIDDED", "GRID_150km_2500m")
    elif RES == 1000:
        DIM_LEN = 141
        XDIM = np.arange(-70000, 70000 + 1, 1000)
        INDIR = os.path.join(INPUT_DIR, "GRIDDED", "GRID_70km_1000m")

    if not os.path.exists(INDIR):
        raise FileNotFoundError("Input directory {} does not exists".format(INDIR))

    main()

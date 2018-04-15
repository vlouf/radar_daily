"""
CPOL Level 2 main production line.

@title: CPOL_LEVEL2
@author: Valentin Louf <valentin.louf@monash.edu>
@institution: Bureau of Meteorology
@date: 15/05/2017
@version: 0.1

.. autosummary::
    :toctree: generated/

    write_ncfile
    daily_file
    multiproc_buffer
    main
"""
# Python Standard Library
import gc
import os
import sys
import glob
import time
import argparse
import datetime
import warnings
import traceback

from multiprocessing import Pool
from copy import deepcopy

# Other libraries
import crayons
import pyart
import netCDF4
import pandas as pd
import numpy as np

# Custom modules
from processing_codes.util_codes import *

warnings.simplefilter('ignore')


def write_ncfile(outfilename, time, xdim, ydim, latitude, longitude, moment, moment_name, moment_meta, gnrl_meta):
    """
    Write level 2 netCDF4 file.
    """
    # Write netCDF4 file.
    with netCDF4.Dataset(outfilename, "w", format="NETCDF4") as rootgrp:
        # Create dimension
        rootgrp.createDimension("x", DIM_LEN)  # DIM_LEN is a global variable.
        rootgrp.createDimension("y", DIM_LEN)
        rootgrp.createDimension('time', 144)

        # Create variables.
        mymoment = rootgrp.createVariable(moment_name, 'i4', ("time", "x", "y"), zlib=True, fill_value=-9999)
        nclat = rootgrp.createVariable('latitude', 'f8', ("x", "y"), zlib=True)
        nclon = rootgrp.createVariable('longitude', 'f8', ("x", "y"), zlib=True)
        nctime = rootgrp.createVariable('time', 'f8', 'time')
        ncx = rootgrp.createVariable('x', 'i4', 'x')
        mcy = rootgrp.createVariable('y', 'i4', 'y')

        # Assign values.
        mymoment[:] = moment
        nclat[:] = latitude
        nclon[:] = longitude
        nctime[:] = time
        ncx[:] = xdim
        mcy[:] = ydim

        # Set units.
        mymoment.units = moment_meta['units']
        nclat.units = "degrees North"
        nclon.units = "degrees West"
        nctime.units = TIME_UNIT  # Global variable.
        ncx.units = "meters"
        mcy.units = "meters"

        # Set main metadata
        for mykey in gnrl_meta.keys():
            rootgrp.setncattr_string(mykey, gnrl_meta[mykey])

        # Set moment metadata.
        for mykey in moment_meta.keys():
            if mykey == "units":
                continue

            try:
                mymoment.setncattr_string(mykey, moment_meta[mykey])
            except AttributeError:
                continue

    return None


def daily_file(input_dir, output_dir, year, month, day):
    """
    Makeing a daily file for dis.

    Parameters:
    ===========
        input_dir: str
            Input directory containing radar gridded files.
        output_dir: str
            Output directory where to save data.
    """
    # Moment to process.
    mykey = "echo_classification"

    # Creating a 24h time array with a 10 min resolution.
    stt = datetime.datetime(year, month, day, 0, 0, 0)  # Start time.
    edt = stt + datetime.timedelta(hours=23, minutes=50)  # End time.
    drange = pd.date_range(stt, edt, freq='10min')

    # Creating netCDF4 compliant time array.
    for cnt, myrange in enumerate(drange):
        TIME[cnt] = netCDF4.date2num(myrange, TIME_UNIT)

    # Initialize empty storage
    MOMENT = dict()
    MOMENT[mykey] = np.zeros((144, DIM_LEN, DIM_LEN)) + np.NaN

    # Get file list of radars data.
    flist = get_flist(input_dir, drange)
    if flist is None:
        return None

    # Check if all files are None.
    if all(onefile is None for onefile in flist):
        print("No file found.")
        return None

    # Extract data.
    for cnt, input_file in enumerate(flist):
        if input_file is None:
            continue
        try:
            radar = pyart.io.read_grid(input_file)
            print("{} read.".format(os.path.basename(input_file)))
        except OSError:
            continue

        # Store value in big dictionnary.
        # TODO: Steiner class.
        rslt = steiner_classification(radar)
        MOMENT[mykey][cnt, :, :] = deepcopy(rslt['data'])

    # Get general metadata.
    main_meta = get_radar_meta(radar)
    # Extract latitude/longitude at level 5 (2.5 km of altitude)
    longitude, latitude = radar.get_point_longitude_latitude(level=5)

    # Check if output directory exists -- Moment name.
    output_dir_ncfile = os.path.join(output_dir, mykey.upper())
    if not os.path.isdir(output_dir_ncfile):
        try:
            os.mkdir(output_dir_ncfile)
        except FileExistsError:
            pass

    # Check if output directory exists -- Year.
    output_dir_ncfile = os.path.join(output_dir_ncfile, str(year))
    if not os.path.isdir(output_dir_ncfile):
        try:
            os.mkdir(output_dir_ncfile)
        except FileExistsError:
            pass

    # Generate output file name and check if it already exists.
    outfilename = "CPOL_" + mykey.upper() + "_" + stt.strftime("%Y%m%d") + "_level2.nc"
    outfilename = os.path.join(output_dir_ncfile, outfilename)
    if os.path.isfile(outfilename):
        print(crayons.yellow("{} already exists. Doing nothing.".format(outfilename)))
        return None

    moment_meta = {'standard_name': 'echo_classification',
                   'long_name': 'Steiner echo classification',
                   'valid_min': 0, 'valid_max': 2,
                   'units': 'UA',
                   'comment_1': 'Convective-stratiform echo classification based on Steiner et al. (1995)',
                   'comment_2': '0 = Undefined, 1 = Stratiform, 2 = Convective'}

    moment = MOMENT[mykey]
    write_ncfile(outfilename, TIME, XDIM, XDIM, latitude, longitude, moment, mykey, moment_meta, main_meta)
    print("{} written.".format(outfilename))
    print("Process for {} done.".format(stt.strftime("%Y%m%d")))

    return None


def multiproc_buffer(input_dir, output_dir, dtime):
    """
    Buffer function that is just here to handle errors during multiprocessing.
    Input arguments are the same as daily_file input arguments.
    """

    print("Looking at {}.".format(dtime.isoformat()))
    year = dtime.year
    month = dtime.month
    day = dtime.day

    try:
        tic = time.time()
        daily_file(input_dir, output_dir, year, month, day)
        toc = time.time()
        print(crayons.green("Processed in {}s.".format(toc-tic)))
    except Exception:
        traceback.print_exc()
        return None

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
    with Pool(NCPU) as pool:
        pool.starmap(multiproc_buffer, args_list)

    return None


if __name__ == "__main__":
    # Global variables that are to be set through argument parser in the future.
    RES = 2500

    # Parse arguments
    parser_description = "Leveling treatment of CPOL data from level 1a to level 1b."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument('-j', '--cpu', dest='ncpu', default=16, type=int, help='Number of process')
    parser.add_argument('-s', '--start-date', dest='start_date', default=None, type=str, help='Starting date.')
    parser.add_argument('-e', '--end-date', dest='end_date', default=None, type=str, help='Ending date.')
    parser.add_argument('-o', '--output', dest='outdir', default="/g/data2/rr5/vhl548/CPOL_level_2/", type=str, help='Output directory.')

    args = parser.parse_args()
    NCPU = args.ncpu
    START_DATE = args.start_date
    END_DATE = args.end_date
    OUTDIR = args.outdir

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
    TIME = np.zeros((144,))
    TIME_UNIT = "seconds since 1970-01-01 00:00"

    if RES == 2500:
        DIM_LEN = 117
        XDIM = np.arange(-145000, 145000 + 1, 2500)
        INDIR = "/g/data2/rr5/vhl548/v2CPOL_PROD_1b/GRIDDED/GRID_150km_2500m/"
    elif RES == 1000:
        DIM_LEN = 141
        XDIM = np.arange(-70000, 70000 + 1, 1000)
        INDIR = "/g/data2/rr5/vhl548/v2CPOL_PROD_1b/GRIDDED/GRID_70km_1000m/"

    main()

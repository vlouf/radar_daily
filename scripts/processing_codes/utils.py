# Python Standard Library
import os
import datetime

# Other libraries.
import crayons
import numpy as np

from .io import get_key_metadata


def generate_outfilename(output_dir, mykey, stt):
    """
    Generate output file name. Will create output directory if needed.

    Parameters:
    ===========
    output_dir: str
        Root path to output directory.
    mykey: str
        Name of the parameter we want to save.
    stt: datetime
        Date of the data being saved.

    Returns:
    ========
    outfilename: str
        Complete output file name.
    """
    cfkeys = get_key_metadata()
    try:
        key_name = cfkeys[mykey]['standard_name']
    except KeyError:
        key_name = mykey

    year = stt.year
    # Check if output directory exists -- Moment name.
    output_dir_ncfile = os.path.join(output_dir, key_name.upper())
    try:
        os.mkdir(output_dir_ncfile)
    except FileExistsError:
        pass

    # Check if output directory exists -- Year.
    output_dir_ncfile = os.path.join(output_dir_ncfile, str(year))
    try:
        os.mkdir(output_dir_ncfile)
    except FileExistsError:
        pass

    # Generate output file name and check if it already exists.
    outfilename = "CPOL_" + key_name.upper() + "_" + stt.strftime("%Y%m%d") + "_level2.nc"
    outfilename = os.path.join(output_dir_ncfile, outfilename)

    return outfilename


def get_flist(input_dir, drange, grid_resolution="2500"):
    """
    Look for CPOL gridded netcdf files.

    Parameters:
    ===========
        input_dir: str
            Root path where CPOL gridded data are stored.
        drange: array
            Timestamp
        grid_resolution: str
            Or 2500 or 1000

    Returns:
    ========
        flist: List[str, ...]
            File list.
    """
    # Checking the grid resolution parameter.
    if type(grid_resolution) != str:
        grid_resolution = str(grid_resolution)

    if grid_resolution not in ["2500", "1000"]:
        raise ValueError("grid_resolution invalid, must be '2500' or '1000'")

    myrange = drange[0]
    indir = os.path.join(input_dir, str(myrange.year), myrange.strftime("%Y%m%d"))

    if not os.path.isdir(indir):
        print("Input directory does not exist for given date {}.".format(myrange.strftime("%Y%m%d_%H%M")))
        return None

    flist = [None] * len(drange)
    for file_number in range(len(drange)):
        myrange = drange[file_number]
        proto_file = "CPOL_{}_GRIDS_{}m.nc".format(myrange.strftime("%Y%m%d_%H%M"), grid_resolution)
        infile = os.path.join(indir, proto_file)

        if not os.path.isfile(infile):
            print(crayons.red("File {} does not exists.".format(infile)))
            flist[file_number] = None
            continue

        flist[file_number] = infile

    return flist


def get_moment_meta(radar, moment_name):
    """
    Extract radar's moment metadata and add comment.

    Parameters:
    ===========
        radar: struct
            PyART radar structure.
        moment_name: str
            Radar fields name.

    Returns:
    ========
        moment_meta: dict
            Metadata for given moment_name.
    """
    moment_meta = dict()
    for key, val in radar.fields[moment_name].items():
        if key == "data":
            continue
        else:
            moment_meta[key] = val

    return moment_meta


def get_radar_meta(radar):
    """
    Extract radar metadata and add comments.

    Parameters:
    ===========
        radar: struct
            PyART radar structure.

    Returns:
    ========
        mymeta: dict
            Metadata for radar.
    """
    mymeta = radar.metadata
    return mymeta

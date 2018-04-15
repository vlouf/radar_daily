# Python Standard Library
import os

# Other libraries.
import pyart
import numpy as np


def get_flist(input_dir, drange):
    """
    Look for CPOL gridded netcdf files.

    Parameters:
    ===========
        input_dir: str
            Root path where CPOL gridded data are stored.
        drange: array
            Timestamp

    Returns:
    ========
        flist: List[str, ...]
            File list.
    """
    myrange = drange[0]
    indir = os.path.join(input_dir, str(myrange.year), myrange.strftime("%Y%m%d"))

    if not os.path.isdir(indir):
        print("Input directory does not exist for given date {}.".format(myrange.strftime("%Y%m%d_%H%M")))
        return None

    flist = [None]*len(drange)
    for file_number in range(len(drange)):
        myrange = drange[file_number]
        proto_file = "CPOL_{}_GRIDS_2500m.nc".format(myrange.strftime("%Y%m%d_%H%M"))
        infile = os.path.join(indir, proto_file)

        if not os.path.isfile(infile):
            print("File {} does not exists.".format(infile))
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
    moment_meta = radar.fields[moment_name]
    silent = moment_meta.pop('data')
    moment_meta['comment'] = "Fill value is -9999 (=data exists, just empty). NaN represents an absence of data (= does not exist)."
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
    mymeta['comment'] = "CPOL product level 2 at 2.5 km altitude."
    mymeta['title'] = "CPOL product level 2"
    return mymeta


def read_extract(radar, moment_name, fillvalue=-9999):
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

    rainrate = radar.fields[moment_name]

    z = radar.z['data']
    moment_data = np.squeeze(rainrate['data'][z == 2500, :, :].filled(fillvalue))

    #  Check if maximum range is 70 km or 145 km.
    if np.max(radar.x['data']) > 140000:
        x = radar.x['data']
        y = radar.y['data']

        # NaNing data outside of radar horizon.
        [X, Y] = np.meshgrid(x, y)
        moment_data[(X**2 + Y**2) > 145000**2] = np.NaN

    return moment_data


def steiner_classification(radar, refl_name="corrected_reflectivity", altitude=2500):
    """
    Parameters:
    ===========
        input_file: str
            Input file name.
        refl_name: str
            Radar reflectivity field name.
        altitude: float
            Work level altitude in m.

    Returns:
    ========
        eclass: dict
            Moment data fill value is FILLVALUE.
    """
    # Extract coordinates
    x = radar.x['data']
    y = radar.y['data']
    dx = np.abs(x[1] - x[0])
    dy = np.abs(y[1] - y[0])

    # Compute steiner classificaiton
    eclass = pyart.retrieve.steiner_conv_strat(radar, dx=dx, dy=dy, work_level=altitude, refl_field=refl_name)
    eclass['data'] = np.ma.masked_where(eclass['data'] == 0, eclass['data'])

    return eclass

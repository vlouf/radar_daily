# Python Standard Library
import os
import uuid  # CF convention metadata.
import datetime

# Other libraries
import pyart
import netCDF4
import numpy as np


def get_key_metadata():
    cfkeys = dict()
    cfkeys['corrected_differential_reflectivity'] = {'standard_name': 'log_differential_reflectivity_hv',
                                                     'short_name': 'ZDR',
                                                     'units': 'dB'}

    cfkeys['reflectivity'] = {'standard_name': 'equivalent_reflectivity_factor',
                              'short_name': 'DBZ',
                              'units': 'dBZ'}

    cfkeys['region_dealias_velocity'] = {'standard_name': 'radial_velocity_of_scatterers_away_from_instrument',
                                         'short_name': 'VEL',
                                         'units': 'm/s'}

    cfkeys['giangrande_differential_phase'] = {'standard_name': 'differential_phase_hv',
                                               'short_name': 'PHIDP',
                                               'units': 'degrees'}

    cfkeys['giangrande_specific_differential_phase'] = {'standard_name': 'specific_differential_phase_hv',
                                                        'short_name': 'KDP',
                                                        'units': 'degrees/km'}

    cfkeys['cross_correlation_ratio'] = {'standard_name': 'cross_correlation_ratio_hv',
                                         'short_name': 'RHOHV',
                                         'units': ''}

    cfkeys['radar_echo_classification'] = {'standard_name': 'radar_echo_classification',
                                           'short_name': 'REC',
                                           'units': 'legend'}

    cfkeys['radar_estimated_rain_rate'] = {'standard_name': 'radar_estimated_rain_rate',
                                           'short_name': 'RRR',
                                           'units': 'mm/hr'}

    cfkeys['cloud_top_height'] = {'standard_name': 'convective_cloud_top_height',
                                  'short_name': 'CTH',
                                  'units': 'm'}

    cfkeys['0dB_echo_top_height'] = {'standard_name': '0dB_echo_top_height',
                                     'short_name': '0DB_ETH',
                                     'units': 'm'}

    cfkeys['10dB_echo_top_height'] = {'standard_name': '10dB_echo_top_height',
                                      'short_name': '10DB_ETH',
                                      'units': 'm'}

    cfkeys['17dB_echo_top_height'] = {'standard_name': '17dB_echo_top_height',
                                      'short_name': '17DB_ETH',
                                      'units': 'm'}

    cfkeys['40dB_echo_top_height'] = {'standard_name': '40dB_echo_top_height',
                                      'short_name': '40DB_ETH',
                                      'units': 'm'}

    cfkeys['steiner_echo_classification'] = {'standard_name': 'steiner_echo_classification',
                                             'long_name': 'Convetive stratiform echo classification',
                                             'units': ''}

    cfkeys['thurai_echo_classification'] = {'standard_name': 'thurai_echo_classification',
                                            'long_name': 'Convetive stratiform echo classification',
                                            'units': ''}

    return cfkeys


def read_radar(input_file):
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

    try:
        radar = pyart.io.read_grid(input_file)
        print("{} read.".format(input_file))
    except OSError:
        return None

    return radar


def write_ncfile(outfilename, time, xdim, ydim, latitude, longitude, moment, mykey_name, gnrl_meta, file_exist):
    """
    Write level 2 netCDF4 file.
    """
    cfkeys = get_key_metadata()
    try:
        moment_name = cfkeys[mykey_name]['standard_name']
    except KeyError:
        moment_name = mykey_name

    dtime = netCDF4.num2date(time['data'], time['units'])
    # Prepare CF compliant metadata
    minlon = longitude.min()
    maxlon = longitude.max()
    lonres = 0.0225
    minlat = latitude.min()
    maxlat = latitude.max()
    latres = 0.0225

    global_metadata = dict()
    global_metadata['title'] = f"CPOL L2 {moment_name.replace('_', ' ')}"
    global_metadata['summary'] = f"{moment_name.replace('_', ' ')} retrievals produced" + \
                                 " at the Australian Bureau of Meteorology on the C-Band" + \
                                 " Dual-Polarisation (CPOL) radar in Darwin, Northern Australia. "
    global_metadata['keywords'] = f"radar, {moment_name.replace('_', ' ')}"
    global_metadata['Conventions'] = "CF-1.6"
    global_metadata['uuid'] = str(uuid.uuid4())
    global_metadata['naming_authority'] = 'au.org.nci'
    global_metadata['source'] = "Processing from CPOL radar at the Bureau of Meteorology"
    global_metadata['instrument'] = "CPOL"
    global_metadata['instrument_type'] = "radar"
    global_metadata['platform_type'] = "fixed"
    global_metadata['processing_level'] = "L2"
    global_metadata['standard_name_vocabulary'] = "CF and CF/radials Metadata Convention"
    global_metadata['acknowledgement'] = "This work has been supported by the U.S. Department " + \
                                         "of Energy Atmospheric Systems Research Program through " + \
                                         "the grant DE-SC0014063. Data may be freely distributed."

    global_metadata['product_version'] = f"{datetime.date.today().year}.{datetime.date.today().month:02}"

    global_metadata['references'] = "Contact V. Louf <valentin.louf@bom.gov.au>"
    global_metadata['creator_name'] = "Valentin Louf"
    global_metadata['creator_email'] = "valentin.louf@bom.gov.au"
    global_metadata['creator_url'] = "github.com/vlouf"
    global_metadata['creator_type'] = "person for Monash University and Australian Bureau of Meteorology"
    global_metadata['institution'] = "Australian Bureau of Meteorology"
    global_metadata['project'] = "The vertical structure of convective mass-flux derived from modern " + \
                                 "radar systems - Data analysis in support of cumulus parametrization"
    global_metadata['program'] = ""
    global_metadata['publisher_name'] = "Australian Bureau of Meteorology"
    global_metadata['publisher_email'] = ""
    global_metadata['publisher_url'] = "bom.gov.au"
    global_metadata['publisher_type'] = "institution"
    global_metadata['publisher_institution'] = "Australian Bureau of Meteorology"
    global_metadata['geospatial_bounds'] = f"({minlon}, {maxlon}, {minlat}, {maxlat})"
    global_metadata['geospatial_lat_min'] = minlat
    global_metadata['geospatial_lat_max'] = maxlat
    global_metadata['geospatial_lat_units'] = "degrees_north"
    global_metadata['geospatial_lat_resolution'] = latres
    global_metadata['geospatial_lon_min'] = minlon
    global_metadata['geospatial_lon_max'] = maxlon
    global_metadata['geospatial_lon_units'] = "degrees_east"
    global_metadata['geospatial_lon_resolution'] = latres
    global_metadata['geospatial_vertical_min'] = 0
    global_metadata['geospatial_vertical_max'] = 20000
    global_metadata['geospatial_vertical_resolution'] = 500
    global_metadata['geospatial_vertical_units'] = "meters"
    global_metadata['time_coverage_start'] = dtime[0].isoformat()
    global_metadata['time_coverage_end'] = dtime[-1].isoformat()
    global_metadata['time_coverage_resolution'] = "00:10:00"
    global_metadata['date_created'] = datetime.datetime.now().isoformat()
    global_metadata['site_name'] = "Gunn_Pt"
    global_metadata['country'] = "Australia"

    try:
        global_metadata['history'] = 'cpol_level2.py;' + gnrl_meta['history']
        global_metadata['comment'] = gnrl_meta['comment']
    except KeyError:
        pass

    try:
        global_metadata['calibration'] = gnrl_meta['calibration']
    except KeyError:
        pass

    DIM_LEN = len(xdim)
    # Write netCDF4 file.
    with netCDF4.Dataset(outfilename, "w", format="NETCDF4") as ncid:
        # Create dimension
        ncid.createDimension("x", DIM_LEN)
        ncid.createDimension("y", DIM_LEN)
        ncid.createDimension('time', 144)

        # Create variables.
        mymoment = ncid.createVariable(moment_name, moment['data'].dtype, ("time", "x", "y"),
                                       zlib=True, fill_value=-9999)

        # Others variables.
        nclat = ncid.createVariable('latitude', latitude.dtype, ("x", "y"), zlib=True)
        nclon = ncid.createVariable('longitude', longitude.dtype, ("x", "y"), zlib=True)
        nctime = ncid.createVariable('time', time['data'].dtype, 'time')
        ncx = ncid.createVariable('x', xdim.dtype, 'x')
        mcy = ncid.createVariable('y', ydim.dtype, 'y')

        # Get data.
        ncquality = ncid.createVariable('qc_exist', 'i4', 'time')
        ncquality[:] = file_exist
        ncquality.units = ""
        ncquality.setncattr('standard_name', 'quality_check_measurement_exist')
        ncquality.setncattr('description', '0: no measurement available at time step, 1: data exist.')
        ncquality.setncattr('comment', 'In case of a slice, at a given timestep, full of FillValue,' +
                                       'this variable will tell you that this slice is empty because ' +
                                       'there was nothing to measure or if it is empty because the ' +
                                       'radar did not work at this timestep.')

        # Assign values.
        mymoment[:] = moment['data']
        nclat[:] = latitude
        nclon[:] = longitude
        nctime[:] = time['data']
        ncx[:] = xdim
        mcy[:] = ydim

        # Set units.
        try:
            mymoment.units = cfkeys[mykey_name]['units']
        except KeyError:
            mymoment.units = ''

        for attr_name in ['standard_name', 'short_name', 'long_name']:
            try:
                mymoment.setncattr(attr_name, cfkeys[mykey_name][attr_name])
            except KeyError:
                continue

        nclat.units = "degree_north"
        nclon.units = "degree_east"
        nclat.setncattr('standard_name', 'longitude')
        nclon.setncattr('standard_name', 'latitude')

        nctime.units = time['units']  # Global variable.
        nctime.setncattr('standard_name', 'time')

        ncx.units = "meters"
        mcy.units = "meters"
        ncx.setncattr('standard_name', 'easthward_distance')
        mcy.setncattr('standard_name', 'northward_distance')

        # # Set main metadata
        for mykey in global_metadata.keys():
            ncid.setncattr(mykey, global_metadata[mykey])

    return None

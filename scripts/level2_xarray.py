import os
import glob
import uuid
import datetime

import numpy as np
import xarray as xr
import pandas as pd

import dask
import dask.bag as db

from numba import jit
from numba import int32


@jit(nopython=True)
def _steiner_conv_strat(refl, x, y, dx, dy, intense=42, peak_relation=0,
                        area_relation=1, bkg_rad=11000, use_intense=True):
    """
    We perform the Steiner et al. (1995) algorithm for echo classification
    using only the reflectivity field in order to classify each grid point
    as either convective, stratiform or undefined. Grid points are
    classified as follows,
    0 = Undefined
    1 = Stratiform
    2 = Convective
    """
    def convective_radius(ze_bkg, area_relation):
        """
        Given a mean background reflectivity value, we determine via a step
        function what the corresponding convective radius would be.
        Higher background reflectivitives are expected to have larger
        convective influence on surrounding areas, so a larger convective
        radius would be prescribed.
        """
        if area_relation == 0:
            if ze_bkg < 30:
                conv_rad = 1000.
            elif (ze_bkg >= 30) & (ze_bkg < 35.):
                conv_rad = 2000.
            elif (ze_bkg >= 35.) & (ze_bkg < 40.):
                conv_rad = 3000.
            elif (ze_bkg >= 40.) & (ze_bkg < 45.):
                conv_rad = 4000.
            else:
                conv_rad = 5000.

        if area_relation == 1:
            if ze_bkg < 25:
                conv_rad = 1000.
            elif (ze_bkg >= 25) & (ze_bkg < 30.):
                conv_rad = 2000.
            elif (ze_bkg >= 30.) & (ze_bkg < 35.):
                conv_rad = 3000.
            elif (ze_bkg >= 35.) & (ze_bkg < 40.):
                conv_rad = 4000.
            else:
                conv_rad = 5000.

        if area_relation == 2:
            if ze_bkg < 20:
                conv_rad = 1000.
            elif (ze_bkg >= 20) & (ze_bkg < 25.):
                conv_rad = 2000.
            elif (ze_bkg >= 25.) & (ze_bkg < 30.):
                conv_rad = 3000.
            elif (ze_bkg >= 30.) & (ze_bkg < 35.):
                conv_rad = 4000.
            else:
                conv_rad = 5000.

        if area_relation == 3:
            if ze_bkg < 40:
                conv_rad = 0.
            elif (ze_bkg >= 40) & (ze_bkg < 45.):
                conv_rad = 1000.
            elif (ze_bkg >= 45.) & (ze_bkg < 50.):
                conv_rad = 2000.
            elif (ze_bkg >= 50.) & (ze_bkg < 55.):
                conv_rad = 6000.
            else:
                conv_rad = 8000.

        return conv_rad

    def peakedness(ze_bkg, peak_relation):
        """
        Given a background reflectivity value, we determine what the necessary
        peakedness (or difference) has to be between a grid point's
        reflectivity and the background reflectivity in order for that grid
        point to be labeled convective.
        """
        if peak_relation == 0:
            if ze_bkg < 0.:
                peak = 10.
            elif (ze_bkg >= 0.) and (ze_bkg < 42.43):
                peak = 10. - ze_bkg ** 2 / 180.
            else:
                peak = 0.

        elif peak_relation == 1:
            if ze_bkg < 0.:
                peak = 14.
            elif (ze_bkg >= 0.) and (ze_bkg < 42.43):
                peak = 14. - ze_bkg ** 2 / 180.
            else:
                peak = 4.

        return peak

    sclass = np.zeros(refl.shape, dtype=int32)
    ny, nx = refl.shape

    for i in range(0, nx):
        # Get stencil of x grid points within the background radius
        imin = np.max(np.array([1, (i - bkg_rad / dx)], dtype=int32))
        imax = np.min(np.array([nx, (i + bkg_rad / dx)], dtype=int32))

        for j in range(0, ny):
            # First make sure that the current grid point has not already been
            # classified. This can happen when grid points within the
            # convective radius of a previous grid point have also been
            # classified.
            if ~np.isnan(refl[j, i]) & (sclass[j, i] == 0):
                # Get stencil of y grid points within the background radius
                jmin = np.max(np.array([1, (j - bkg_rad / dy)], dtype=int32))
                jmax = np.min(np.array([ny, (j + bkg_rad / dy)], dtype=int32))

                n = 0
                sum_ze = 0

                # Calculate the mean background reflectivity for the current
                # grid point, which will be used to determine the convective
                # radius and the required peakedness.

                for l in range(imin, imax):
                    for m in range(jmin, jmax):
                        if not np.isnan(refl[m, l]):
                            rad = np.sqrt(
                                (x[l] - x[i]) ** 2 + (y[m] - y[j]) ** 2)

                        # The mean background reflectivity will first be
                        # computed in linear units, i.e. mm^6/m^3, then
                        # converted to decibel units.
                            if rad <= bkg_rad:
                                n += 1
                                sum_ze += 10. ** (refl[m, l] / 10.)

                if n == 0:
                    ze_bkg = np.inf
                else:
                    ze_bkg = 10.0 * np.log10(sum_ze / n)

                # Now get the corresponding convective radius knowing the mean
                # background reflectivity.
                conv_rad = convective_radius(ze_bkg, area_relation)

                # Now we want to investigate the points surrounding the current
                # grid point that are within the convective radius, and whether
                # they too are convective, stratiform or undefined.

                # Get stencil of x and y grid points within the convective
                # radius.
                lmin = np.max(
                    np.array([1, int(i - conv_rad / dx)], dtype=int32))
                lmax = np.min(
                    np.array([nx, int(i + conv_rad / dx)], dtype=int32))
                mmin = np.max(
                    np.array([1, int(j - conv_rad / dy)], dtype=int32))
                mmax = np.min(
                    np.array([ny, int(j + conv_rad / dy)], dtype=int32))

                if use_intense and (refl[j, i] >= intense):
                    sclass[j, i] = 2

                    for l in range(lmin, lmax):
                        for m in range(mmin, mmax):
                            if not np.isnan(refl[m, l]):
                                rad = np.sqrt(
                                    (x[l] - x[i]) ** 2
                                    + (y[m] - y[j]) ** 2)

                                if rad <= conv_rad:
                                    sclass[m, l] = 2

                else:
                    peak = peakedness(ze_bkg, peak_relation)

                    if refl[j, i] - ze_bkg >= peak:
                        sclass[j, i] = 2

                        for l in range(imin, imax):
                            for m in range(jmin, jmax):
                                if not np.isnan(refl[m, l]):
                                    rad = np.sqrt(
                                        (x[l] - x[i]) ** 2
                                        + (y[m] - y[j]) ** 2)

                                    if rad <= conv_rad:
                                        sclass[m, l] = 2

                    else:
                        # If by now the current grid point has not been
                        # classified as convective by either the intensity
                        # criteria or the peakedness criteria, then it must be
                        # stratiform.
                        sclass[j, i] = 1

    return sclass


def get_steiner(dataset):
    x = dataset.x.values
    y = dataset.x.values
    dx = np.abs(x[1] - x[0])
    dy = np.abs(y[1] - y[0])
    refl = dataset.reflectivity_gridded_Z.sel({"z": 2500}).values
    nlen = len(dataset.time)
    stein = np.zeros(refl.shape, dtype=np.int32)
    for i in range(nlen):
        stein[i, :, :] = _steiner_conv_strat(refl[i, :, :], x, y, dx, dy)

    dataset = dataset.merge({'steiner_echo_classification': (('time', 'x', 'y'), stein.astype(np.int32))})
    dataset.steiner_echo_classification.attrs = {'standard_name': 'echo_classification',
                                                'long_name': 'Steiner echo classification',
                                                'valid_min': 0,
                                                'valid_max': 2,
                                                'comment_1': ('Convective-stratiform echo '
                                                              'classification based on '
                                                              'Steiner et al. (1995)'),
                                                'comment_2': ('0 = Undefined, 1 = Stratiform, '
                                                              '2 = Convective')}

    return dataset


def read_level1b(flist):
    if len(flist) != 144:
        some_missing = xr.open_mfdataset(flist, concat_dim="time")
        some_missing = get_steiner(some_missing)
        some_missing = some_missing.merge({'isfile': (('time'), [1] * len(some_missing.time))})
        mydate = some_missing.time.to_pandas()[0].strftime("%Y-%m-%d")
        trange = pd.date_range(mydate + "T00:00:00", mydate + "T23:50:00", freq="10Min")
        ndset = xr.Dataset({'time': (('time'), trange)})
        dset = some_missing.reindex_like(ndset, method='nearest', tolerance='1Min')
    else:
        dset = xr.open_mfdataset(flist, concat_dim="time", parallel=True)
        dset = get_steiner(dset)
        dset = dset.merge({'isfile': (('time'), [1] * len(dset.time))})

    dset.steiner_echo_classification.values = np.ma.masked_invalid(dset.steiner_echo_classification.values).filled(0).astype(np.int32)
    dset.radar_echo_classification.values = np.ma.masked_invalid(dset.radar_echo_classification.values).filled(0).astype(np.int32)

    dset['isfile'].attrs['description'] = "1: if data exists at timestep; NaN: no data for this timestep"
    return dset


def process(flist):
    varlist = [("rain", "radar_estimated_rain_rate"),
           ("d0", "D0"),
           ("nw", "NW"),
           ("zdr", "corrected_differential_reflectivity"),
           ("hclass", "radar_echo_classification"),
           ("reflz", "reflectivity_gridded_Z"),
           ("refldbz", "reflectivity_gridded_dBZ"),
           ("steiner", "steiner_echo_classification")]

    dset = read_level1b(flist)
    ground_set = dset.sel({"z": 0})
    dset_2500m = dset.sel({"z": 2500})
    time = pd.to_datetime(dset.time.values)
    date = time[0].strftime('%Y%m%d')

    for shortname, myvar in varlist:
        namefile = f"twp1440cpol.{shortname}.c1.{date}.nc"
        if myvar == "radar_estimated_rain_rate":
            var_dset = ground_set[myvar].to_dataset()
            var_dset = var_dset.merge({"latitude": ground_set.point_latitude,
                                    "longitude": ground_set.point_longitude})
        else:
            var_dset = dset_2500m[myvar].to_dataset()
            var_dset = var_dset.merge({"latitude": dset_2500m.point_latitude,
                                    "longitude": dset_2500m.point_longitude})

        var_dset.attrs = dset.attrs
        var_dset.attrs['processing_level'] = 'c1'
        var_dset.attrs['title'] = 'Daily gridded radar volume on a 150x150km grid from CPOL'
        var_dset.attrs['uuid'] = str(uuid.uuid4())
        var_dset.attrs['field_names'] = myvar
        var_dset.attrs['history'] = f'created by Valentin Louf on raijin.nci.org.au at {datetime.datetime.now().isoformat()}'
        var_dset.attrs['time_coverage_start'] = time[0].isoformat()
        var_dset.attrs['time_coverage_end'] = time[-1].isoformat()
        var_dset.attrs['time_coverage_resolution'] = "00:10:00"

        try:
            var_dset[myvar].attrs.pop('coordinates')
            var_dset.attrs.pop('version')
            var_dset.attrs.pop('Conventions')
        except Exception:
            pass

        var_dset.to_netcdf(os.path.join(OUTPATH, namefile), encoding={myvar: {'zlib':False},
                                                                    "latitude": {'zlib':True},
                                                                    "longitude": {'zlib':True}})

    return None


def nbfiles(indir):
    onlyfiles = next(os.walk(indir))[2] #dir is your directory path as string
    return len(onlyfiles)


def main():
    dirlist = sorted(glob.glob("/g/data/hj10/cpol_level_1b/v2019/gridded/grid_150km_2500m/**/**/"))
    nbf = [nbfiles(d) for d in dirlist]
    missdirs = [f for n, f in zip(nbf, dirlist) if n is not None]
    flist = [sorted(glob.glob(m + "*.nc")) for m in missdirs]

    bag = db.from_sequence(flist).map(process)
    bag.compute()

    return None


if __name__ == "__main__":
    OUTPATH = "/g/data/kl02/vhl548/cpol_level2"
    main()
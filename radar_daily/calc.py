# Python Standard Library
import os

# Other libraries.
import numba
import scipy
import pyart
import crayons
import numpy as np

from numba import jit
from scipy import ndimage


@jit(nopython=True)
def _echo_top_height(x, y, z, refl, stein, altitude_work, threshold, tolerance):
    kgood = np.where(z == altitude_work)[0][0]

    nx = len(x)
    ny = len(y)
    nz = len(z)

    zerodBZ_eth = np.zeros((nx, ny))

    for j in range(ny):
        for i in range(nx):
            if stein[i, j] != 2:
                continue

            ipass = 0
            ktop = kgood
            for k in range(kgood, nz):
                if (ipass == 0) and (refl[k - 1, i, j] >= threshold) and (refl[k, i, j] <= threshold):
                    ipass = 1
                    ktop = k
                    break

            for k in range(kgood, nz):
                if (ipass == 0) and (refl[k, i, j] < threshold + tolerance):
                    ipass = 1
                    ktop = k
                    zdiff = threshold + tolerance

                    if ktop < nz - 1:
                        if np.abs(refl[ktop + 1, i, j]) < zdiff:
                            zdiff = np.abs(refl[ktop + 1, i, j])
                            ktop += 1

                    if ktop < nz - 2:
                        if np.abs(refl[ktop + 2, i, j]) < zdiff:
                            zdiff = np.abs(refl[ktop + 2, i, j])
                            ktop += 2

                    if ktop < nz - 3:
                        if np.abs(refl[ktop + 3, i, j]) < zdiff:
                            zdiff = np.abs(refl[ktop + 3, i, j])
                            ktop += 3

                    if ktop < nz - 4:
                        if np.abs(refl[ktop + 4, i, j]) < zdiff:
                            zdiff = np.abs(refl[ktop + 4, i, j])
                            ktop += 4

            zerodBZ_eth[i, j] = z[ktop]

    return zerodBZ_eth


def echo_top_height(x, y, z, refl, steiner, altitude_work=2500, threshold=17, tolerance=5):
    """
    Comput the echo top height for a certain reflectivity threshold with a
    plus/minus tolerance value.

    Parameters:
    ===========
    refl: ndarray <z, x, y>
        Reflecitivity array.
    steiner: ndarray <x, y>
        Steiner convective/stratiform classification.
    threshold: float
        Reflectivity threshold for which the echo top height is computed.
    tolerance: float
        Echo top height = z(threshold +/- tolerance)

    Returns:
    ========
    echoheight: dict
        Dictionnary containing the echo top height and its metadata.
    """
    # Check if there is convection.
    if np.sum(steiner == 2) == 0:
        print(crayons.yellow("No convection found for computing the echo top height."))
        cld_th = np.zeros_like(steiner, dtype=int)
    else:
        try:
            myrefl = refl.filled(-9999)
        except AttributeError:
            myrefl = refl.copy()
            myrefl[np.isnan(myrefl)] = -9999
        # Calling the algorithm
        cld_th = _echo_top_height(x, y, z, myrefl, steiner, altitude_work, threshold, tolerance)

    # Masking invalid values.
    cld_th = np.ma.masked_where(steiner != 2, cld_th)
    echoheight = {
        "data": cld_th,
        "standard_name": "{}dB_echo_top_height".format(threshold),
        "long_name": "{} dB echo top height".format(threshold),
        "valid_max": 20000,
        "valid_min": 0,
        "units": "meters",
    }

    return echoheight


def steiner_classification(radar, altitude=2500):
    """
    Parameters:
    ===========
        radar:
            Py-ART grid.
        altitude: float
            Work level altitude in m.

    Returns:
    ========
        eclass: dict
            Moment data fill value is FILLVALUE.
    """
    # Extract coordinates
    x = radar.x["data"]
    y = radar.y["data"]
    dx = np.abs(x[1] - x[0])
    dy = np.abs(y[1] - y[0])

    # Compute steiner classificaiton
    eclass = pyart.retrieve.steiner_conv_strat(radar, dx=dx, dy=dy, work_level=altitude, refl_field="reflectivity")
    eclass["data"] = eclass["data"].astype(np.int32)

    return eclass


def thurai_echo_classification(
    radar, work_level=2500, dbz_name="reflectivity", rain_name="radar_estimated_rain_rate", d0_name="D0", nw_name="NW"
):
    """
    Merhala Thurai's has a criteria for classifying rain either Stratiform
    Convective or Mixed, based on the D-Zero value and the log10(Nw) value.
    Merhala's rain classification is 1 for Stratiform, 2 for Convective and 3
    for Mixed, 0 if no rain.

    """
    z = radar.z["data"]
    zpos = z == work_level
    # Extracting data.
    d0 = radar.fields[d0_name]["data"][zpos, :, :]
    nw = radar.fields[nw_name]["data"][zpos, :, :]
    rainrate = radar.fields[rain_name]["data"][zpos, :, :]
    dbz = radar.fields[dbz_name]["data"][zpos, :, :]

    classification = np.zeros_like(dbz, dtype=int)

    # Invalid data
    pos0 = (d0 >= -5) & (d0 <= 100)
    pos1 = (nw >= -10) & (nw <= 100)

    # Classification index.
    indexa = nw - 6.4 + 1.7 * d0

    # Classifying
    classification[(indexa > 0.1) & (dbz > 20)] = 2
    classification[(indexa > 0.1) & (dbz <= 20)] = 1
    classification[indexa < -0.1] = 1
    classification[(indexa >= -0.1) & (indexa <= 0.1)] = 3

    # Masking invalid data.
    classification = np.ma.masked_where(~pos0 | ~pos1 | dbz.mask, classification)

    # Generate metada.
    class_meta = {
        "data": classification,
        "standard_name": "echo_classification",
        "long_name": "Merhala Thurai echo classification",
        "comment_1": "Convective-stratiform echo classification based on Merhala Thurai",
        "comment_2": "0 = Undefined, 1 = Stratiform, 2 = Convective, 3 = Mixed",
    }

    return class_meta


def cloud_top_height(pyart_grid, tvel_name="velocity_texture", dbz_name="reflectivity"):
    """
    Computing the cloud top height. Algo from Bobby Jackson.

    Parameters:
    ===========
    pyart_grid:
        Py-ART grid structure.

    Returns:
    ========
    echo_top_meta: dict
        Cloud top height.
    """
    texture = pyart_grid.fields[tvel_name]["data"]
    z = pyart_grid.fields[dbz_name]["data"]
    grid_z = pyart_grid.point_z["data"]
    grid_y = pyart_grid.point_y["data"]
    grid_x = pyart_grid.point_x["data"]

    array_shape = texture.shape
    echo_top = np.zeros((array_shape[1], array_shape[2]))
    z_values, y_values, x_values = np.meshgrid(
        range(0, array_shape[0]), range(0, array_shape[1]), range(0, array_shape[2]), indexing="ij"
    )
    labels = y_values * array_shape[2] + x_values
    in_cloud = np.ma.masked_where(np.logical_or(z.mask, texture > 3), texture)
    in_cloud[~in_cloud.mask] = labels[~in_cloud.mask]
    echo_top = ndimage.measurements.maximum(grid_z, labels=in_cloud, index=in_cloud)
    echo_top = echo_top[0, :, :]
    echo_top_meta = {
        "data": echo_top,
        "standard_name": "cloud_top_height",
        "long_name": "Cloud top height",
        "units": "meters",
    }

    return echo_top_meta


import numpy as np
import xarray.ufuncs as xu


def lon_lat_to_cartesian(lon, lat, radius=1):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius radius
    """

    # Unpack xarray object into plane arrays
    if hasattr(lon, 'data'):
        lon = lon.data
    if hasattr(lat, 'data'):
        lat = lat.data

    if lon.ndim != lat.ndim:
        raise ValueError('coordinate must share the same number of dimensions')

    if lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    lon_r = xu.radians(lon)
    lat_r = xu.radians(lat)

    x = radius * xu.cos(lat_r) * xu.cos(lon_r)
    y = radius * xu.cos(lat_r) * xu.sin(lon_r)
    z = radius * xu.sin(lat_r)

    return x.flatten(), y.flatten(), z.flatten()

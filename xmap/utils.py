
import numpy as np


def lon_lat_to_cartesian(lon, lat, radius=1):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius radius
    """

    lon, lat = np.meshgrid(lon, lat)

    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = radius * np.cos(lat_r) * np.cos(lon_r)
    y = radius * np.cos(lat_r) * np.sin(lon_r)
    z = radius * np.sin(lat_r)
    return x, y, z

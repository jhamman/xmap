import pytest
import numpy as np
import xarray as xr

import xmap

np.random.seed(3)


@pytest.fixture(scope="module")
def da_lat_lon():
    return xr.DataArray(np.random.random((10, 20)), dims=('lat', 'lon'),
                        coords={'lat': np.linspace(-90, 90, 10, endpoint=True),
                                'lon': np.linspace(0, 360, 20)})


@pytest.fixture(scope="module")
def da_time_lat_lon():
    return xr.DataArray(np.random.random((5, 10, 20)),
                        dims=('time', 'lat', 'lon'),
                        coords={'time': np.arange(5),
                                'lat': np.linspace(-90, 90, 10, endpoint=True),
                                'lon': np.linspace(0, 360, 20)})


@pytest.fixture(scope="module")
def da_target():
    return xr.DataArray(np.random.random((30, 30)), dims=('lat', 'lon'),
                        coords={'lat': np.linspace(-30, 0, 30),
                                'lon': np.linspace(10, 40, 30)})


def test_xmap_setup_lon_lat(da_lat_lon):
    assert hasattr(da_lat_lon, 'xmap')

    da_lat_lon.xmap.set_coords('lon', 'lat')
    assert da_lat_lon.xmap.xcoord == 'lon'
    assert da_lat_lon.xmap.ycoord == 'lat'
    assert da_lat_lon.xmap.tcoord is None


def test_xmap_setup_time_lon_lat(da_time_lat_lon):
    assert hasattr(da_time_lat_lon, 'xmap')
    da_time_lat_lon.xmap.set_coords('lon', 'lat', t='time')
    assert da_time_lat_lon.xmap.xcoord == 'lon'
    assert da_time_lat_lon.xmap.ycoord == 'lat'
    assert da_time_lat_lon.xmap.tcoord == 'time'


def test_remap_like(da_time_lat_lon, da_target):
    da_time_lat_lon.xmap.set_coords('lon', 'lat', t='time')
    print('da ', da_time_lat_lon)
    print('target ', da_target)
    for method, k in [('nearest', 1), ('distance_weighted', 4)]:
        print(method)
        new = da_time_lat_lon.xmap.remap_like(da_target, xcoord='lon',
                                              ycoord='lat', how=method, k=k)
        assert new.shape[1:] == da_target.shape


def test_xmap_remap_to(da_lat_lon):
    with pytest.raises(NotImplementedError):
        da_lat_lon.xmap.remap_to(None)

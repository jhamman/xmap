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


def test_has_xmap_property(da_lat_lon):
    assert hasattr(da_lat_lon, 'xmap')


def test_xmap_remap_like(da_lat_lon):
    with pytest.raises(NotImplementedError):
        print(da_lat_lon)
        da_lat_lon.xmap.remap_like(None)


def test_xmap_remap_to(da_lat_lon):
    with pytest.raises(NotImplementedError):
        da_lat_lon.xmap.remap_to(None)

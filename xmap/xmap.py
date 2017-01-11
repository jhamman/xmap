import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from .utils import lon_lat_to_cartesian

DEFAULT_QUERRY_ARGS = dict(k=1, eps=0, p=2, distance_upper_bound=np.inf)

# TODO:
# 1) don't use hard-coded coordinate names
# 2) don't assume dims
# 3) put all results in empty dataarrays that look like the target


@xr.register_dataarray_accessor('xmap')
class XMap(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._kdtree = None
        self.kdtree_options = {}

    def remap_like(self, target, how='nearest', **kwargs):
        '''Remap an xarray object to match the spatial grid of target'''

        if how == 'nearest':
            new = self._remap_nearest(target, **kwargs)
        elif how == 'bilinear':
            new = self._remap_bilinear(target, **kwargs)
        elif how == 'bicubic':
            new = self._remap_bicubic(target, **kwargs)
        elif how == 'distance_weighted':
            new = self._remap_distance_weighted(target, **kwargs)
        elif how == 'conservative':
            new = self._remap_conservative(target, **kwargs)
        elif how == 'largest_area':
            new = self._remap_largest_area(target, **kwargs)
        else:
            raise ValueError('%s is not a valid argument to how')

        return new

    def remap_to(self, target, how='nearest', **kwargs):
        '''Remap an xarray object to match the spatial grid of target'''
        raise NotImplementedError('remap_to is not yet implemented')

    @property
    def kdtree(self):
        if self._kdtree is None:
            xs, ys, zs = lon_lat_to_cartesian(self._obj['lon'].data.flatten(),
                                              self._obj['lat'].data.flatten())

            # Setup the kdtree for use later
            self._kdtree = cKDTree(zip(xs, ys, zs), **self.kdtree_options)
        return self._kdtree

    def _extract_new_dims_and_coords(self, target):
        '''get the dims and coords of the remapped object'''
        dims = ('time', 'lat', 'lon')
        coords = target.coords
        coords['time'] = self._obj['time']

        return dims, coords

    def _remap_nearest(self, target, **kwargs):
        '''nearest neighbor remapping'''
        # taken in part from
        # http://earthpy.org/interpolation_between_grids_with_ckdtree.html

        xt, yt, zt = lon_lat_to_cartesian(target['lon'].data.flatten(),
                                          target['lat'].data.flatten())

        query_kwargs = DEFAULT_QUERRY_ARGS
        query_kwargs.update(kwargs)
        if query_kwargs['k'] > 1:
            raise ValueError('Nearest neighbor remapping may only use 1 '
                             'neighbor, got k=%d' % query_kwargs['k'])

        # find indices of the nearest neighbors in the flattened array
        _, inds = self.kdtree.query(zip(xt, yt, zt), **query_kwargs)
        # get interpolated 2d field
        new = self._obj.data.flatten()[inds].reshape(target.shape)

        new_dims, new_coords = self._extract_new_dims_and_coords(target)

        return xr.DataArray(new, dims=new_dims, coords=new_coords)

    def _remap_bilinear(self, target, **kwargs):
        raise NotImplementedError()

    def _remap_bicubic(self, target, k=10, **kwargs):
        raise NotImplementedError()

    def _remap_distance_weighted(self, target, **kwargs):
        xt, yt, zt = lon_lat_to_cartesian(target['lon'].data.flatten(),
                                          target['lat'].data.flatten())

        query_kwargs = DEFAULT_QUERRY_ARGS
        query_kwargs.update(kwargs)
        d, inds = self.kdtree.query(zip(xt, yt, zt), **query_kwargs)
        w = 1.0 / d**2
        new = ((w * self._obj.data.flatten()[inds]).sum(axis=1) / (w).sum(
            axis=1)).reshape(target.shape)
        new_dims, new_coords = self._extract_new_dims_and_coords(target)
        return xr.DataArray(new, dims=new_dims, coords=new_coords)

    def _remap_conservative(self, target, order=1, **kwargs):
        raise NotImplementedError()

    def _remap_largest_area(self, target, **kwargs):
        raise NotImplementedError()

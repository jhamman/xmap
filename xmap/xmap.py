import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from .utils import lon_lat_to_cartesian

DEFAULT_QUERRY_ARGS = dict(k=1, eps=0, p=2, distance_upper_bound=np.inf)


@xr.register_dataarray_accessor('xmap')
class XMap(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._kdtree = None
        self.kdtree_options = {}

        self._shape2d = ()

    def set_coords(self, x, y, t=None):
        self.xcoord = x
        self.ycoord = y
        self.tcoord = t

        if t is None:
            self._shape2d = self._obj.shape
            self._nt = 0
        else:
            if self._obj.dims[0] != t:
                raise ValueError(
                    'current code requires the t dimension to come first')
            self._shape2d = self._obj.isel(**{t: 0}, drop=True).shape
            self._nt = len(self._obj[t])

    def remap_like(self, target, xcoord='lon', ycoord='lat', how='nearest',
                   **kwargs):
        '''Remap an xarray object to match the spatial grid of target

        Parameters
        ----------
        target : xr.DataArray
            DataArray that defines the target grid for the remap operation
        xcoord : str
            Name of x-coordinate (e.g. `lon`)
        ycoord : str
            Name of y-coordinate (e.g. `lat`)
        tcoord : str
            Name of t-coordinate (e.g. `time`), optional
        how : str
            Name of remap method to use. Valid options are: {``nearest``,
            ``distance_weighted``}.
        kwargs : dict
            Additional keyword parameters to be passed on to the remap method.
        '''

        if how == 'nearest':
            new = self._remap_nearest(target, xcoord=xcoord, ycoord=ycoord,
                                      **kwargs)
        elif how == 'bilinear':
            new = self._remap_bilinear(target, xcoord=xcoord, ycoord=ycoord,
                                       **kwargs)
        elif how == 'bicubic':
            new = self._remap_bicubic(target, xcoord=xcoord, ycoord=ycoord,
                                      **kwargs)
        elif how == 'distance_weighted':
            new = self._remap_distance_weighted(target, xcoord=xcoord,
                                                ycoord=ycoord, **kwargs)
        elif how == 'conservative':
            new = self._remap_conservative(target, xcoord=xcoord,
                                           ycoord=ycoord, **kwargs)
        elif how == 'largest_area':
            new = self._remap_largest_area(target, xcoord=xcoord,
                                           ycoord=ycoord, **kwargs)
        else:
            raise ValueError('%s is not a valid argument to how')

        return new

    def remap_to(self, target, how='nearest', **kwargs):
        '''Remap an xarray object to match the spatial grid of target'''
        raise NotImplementedError('remap_to is not yet implemented')

    @property
    def kdtree(self):
        if self._kdtree is None:
            xs, ys, zs = lon_lat_to_cartesian(self._obj[self.xcoord].data,
                                              self._obj[self.ycoord].data)

            # Setup the kdtree for use later
            self._kdtree = cKDTree(list(zip(xs, ys, zs)),
                                   **self.kdtree_options)
        return self._kdtree

    def _extract_new_dims_and_coords(self, target, xcoord, ycoord):
        '''get the dims and coords of the remapped object'''

        dims = list(target.dims)
        if self.tcoord is not None:
            if len(dims) == 2:
                dims = [self.tcoord] + dims
            else:
                dims[0] = self.tcoord

        coords = {xcoord: target.coords[xcoord],
                  ycoord: target.coords[ycoord]}

        if self.tcoord is not None:
            coords[self.tcoord] = self._obj[self.tcoord]

        return dims, coords

    def _extract_new_shape(self, target, tcoord=None):

        if tcoord is None:
            target_shape2d = target.shape
        else:
            target_shape2d = target.isel(**{tcoord: 0}).shape

        if self._nt > 0:
            new_shape = (self._nt, ) + target_shape2d
        else:
            new_shape = target_shape2d

        return target_shape2d, new_shape

    def _remap_nearest(self, target, xcoord=None, ycoord=None, tcoord=None,
                       **kwargs):
        '''nearest neighbor remapping'''
        # taken in part from
        # http://earthpy.org/interpolation_between_grids_with_ckdtree.html
        target_shape2d, new_shape = self._extract_new_shape(target,
                                                            tcoord=tcoord)

        xt, yt, zt = lon_lat_to_cartesian(target[xcoord].data,
                                          target[ycoord].data)

        query_kwargs = DEFAULT_QUERRY_ARGS.copy()
        query_kwargs.update(kwargs)
        if query_kwargs['k'] > 1:
            raise ValueError('Nearest neighbor remapping may only use 1 '
                             'neighbor, got k=%d' % query_kwargs['k'])

        # find indices of the nearest neighbors in the flattened array
        _, inds = self.kdtree.query(list(zip(xt, yt, zt)), **query_kwargs)

        if self.tcoord is None:
            new = self._obj.data.ravel()[inds].reshape(new_shape)
        else:
            ii, jj = np.unravel_index(inds, dims=self._shape2d)
            # this assumes the time axis is in the first position
            new = self._obj.data[:, ii, jj].reshape(new_shape)

        new_dims, new_coords = self._extract_new_dims_and_coords(target,
                                                                 xcoord,
                                                                 ycoord)
        print('------->', new_dims, new_coords)
        return xr.DataArray(new, dims=new_dims, coords=new_coords)

    def _remap_bilinear(self, target, xcoord=None, ycoord=None, **kwargs):
        raise NotImplementedError()

    def _remap_bicubic(self, target, xcoord=None, ycoord=None, k=10, **kwargs):
        raise NotImplementedError()

    def _remap_distance_weighted(self, target, xcoord=None, ycoord=None,
                                 tcoord=None, **kwargs):

        target_shape2d, new_shape = self._extract_new_shape(target,
                                                            tcoord=tcoord)

        xt, yt, zt = lon_lat_to_cartesian(target[xcoord].data,
                                          target[ycoord].data)

        query_kwargs = DEFAULT_QUERRY_ARGS.copy()
        query_kwargs.update(kwargs)
        if query_kwargs['k'] < 2:
            raise ValueError('Distance weighted remapping must use more than '
                             '1 neighbor, got k=%d' % query_kwargs['k'])
        # find indices of the nearest neighbors in the flattened array
        d, inds = self.kdtree.query(list(zip(xt, yt, zt)), **query_kwargs)
        w = 1.0 / d**2

        if self.tcoord is None:
            new = ((w * self._obj.data.flatten()[inds]).sum(axis=1) / (w).sum(
                axis=1)).reshape(new_shape)
        else:
            ii, jj = np.unravel_index(inds, dims=self._shape2d)
            # this assumes the time axis is in the first position
            new = ((w * self._obj.data[:, ii, jj]).sum(axis=2) / (w).sum(
                axis=1)).reshape(new_shape)

        new_dims, new_coords = self._extract_new_dims_and_coords(target,
                                                                 xcoord,
                                                                 ycoord)
        return xr.DataArray(new, dims=new_dims, coords=new_coords)

    def _remap_conservative(self, target, xcoord=None, ycoord=None, order=1,
                            **kwargs):
        raise NotImplementedError()

    def _remap_largest_area(self, target, xcoord=None, ycoord=None, **kwargs):
        raise NotImplementedError()

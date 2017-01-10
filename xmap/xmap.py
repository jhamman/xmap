import xarray as xr


@xr.register_dataarray_accessor('xmap')
class XMap(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def remap_like(self, target, how='nearest'):
        '''Remap an xarray object to match the spatial grid of target'''
        raise NotImplementedError('remap_like is not yet implemented')

    def remap_to(self, target, how='nearest'):
        '''Remap an xarray object to match the spatial grid of target'''
        raise NotImplementedError('remap_to is not yet implemented')

    def _remap_bilinear(self, target):
        raise NotImplementedError()

    def _remap_bicubic(self, target):
        raise NotImplementedError()

    def _remap_distance_weighted(self, target):
        raise NotImplementedError()

    def _remap_conservative1(self, target):
        raise NotImplementedError()

    def _remap_conservative2(self, target):
        raise NotImplementedError()

    def _remap_largest_area(self, target):
        raise NotImplementedError()

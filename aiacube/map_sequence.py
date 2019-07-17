"""
A loose collection of SunPy maps
"""
import copy

import numpy as np
import dask
import dask.array as da
import distributed
import astropy.wcs
import astropy.time
import astropy.units as u
from sunpy.map import Map
from sunpy.instr.aia import aiaprep
from sunpy.physics.differential_rotation import solar_rotate_coordinate
import ndcube
from scipy.ndimage.interpolation import shift

from aiacube.io import validate_dtype_shape, get_header, DelayedFITS

__all__ = ['AIASequence', 'derotate']


class AIASequence(ndcube.NDCubeSequence):

    @classmethod
    def from_files(cls, files, hdu=0, verify=False):
        """
        Create NDCubeSequence from a list of AIA FITS files

        Parameters
        ----------
        files : `str` or `list`
            List of FITS files to load or glob pattern
        """
        openfiles = dask.bytes.open_files(files)
        headers = cls._get_headers(openfiles, hdu)
        dtype_shape = [validate_dtype_shape(h) for h in headers]
        if not all([d == dtype_shape[0] for d in dtype_shape]):
            raise ValueError('All maps must have same shape and dtype')
        # TODO: create wcs staight from headers rather than creating intermediate map
        maps = [Map(da.from_array(DelayedFITS(f, shape=d[1], dtype=d[0], hdu=hdu, verify=verify),
                                  chunks=d[1]), h)
                for f, d, h in zip(openfiles, dtype_shape, headers)]
        return cls([ndcube.NDCube(m.data, m.wcs, meta=m.meta) for m in maps])

    @staticmethod
    def _get_headers(openfiles, hdu):
        try:
            client = distributed.get_client()
        except ValueError:
            return [get_header(f, hdu=hdu) for f in openfiles]
        else:
            futures = client.map(get_header, openfiles, hdu=hdu)
            return client.gather(futures)

    @property
    def time(self,):
        return u.Quantity([(astropy.time.Time(m.meta['t_obs'])
                            - astropy.time.Time(self.data[0].meta['t_obs'])).to(u.s) for m in self.data])

    @property
    def maps(self,):
        """
        Return a list of all cubes as `sunpy.map.Map` objects.
        """
        return [Map(m.data, m.meta) for m in self]

    def cube_to_map(self, index):
        """
        Convert a single `ndcube.NDCube` into a `sunpy.map.Map`
        """
        return Map(self[index].data, self[index].meta)

    def prep(self,):
        """
        Return an `AIASequence` in which each cube has been preppped.
        """
        maps_prep = [aiaprep(m) for m in self.maps]
        return AIASequence([ndcube.NDCube(m.data, m.wcs, meta=m.meta) for m in maps_prep])

    def derotate(self, reference_index, **kwargs):
        ref_map = self.cube_to_map(reference_index)
        maps_derot = [derotate(m, ref_map, **kwargs) for m in self.maps]
        return AIASequence([ndcube.NDCube(m.data, m.wcs, meta=m.meta) for m in maps_derot])

    def coalign(self,):
        pass

    def to_cube(self,):
        """
        Stack prepped and aligned 2D `ndcube.NDCube` objects into a single 3D `ndcube.NDCube`
        """
        data_stacked = da.stack([m.data for m in self.data])
        meta_all = {i: m.meta for i, m in enumerate(self.data)}
        wcs = self[0].wcs.to_header()
        wcs['CTYPE3'] = 'TIME'
        wcs['CUNIT3'] = 's'
        wcs['CDELT3'] = np.diff(self.time)[0].to(u.s).value
        wcs['CRPIX3'] = 0
        wcs['CRVAL3'] = 0
        wcs['NAXIS3'] = len(meta_all)
        wcs['NAXIS1'] = self[0].data.shape[1]
        wcs['NAXIS2'] = self[0].data.shape[0]
        return ndcube.NDCube(data_stacked, astropy.wcs.WCS(wcs), meta=meta_all)


def derotate(smap, ref_map, **kwargs):
    new_coord = solar_rotate_coordinate(smap.center, observer=ref_map.observer_coordinate,
                                        **kwargs)
    # Calculate shift
    x_shift = (new_coord.Tx - ref_map.center.Tx)/smap.scale.axis1
    y_shift = (new_coord.Ty - ref_map.center.Ty)/smap.scale.axis2

    # TODO: implement in Dask
    delayed_data_shifted = dask.delayed(shift)(smap.data, [y_shift.value, x_shift.value])
    data_shifted = dask.array.from_delayed(delayed_data_shifted, dtype=smap.data.dtype,
                                           shape=smap.data.shape)

    # Update metadata
    new_meta = copy.deepcopy(smap.meta)
    # TODO: Should any keywords be updated here?
    return Map(data_shifted, new_meta)

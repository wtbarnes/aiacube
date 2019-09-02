"""
A loose collection of SunPy maps
"""
import warnings
import distributed
from sunpy.map.sources import AIAMap
import ndcube

from aiacube.util import maps_to_cube, futures_to_maps, files_to_maps
from aiacube.prep import derotate

__all__ = ['AIACube']


class AIACube(ndcube.NDCube):
    """
    Container for sequential level 1.5 images for a single wavelength
    """
    @classmethod
    def from_maps(cls, maps):
        """
        Create a data cube from a list of `~sunpy.map.Map` objects
        """
        cube = maps_to_cube(maps)
        return cls(cube.data, cube.wcs, meta=cube.meta, unit=cube.unit)

    @classmethod
    def from_futures(cls, futures):
        """
        Create a data cube from a list of futures that each return a
        `~sunpy.map.Map` object in order of increasing time.

        .. warning:: It is assumed that all resulting maps are aligned and
                     sorted prior to loading them into a cube!
        """
        cube = maps_to_cube(futures_to_maps(futures))
        return cls(cube.data, cube.wcs, meta=cube.meta, unit=cube.unit)

    @classmethod
    def from_files(cls, files, **kwargs):
        """
        Create a data cube from a list of level 1.5 AIA FITS files of a
        single wavelength.

        .. warning:: It is assumed that all maps are aligned and
                     sorted prior to loading them into a cube!
        """
        cube = maps_to_cube(files_to_maps(files, **kwargs))
        return cls(cube.data, cube.wcs, meta=cube.meta, unit=cube.unit)

    @property
    def maps(self,):
        return [AIAMap(d, self.meta[i]) for i, d in enumerate(self.data)]

    def derotate(self, index, **kwargs):
        """
        Remove effect of solar rotation. Does not account for differential
        rotation.

        Parameters
        ----------
        index: `int`
            Index of the reference map to derotate to
        """
        maps = self.maps
        try:
            client = distributed.get_client()
        except ValueError:
            warnings.warn('No Dask client found. Derotating eagerly...')
            maps_derot = [derotate(m, ref_map=maps[index], **kwargs)
                          for m in maps]
        else:
            ref_map = client.scatter(maps[index])
            futures = client.map(derotate, maps, ref_map=ref_map, **kwargs)
            distributed.wait(futures)
            maps_derot = futures_to_maps(futures)

        cube = maps_to_cube(maps_derot)
        return self.__class__(cube.data, cube.wcs, meta=cube.meta,
                              unit=cube.unit)

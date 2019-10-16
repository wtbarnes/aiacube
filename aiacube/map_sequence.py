"""
Containers for storing stacked AIA images
"""
from sunpy.map.sources import AIAMap
import ndcube

from aiacube.util import maps_to_cube, futures_to_maps, files_to_maps

__all__ = ['AIACube']


class AIACube(ndcube.NDCube):
    """
    Container for sequential level 1.5 images for a single wavelength
    """
    @classmethod
    def from_maps(cls, maps, sort=True):
        """
        Create a data cube from a list of `~sunpy.map.Map` objects
        """
        cube = maps_to_cube(maps, sort=sort)
        return cls(cube.data, cube.wcs, meta=cube.meta, unit=cube.unit)

    @classmethod
    def from_futures(cls, futures, sort=True):
        """
        Create a data cube from a list of futures that each return a
        `~sunpy.map.Map` object in order of increasing time.

        .. warning:: It is assumed that all resulting maps are aligned
                     prior to loading them into a cube!
        """
        cube = maps_to_cube(futures_to_maps(futures), sort=sort)
        return cls(cube.data, cube.wcs, meta=cube.meta, unit=cube.unit)

    @classmethod
    def from_files(cls, files, sort=True, **kwargs):
        """
        Create a data cube from a list of level 1.5 AIA FITS files of a
        single wavelength.

        .. warning:: It is assumed that all maps are aligned prior to loading
                     them into a cube!
        """
        cube = maps_to_cube(files_to_maps(files, **kwargs), sort=sort)
        return cls(cube.data, cube.wcs, meta=cube.meta, unit=cube.unit)

    @property
    def maps(self,):
        return [AIAMap(d, self.meta[i]) for i, d in enumerate(self.data)]

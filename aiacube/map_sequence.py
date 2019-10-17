"""
Containers for storing stacked AIA images
"""
import astropy.units as u
import astropy.wcs
from sunpy.map.sources import AIAMap
import dask.array
import ndcube
import zarr

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

    @u.quantity_input
    @classmethod
    def from_zarr(cls, url, wavelength: u.angstrom):
        """
        Load a level 2 data cube for a given wavelength from a Zarr dataset

        Parameters
        ----------
        url: `str`
        wavelength: `~astropy.units.Quantity`

        Returns
        -------
        cube : AIACube
        """
        group = f'{wavelength.to(u.angstrom).value:.0f}'
        data = dask.array.from_zarr(url, component=group)
        z = zarr.open(url, mode='r')
        wcs = z[group].attrs['wcs']
        for i in range(data.ndim):
            wcs[f'NAXIS{data.ndim - i}'] = data.shape[i]
        wcs = astropy.wcs.WCS(wcs)
        meta = z[group].attrs['meta']
        meta = {int(k): v for k, v in meta.items()}
        unit = meta[0]['bunit'] if 'bunit' in meta[0] else None
        return cls(data, wcs, meta=meta, unit=unit)

    @property
    def maps(self,):
        return [AIAMap(d, self.meta[i]) for i, d in enumerate(self.data)]

    @property
    def wavelength(self):
        # All wavelengths should be the same in a cube
        return self.maps[0].wavelength

    def to_zarr(self, url):
        """
        Store data cube as a Zarr dataset
        """
        group = f'{self.wavelength.to(u.angstrom).value:.0f}'
        dask.array.to_zarr(self.data, url, component=group)
        z = zarr.open(url, mode='a')
        z[group].attrs['wcs'] = dict(self.wcs.to_header())
        z[group].attrs['meta'] = self.meta

"""
Utilities for building data cubes
"""
import numpy as np
import dask
import dask.array as da
import distributed
import astropy.units as u
import astropy.time
import astropy.wcs
from sunpy.map import Map
import ndcube

from aiacube.io import validate_dtype_shape, get_header, DelayedFITS

__all__ = ['make_time_wcs', 'files_to_maps', 'futures_to_maps', 'maps_to_cube',
           'files_to_cube', 'futures_to_cube']


def get_headers(openfiles, hdu):
    """
    Read in headers only from FITS files. Use Dask if a
    client is available
    """
    try:
        client = distributed.get_client()
    except ValueError:
        return [get_header(f, hdu=hdu) for f in openfiles]
    else:
        futures = client.map(get_header, openfiles, hdu=hdu)
        return client.gather(futures)


def make_time_wcs(smap_ref, time):
    wcs = smap_ref.wcs.to_header()
    wcs['CTYPE3'] = 'TIME'
    wcs['CUNIT3'] = 's'
    wcs['CDELT3'] = np.diff(time)[0].to(u.s).value
    wcs['CRPIX3'] = 1
    wcs['CRVAL3'] = time[0].to(u.s).value
    wcs['NAXIS3'] = time.shape[0]
    wcs['NAXIS1'] = smap_ref.data.shape[1]
    wcs['NAXIS2'] = smap_ref.data.shape[0]
    return astropy.wcs.WCS(wcs)


def files_to_maps(files, hdu=0, verify=False):
    openfiles = dask.bytes.open_files(files)
    headers = get_headers(openfiles, hdu)
    dtype_shape = [validate_dtype_shape(h) for h in headers]
    if not all([d == dtype_shape[0] for d in dtype_shape]):
        raise ValueError('All maps must have same shape and dtype')
    # TODO: create wcs staight from headers
    arrays = [da.from_array(DelayedFITS(f, shape=s, dtype=d, hdu=hdu,
                                        verify=verify), chunks=s)
              for f, (d, s) in zip(openfiles, dtype_shape,)]
    return [Map(a, h) for a, h in zip(arrays, headers)]


def futures_to_maps(futures):
    """
    Create a list of `~sunpy.map.Map` objects from a list of
    `~distributed.Future` objects which return `~sunpy.map.Map`
    objects.

    The reasoning behind this function is that if we have
    `~distributed.Future` objects that return `sunpy.map.Map`
    objects, we only want to bring the header into local memory
    and keep the image data in the memory of the cluster. Thus,
    this function separates the data from the header and rebuilds a
    `sunpy.map.Map` with a local header and data represented by a
    `dask.array.Array`.
    """
    # NOTE: If you don't have a client, you don't have futures
    client = distributed.get_client()
    # Bring all header info into local memory
    fheaders = client.map(lambda x: x.meta, futures, pure=True)
    headers = client.gather(fheaders)
    dtype_shape = [validate_dtype_shape(h) for h in headers]
    # Push image data into cluster and representative as Dask array
    farrays = client.map(lambda x: x.data, futures, pure=True)
    arrays = [da.from_delayed(fa, s, dtype=d)
              for fa, (d, s) in zip(farrays, dtype_shape)]

    return [Map(a, h) for a, h in zip(arrays, headers)]


def maps_to_cube(maps):
    """
    Create an `~ndcube.NDCube` with a time axis from a list of
    `~sunpy.map.Map` objects. It is assumed that the maps are
    sorted with increasing time.
    """
    data_stacked = da.stack([m.data for m in maps])
    meta_all = {i: m.meta for i, m in enumerate(maps)}
    t0 = astropy.time.Time(maps[0].meta['t_obs'])
    time = u.Quantity([(astropy.time.Time(m.meta['t_obs']) - t0).to(u.s)
                       for m in maps])
    return ndcube.NDCube(data_stacked, make_time_wcs(maps[0], time),
                         meta=meta_all)


def files_to_cube(files, **kwargs):
    """
    Create lazily-loaded NDCube backed by a Dask array from a list of files.
    It is assumed that all maps are aligned prior to loading them into a cube.
    """
    return maps_to_cube(files_to_maps(files, **kwargs))


def futures_to_cube(futures):
    """
    Wrapper function for creating `ndcube.NDCube` from `distributed.Future`
    objects.
    """
    return maps_to_cube(futures_to_maps(futures))

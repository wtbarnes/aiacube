"""
Utilities for building data cubes
"""
import numpy as np
import dask
import dask.bytes
import dask.array as da
import distributed
import astropy.units as u
import astropy.time
import astropy.wcs
from sunpy.map import Map
import ndcube

from aiacube.io import validate_dtype_shape, get_header, DelayedFITS

__all__ = ['make_time_wcs', 'files_to_maps', 'futures_to_maps', 'maps_to_cube']


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


def files_to_maps(files, keep_in_memory=False, hdu=0, verify=False):
    """
    Create a list of `~sunpy.map.Map` objects from a list of FITS files.
    
    Parameters
    ----------
    files : `list`
        List of FITS files
    keep_in_memory : `bool`, optional
        If True, keep the image data in the memory of the cluster
    hdu : `int`, optional
    verify : `bool`, optional
        If True, try to fix the FITS header before loading it in.
    """
    if keep_in_memory:
        client = distributed.get_client()
        futures = client.map(Map, files, pure=True)
        return futures_to_maps(futures)

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
    # Bug in Dask that mistakenly casts the stacked array to
    # type 'O' even when all are float64
    if not all([m.data.dtype == maps[0].data.dtype for m in maps]):
        raise ValueError('All darrays must have same data type')
    data_stacked = data_stacked.astype(maps[0].data.dtype)
    meta_all = {i: m.meta for i, m in enumerate(maps)}
    t0 = astropy.time.Time(maps[0].meta['t_obs'])
    time = u.Quantity([(astropy.time.Time(m.meta['t_obs']) - t0).to(u.s)
                       for m in maps])
    return ndcube.NDCube(data_stacked,
                         make_time_wcs(maps[0], time),
                         meta=meta_all,
                         unit=maps[0].meta.get('bunit'))

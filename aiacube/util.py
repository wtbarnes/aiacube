"""
Utilities for building data cubes
"""
import numpy as np
import dask.array as da
import distributed
import astropy.units as u
import astropy.time
import astropy.wcs
import sunpy.map
import ndcube

import aiacube


def futures_to_maps(futures):
    """
    Create a list of `~sunpy.map.Map` objects from a list of `~distributed.Future` objects
    """
    client = distributed.get_client()
    # Gather all headers
    fheaders = client.map(lambda x: x.meta, futures, pure=True)
    headers = client.gather(fheaders)
    # Get dtype and shape from headers
    dtype_shape = [aiacube.io.validate_dtype_shape(h) for h in headers]
    # Map only arrays into cluster
    farrays = client.map(lambda x: x.data, futures, pure=True)
    # Create array collections from them
    arrays = [da.from_delayed(fa, s, dtype=d) for fa, (d, s) in zip(farrays, dtype_shape)]

    return [sunpy.map.Map(a, h) for a, h in zip(arrays, headers)]


def maps_to_cube(maps):
    """
    Create an `~ndcube.NDCube` from a list of `~sunpy.map.Map` objects
    """
    # Stack all of the arrays
    data_stacked = da.stack([m.data for m in maps])
    # Collect metadata
    meta_all = {i: m.meta for i, m in enumerate(maps)}
    time = u.Quantity([(astropy.time.Time(m.meta['t_obs'])
                        - astropy.time.Time(maps[0].meta['t_obs'])).to(u.s) for m in maps])
    # Create the WCS
    wcs = maps[0].wcs.to_header()
    wcs['CTYPE3'] = 'TIME'
    wcs['CUNIT3'] = 's'
    wcs['CDELT3'] = np.diff(time)[0].to(u.s).value
    wcs['CRPIX3'] = 1
    wcs['CRVAL3'] = 0
    wcs['NAXIS3'] = len(meta_all)
    wcs['NAXIS1'] = maps[0].data.shape[1]
    wcs['NAXIS2'] = maps[0].data.shape[0]

    return ndcube.NDCube(data_stacked, astropy.wcs.WCS(wcs), meta=meta_all)

"""
Functions for converting level 1 to level 1.5 data
and derotating
"""
import copy

import numpy as np
from scipy.ndimage import shift
import distributed
import astropy.units as u
from sunpy.map.header_helper import get_observer_meta
from sunpy.physics.differential_rotation import solar_rotate_coordinate

__all__ = ['normalize_to_exposure_time', 'derotate', 'aiaprep']


def register_and_derotate(maps, ref_index=0):
    """
    Process images to level 1.5, normalize to the exposure time, and derotate
    to a reference observer.

    Parameters
    ----------
    maps: `list`
        List of `distributed.Future` objects that each return a
        `~sunpy.map.Map`. These should be in sequential order and
        for a single wavelength.
    ref_index: `int`
        Index of the map to use as the reference
    """
    client = distributed.get_client()
    ref_map = client.scatter(aiaprep(client.gather(maps[ref_index])))
    lvl_15_maps = []
    # Sequential calls to submit to avoid keeping intermediate steps in
    # memory; This may stress the scheduler though...
    for m in maps:
        m_reg = client.submit(aiaprep, m, pure=True, missing=0.0)
        m_norm = client.submit(normalize_to_exposure_time, m_reg, pure=True)
        m_derot = client.submit(derotate, m_norm, ref_map=ref_map, pure=True)
        lvl_15_maps.append(m_derot)

    return lvl_15_maps


def normalize_to_exposure_time(smap, default_exposure_time=2.9 * u.s):
    """
    Remove this once it an equivalent function is merged into aiapy
    """
    if smap.exposure_time.value != 0.0:
        exp_time = smap.exposure_time.to(u.s).value
    else:
        exp_time = default_exposure_time.to(u.s).value
    new_meta = copy.deepcopy(smap.meta)
    new_meta['bunit'] = 'ct / pixel / s'
    return smap._new_instance(smap.data / exp_time, new_meta)


def derotate(smap, ref_map=None, rot_type='snodgrass'):
    # NOTE: Alternatively, we could use the
    # `~sunpy.physics.differential_rotate` but it is considerably slower.
    # NOTE: This is a kwarg so that it plays nicely with client.map
    if ref_map is None:
        raise ValueError('Must provide a reference map.')
    new_coord = solar_rotate_coordinate(
        smap.center, observer=ref_map.observer_coordinate, rot_type=rot_type)
    # Calculate shift
    x_shift = (new_coord.Tx - ref_map.center.Tx)/smap.scale.axis1
    y_shift = (new_coord.Ty - ref_map.center.Ty)/smap.scale.axis2
    # TODO: implement in Dask
    data_shifted = shift(smap.data, [y_shift.value, x_shift.value])
    # Update metadata
    new_meta = copy.deepcopy(smap.meta)
    if new_meta.get('date_obs', False):
        del new_meta['date_obs']
    new_meta['date-obs'] = ref_map.observer_coordinate.obstime.strftime(
        "%Y-%m-%dT%H:%M:%S.%f")
    new_meta.update(get_observer_meta(ref_map.observer_coordinate,
                                      new_meta['rsun_ref'] * u.m))

    return smap._new_instance(data_shifted, new_meta, smap.plot_settings)


def aiaprep(aiamap, order=3, use_scipy=False, missing=None):
    """
    Remove this once it an equivalent function is merged into aiapy
    """
    # Target scale is 0.6 arcsec/pixel, but this needs to be adjusted if the
    # map has already been rescaled.
    if ((aiamap.scale[0] / 0.6).round() != 1.0 * u.arcsec / u.pix
            and aiamap.data.shape != (4096, 4096)):
        scale = (aiamap.scale[0] / 0.6).round() * 0.6 * u.arcsec
    else:
        scale = 0.6 * u.arcsec
    scale_factor = aiamap.scale[0] / scale

    if missing is None:
        missing = aiamap.min()

    tempmap = aiamap.rotate(recenter=True,
                            scale=scale_factor.value,
                            missing=missing,
                            order=order,
                            use_scipy=use_scipy,)

    # extract center from padded aiamap.rotate output
    # crpix1 and crpix2 will be equal (recenter=True), as aiaprep does not
    # work with submaps
    center = np.floor(tempmap.meta['crpix1'])
    range_side = (center
                  + np.array([-1, 1]) * aiamap.data.shape[0] / 2) * u.pix
    newmap = tempmap.submap(u.Quantity([range_side[0], range_side[0]]),
                            u.Quantity([range_side[1], range_side[1]]))

    newmap.meta['r_sun'] = newmap.meta['rsun_obs'] / newmap.meta['cdelt1']
    newmap.meta['lvl_num'] = 1.5
    newmap.meta['bitpix'] = -64
    newmap.meta['bunit'] = 'ct / pixel'

    return newmap

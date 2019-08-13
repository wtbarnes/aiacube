"""
Functions for converting level 1 to level 1.5 data
and derotating
"""
import copy

from scipy.ndimage import shift
import distributed
import astropy.units as u
from sunpy.instr.aia import aiaprep
from sunpy.map.header_helper import get_observer_meta
from sunpy.physics.differential_rotation import solar_rotate_coordinate

from aiacube.util import futures_to_maps

__all__ = ['prep_and_normalize', 'normalize_to_exposure_time', 'derotate']


def prep_and_normalize(maps):
    """
    Convert level 1 maps to level 1.5 maps by aligning solar north with the
    vertical axis of the image, aligning the sun center and the center
    of the image, and scaling to a common resolution. Also normalizes by
    the exposure time such that the units of the map are DN pix-1 s-1
    """
    try:
        client = distributed.get_client()
    except ValueError:
        return [normalize_to_exposure_time(aiaprep(m)) for m in maps]
    else:
        maps_prep = client.map(aiaprep, maps, pure=True)
        maps_norm = client.map(normalize_to_exposure_time, maps_prep,
                               pure=True)
        # NOTE: This returns maps which are in the memory of the cluster
        return futures_to_maps(maps_norm)


def normalize_to_exposure_time(smap, default_exposure_time=2.9 * u.s):
    if smap.exposure_time.value != 0.0:
        exp_time = smap.exposure_time.to(u.s).value
    else:
        exp_time = default_exposure_time.to(u.s).value
    return smap._new_instance(smap.data / exp_time, smap.meta)


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

"""
Functions for computing time lag and cross-correlation on lazily-loaded AIA data cubes
"""
import copy

import numpy as np
import dask.array as da
from sunpy.map import GenericMap

__all__ = ['get_lags', 'cross_correlation', 'peak_cross_correlation_map',
           'time_lag_map']


def get_lags(time):
    delta_t = np.diff(time.value).cumsum()
    return np.hstack([-delta_t[::-1], np.array([0]), delta_t]) * time.unit


def cross_correlation(ndcube_a, ndcube_b, lags, **kwargs):
    """
    Lazily compute cross-correlation in each pixel of an AIA map
    """
    # Shape must be the same in spatial direction
    chunks = kwargs.get('chunks', (ndcube_a.data.shape[1]//10,
                                   ndcube_a.data.shape[2]//10))
    cube_a = ndcube_a.data.rechunk(ndcube_a.data.shape[:1]+chunks)
    cube_b = ndcube_b.data.rechunk(ndcube_b.data.shape[:1]+chunks)
    #if self.needs_interpolation:
    #    cube_a = self._interpolate(self[channel_a].time, cube_a)
    #    cube_b = self._interpolate(self[channel_b].time, cube_b)
    # Reverse the first timeseries
    cube_a = cube_a[::-1, :, :]
    # Normalize by mean and standard deviation
    std_a = cube_a.std(axis=0)
    std_a = da.where(std_a == 0, 1, std_a)
    v_a = (cube_a - cube_a.mean(axis=0)[np.newaxis, :, :]) / std_a[np.newaxis, :, :]
    std_b = cube_b.std(axis=0)
    std_b = da.where(std_b == 0, 1, std_b)
    v_b = (cube_b - cube_b.mean(axis=0)[np.newaxis, :, :]) / std_b[np.newaxis, :, :]
    # FFT of both channels
    fft_a = da.fft.rfft(v_a, axis=0, n=lags.shape[0])
    fft_b = da.fft.rfft(v_b, axis=0, n=lags.shape[0])
    # Inverse of product of FFTS to get cross-correlation (by convolution theorem)
    cc = da.fft.irfft(fft_a * fft_b, axis=0, n=lags.shape[0])
    # Normalize by the length of the timeseries
    return cc / cube_a.shape[0]


def peak_cross_correlation_map(ndcube_a, ndcube_b, lags, **kwargs):
    """
    Construct map of peak cross-correlation between two channels in each pixel of
    an AIA map.
    """
    cc = cross_correlation(ndcube_a, ndcube_b, lags, **kwargs)
    bounds = kwargs.get('timelag_bounds', None)
    if bounds is not None:
        indices, = np.where(np.logical_and(lags >= bounds[0], lags <= bounds[1]))
        start = indices[0]
        stop = indices[-1] + 1
    else:
        start = 0
        stop = lags.shape[0] + 1
    max_cc = cc[start:stop, :, :].max(axis=0)
    meta = copy.deepcopy(ndcube_a.meta[0])
    del meta['instrume']
    del meta['t_obs']
    del meta['wavelnth']
    meta['bunit'] = ''
    meta['comment'] = f'{ndcube_a.meta[0]["wavelnth"]}-{ndcube_b.meta[0]["wavelnth"]} cross-correlation'
    plot_settings = {'cmap': 'plasma'}
    plot_settings.update(kwargs.get('plot_settings', {}))
    correlation_map = GenericMap(max_cc, meta, plot_settings=plot_settings)

    return correlation_map


def time_lag_map(ndcube_a, ndcube_b, lags, **kwargs):
    """
    Construct map of timelag values that maximize the cross-correlation between
    two channels in each pixel of an AIA map.
    """
    cc = cross_correlation(ndcube_a, ndcube_b, lags, **kwargs)
    bounds = kwargs.get('timelag_bounds', None)
    if bounds is not None:
        indices, = np.where(np.logical_and(lags >= bounds[0], lags <= bounds[1]))
        start = indices[0]
        stop = indices[-1] + 1
    else:
        start = 0
        stop = lags.shape[0] + 1
    i_max_cc = cc[start:stop, :, :].argmax(axis=0)
    max_timelag = da.from_array(lags[start:stop])[i_max_cc.flatten()].reshape(i_max_cc.shape)
    meta = copy.deepcopy(ndcube_a.meta[0])
    del meta['instrume']
    del meta['t_obs']
    del meta['wavelnth']
    meta['bunit'] = 's'
    meta['comment'] = f'{ndcube_a.meta[0]["wavelnth"]}-{ndcube_b.meta[0]["wavelnth"]} time lag'
    plot_settings = {
        'cmap': 'RdBu_r',
        'vmin': lags[start:stop].value.min(),
        'vmax': lags[start:stop].value.max(),
    }
    plot_settings.update(kwargs.get('plot_settings', {}))
    time_lag_map = GenericMap(max_timelag, meta.copy(), plot_settings=plot_settings.copy())

    return time_lag_map

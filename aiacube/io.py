"""
Low-level functions and classes for lazily reading FITS data
"""
from astropy.io import fits
from astropy.io.fits.hdu.base import BITPIX2DTYPE
from sunpy.util.metadata import MetaDict
import sunpy.io.fits

__all__ = ['validate_dtype_shape', 'get_header', 'DelayedFITS']


def validate_dtype_shape(head):
    naxes = head['NAXIS']
    dtype = BITPIX2DTYPE[head['BITPIX']]
    shape = [head[f'NAXIS{n}'] for n in range(naxes, 0, -1)]
    return dtype, shape


def get_header(fn, hdu=0):
    with fn as fi:
        return MetaDict(sunpy.io.fits.get_header(fi)[hdu])


class DelayedFITS:
    def __init__(self, file, shape, dtype, hdu=0, verify=False):
        self.shape = shape
        self.dtype = dtype
        self.file = file
        self.hdu = hdu
        self.verify = verify

    def __getitem__(self, item):
        with self.file as fi:
            with fits.open(fi, memmap=True) as hdul:
                if self.verify:
                    hdul.verify('silentfix+warn')
                return hdul[self.hdu].data[item]

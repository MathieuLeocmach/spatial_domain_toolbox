"""Author: Gunnar Farnebäck
        Computer Vision Laboratory
        Linköping University, Sweden
        gf@isy.liu.se

Converted to Python by Mathieu Leocmach
"""

import warnings
import math
import numpy as np

def gaussian_app(size, dims, sigma=None):
    """Construct a Gaussian applicability.

size: Size of the neighborhood along each dimension, should be odd.

dims: Number of signal dimensions, must be between 1 and 4.

sigma: Standard deviation of the Gaussian. Default for a cutoff at 0.005.
"""
    assert int(size) == size, 'size should be an (odd) integer'
    if size%2==0:
        warnings.warn('side should be odd, increased it by one.')
        size +=1

    if size == 1:
        return np.array(1).reshape((1)*dims)
    n = int((size-1)//2)

    if sigma is None:
        delta = 0.005
        sigma = n/math.sqrt(-2*log(delta))

    I = np.arange(-n, n+1)**2

    r2 = np.zeros((size,)*dims)
    for dim in range(dims):
        r2 += I.reshape((size,)+(1,)*dim)


    if sigma != 0:
        return np.exp(-r2/(2*sigma**2))
    else:
        return (r2==0).astype(np.float64)

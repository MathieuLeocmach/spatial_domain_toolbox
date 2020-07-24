"""Author: Gunnar Farnebäck
        Computer Vision Laboratory
        Linköping University, Sweden
        gf@isy.liu.se

Converted to Python by Mathieu Leocmach
"""
import numpy as np
from scipy.linalg import lstsq
from numba import jit

def min_quadform(Qtot):
    """Minimize quadratic forms according to equations 6.20 -- 6.24 in Gunnar
Farnebäck's thesis "Polynomial Expansion for Orientation and Motion	Estimation".

Qtot: A collection of quadratic forms, having the size (HEIGTH x WIDTH x ... x N x N.

----
Returns

P: A collection of optimal parameters, having the size HEIGHT x WIDTH x ... x (N-1).
"""
    assert Qtot.shape[-2] == Qtot.shape[-2], "Qtot must be HEIGHT x WIDTH x n x n."
    shape = Qtot.shape[:-2]
    M = Qtot.shape[-1] -1
    #Create the output array.
    params = np.zeros(shape+(M,))
    Q = Qtot[...,:M,:M]
    q = -Qtot[...,:M,M]

    for index in np.ndindex(shape):
        #compute least square coefficients so that |Q*params -q| is minimized
        params[index] = lstsq(Q[index], q[index])[0]
    return params

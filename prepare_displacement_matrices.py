"""Author: Gunnar Farnebäck
        Computer Vision Laboratory
        Linköping University, Sweden
        gf@isy.liu.se

Converted to Python by Mathieu Leocmach
"""
import numpy as np
from numba import jit

@jit(nopython=True)
def prepare_displacement_matrices(A1, b1, A2, b2, displacement=None):
    """Compute matrices used for displacement estimation as defined by equations
(7.32) and (7.33) in Gunnar Farnebäck's thesis "Polynomial Expansion for
Orientation and Motion Estimation".
"""
    shape = A1.shape[:-2]
    N = A1.shape[-1]
    if displacement is None:
        displacement = np.zeros(shape + (N,))
    #flatten all spatial dimensions of A2 and b2
    dimprod = 1
    for s in shape:
        dimprod *= s
    A22 = A2.reshape((dimprod, N, N))
    b22 = b2.reshape((dimprod, N))
    #prepare strides to be able to flatten indices
    strides = (np.array(A2.strides[:-2]) / (N*N*A2.itemsize)).astype(np.int64)

    A = np.zeros(A1.shape)
    Delta_b = np.zeros(b1.shape)
    # If displacement is zero, we will get A = (A1+A2)/2 and b = -(b2-b1)/2.
    for index in np.ndindex(shape):
        #truncate the rounded displacement so that no pixel goes out
        d = np.floor(0.5 + displacement[index]).astype(np.int64)
        for dim in range(N):
            d[dim] = min(max(d[dim], -index[dim]), shape[dim] -index[dim] -1)
        #flatten advected index
        index2 = 0
        for dim in range(N):
            index2 += (d[dim] + index[dim]) * strides[dim]

        # advected average of the two A matrices (Eq. 7.32)
        A[index] = (A1[index] + A22[index2]) / 2
        # advected difference of the two vectors b (Eq. 7.33)
        df = d.astype(A.dtype)
        bb2 = b22[index2] - 2 * A[index] @ df
        Delta_b[index] = -(bb2 - b1[index]) / 2
    return  A, Delta_b

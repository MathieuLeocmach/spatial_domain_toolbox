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

A1,b1: Local polynomial expension coefficients at time 1. A1 is a N+2
dimensional array, where the first N indices indicates the position in the
signal and the last two contains the matrix for each point. In the same way, b1
is a N+1 dimensional array. Such arrays can be obtained via `make_Abc_fast`

A2,b2: Local polynomial expension coefficients at time 2.

displacement: Initial guess of the displacement field.

----
Returns

A: Advected average of A1 and A2 matrices (Eq. 7.32)

Delta_b: advected difference of b2 and b1 (Eq. 7.33)
"""
    shape = A1.shape[:-2]
    N = A1.shape[-1]
    if displacement is None:
        displacement = np.zeros(shape + (N,), dtype=A1.dtype)
    #flatten all spatial dimensions of A2 and b2
    dimprod = 1
    for s in shape:
        dimprod *= s
    A22 = A2.reshape((dimprod, N, N))
    b22 = b2.reshape((dimprod, N))
    #prepare strides to be able to flatten indices
    strides = (np.array(A2.strides[:-2]) / (N*N*A2.itemsize)).astype(np.int64)

    A = np.zeros(A1.shape, dtype=A1.dtype)
    Delta_b = np.zeros(b1.shape, dtype=A1.dtype)
    # If displacement is zero, we will get A = (A1+A2)/2 and b = -(b2-b1)/2.
    for index in np.ndindex(shape):
        #displacement is given fast index first, so we revert
        #we also take the opposite in order to be able to bring next image
        #on previous image
        #truncate the rounded displacement so that no pixel goes out
        d = np.floor(0.5 + displacement[index]).astype(np.int64)
        for dim in range(N):
            if index[dim] + d[dim] <0:
                d[dim] = -index[dim]
            elif index[dim] + d[dim] >= shape[dim]:
                d[dim] = shape[dim]-index[dim]-1
        #flatten advected index
        index2 = 0
        for dim in range(N):
            index2 += (d[dim] + index[dim]) * strides[dim]

        # advected average of the two A matrices (Eq. 7.32)
        A[index] = (A1[index] + A22[index2]) / 2
        # advected difference of the two vectors b (Eq. 7.33)
        df = d.astype(A.dtype)
        Delta_b[index] = -0.5*(b22[index2] - b1[index]) + A[index] @ df
    return  A, Delta_b

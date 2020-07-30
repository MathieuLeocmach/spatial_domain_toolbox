"""Author: Gunnar Farnebäck
        Computer Vision Laboratory
        Linköping University, Sweden
        gf@isy.liu.se

Converted to Python by Mathieu Leocmach
"""

import numpy as np

from make_Abc_fast import conv_results2A, conv_results2b, conv_results2c, make_Abc_fast
from prepare_displacement_matrices import prepare_displacement_matrices

def get_border(shape, width):
    """Make a mask of the borders for an image of given shape and a given
    border width"""
    border = np.zeros(shape, bool)
    for dim in range(border.ndim):
        border[(slice(None),)*dim + (slice(None,width),)] = True
        border[(slice(None),)*dim + (slice(-width),)] = True
    return border

def estimate_displacement(im1, im2, kernelsizes1, kernelsizes2, model, method, d0=None):
    """Estimate displacement according to the algorithms described in chapter 7
of Gunnar Farnebäck's thesis "Polynomial Expansion for Orientation and Motion
Estimation".

im1, im2: Two grayscale images of the same size.

kernelsizes1: Vector of kernelsizes used for polynomial expansion in each
iteration of the displacement computations. The same kernelsize can be repeated.

kernelsizes2: Vector of kernelsizes used for averaging each iteration of the
displacement computations. Must have the same length as kernelsizes1.

model: Choice of parametric motion model, 'constant', 'affine', or 'eightparam'.

method: 'fast' or 'accurate'

d0: A priori displacement estimate. Default is an all-zero displacement field.

----
Returns

d: Estimated displacement field.

c: Estimated (reversed) confidence value. Small values indicate more reliable
displacement estimates.
    """
    assert len(kernelsizes1) == len(kernelsizes2), 'kernelsizes1 and kernelsizes2 must have the same length'
    if d0 is None:
        d0 = np.zeros(im1.shape+(2,))
    d = d0

    for k, (kernelsize1, kernelsize2) in enumerate(zip(kernelsizes1, kernelsizes2)):
        #ensure that matrices A1, b1, A2, b2 are computed only if the kernel size changes
        if k ==0:
            last_kernelsize1 = -1
        else:
            last_kernelsize1 = kernelsizes1[k - 1]
        if kernelsize1 != last_kernelsize1:
            if method == 'fast':
                A1, b1, c1 = make_Abc_fast(im1, kernelsize1)
                A2, b2, c2 = make_Abc_fast(im2, kernelsize2)
                #Ad hoc deweighting of expansion coefficients close to the border.
                cin = np.ones(im1.shape)
                half = int(kernelsize1//2)
                border = get_border(cin.shape, half)
                cin[border] *= 0.05
                border = get_border(cin.shape, int(half//2))
                cin[border] *= 0.05
            else:
                # Ad hoc deweighting of expansion coefficients close to theborder.
                cin = np.ones(im1.shape)
                half = int(kernelsize1//2)
                border = get_border(cin.shape, half)
                cin[border] *= 0.2
                border = get_border(cin.shape, int(half//2))
                cin[border] *= 0.1

                r1 = polyexp(im1, cin, 'quadratic', kernelsize1)
                A1 = conv_results2A(r1)
                b1 = conv_results2b(r1)

                r2 = polyexp(im2, cin, 'quadratic', kernelsize1)
                A2 = conv_results2A(r2)
                b2 = conv_results2b(r2)
        # update the displacement field
        d0 = d
        sigma = 0.15 * (kernelsize2 - 1)
        A, b = prepare_displacement_matrices(A1, b1, A2, b2, d0)
        d, c = compute_displacement(A, b, kernelsize2, sigma, cin, model)
    return d, c

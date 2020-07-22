"""Author: Gunnar Farnebäck
        Computer Vision Laboratory
        Linköping University, Sweden
        gf@isy.liu.se

Converted to Python by Mathieu Leocmach
"""

import numpy as np

def prepare_displacement_matrices(A1, b1, A2, b2, displacement=None):
    """Compute matrices used for displacement estimation as defined by equations
(7.32) and (7.33) in Gunnar Farnebäck's thesis "Polynomial Expansion for
Orientation and Motion Estimation".
"""
    # Numpy implementation, copied on the slow MATLAB implementation
    # using the c implementation in the mex-file should be much faster
    sides = A1.shape[:-2]
    if displacement is None:
        displacement = np.zeros(sides + (2,))
    A = np.zeros(A1.shape)
    b = np.zeros(b1.shape)
    # If displacement is zero, we will get A = (A1+A2)/2 and b = -(b2-b1)/2.
    for j in range(sides[1]):
        for i in range(sides[0]):
            di = displacement[i,j,0]
            if i+di < 0:
                di = -i
            if i+di > sides[0]:
                di = sides[0] - i -1
            dj = displacement[i,j,1]
            if j+dj < 0:
                dj = -j
            if j+dj > sides[1]:
                dj = sides[1] - j -1
            A[i,j] = (A1[i,j] + A2[i+di,j+dj]) / 2
            AA = np.squeeze(A[i,j])
            bb2 = np.squeeze(b2[i+di,j+dj]) - 2 * np.matmul(AA, [di,dj])
            b[i,j] = -(bb2 - b1[i,j]) / 2
    return  A, b

import numpy as np
import pytest
from make_Abc_fast import make_Abc_fast
from estimate_displacement import estimate_displacement

def test_slingle_square():
    """move a single square by 1 pixel"""
    im0 = np.zeros((64,64))
    im0[30:33, 32:35] = 1
    A1, b1, c1 = make_Abc_fast(im0, 5)
    assert np.abs(c1[31,32]-1) <0.1

    #identical images should not detect displacement
    im1 = np.copy(im0)
    displ, err = estimate_displacement(im0, im1, [5], [15], model="constant", method="fast")
    assert np.abs(displ).max() == 0
    #shift by one pixel on axis 0
    im2 = np.zeros((64,64))
    im2[31:34, 32:35] = 1
    A2, b2, c2 = make_Abc_fast(im2, 5)
    assert np.all(A2[1:] == A1[:-1])
    assert np.all(b2[1:] == b1[:-1])
    assert np.all(c2[1:] == c1[:-1])
    d0 = np.zeros_like(displ)
    d0[...,0] = 1
    A, Delta_b = prepare_displacement_matrices(A1, b1, A2, b2, d0)
    assert np.abs(Delta_b[31,32,0]-1) <0.05
    # displ2, err2 = estimate_displacement(im0, im2, [5], [15], model="constant", method="fast")
    # assert np.abs(displ2).max() == 1

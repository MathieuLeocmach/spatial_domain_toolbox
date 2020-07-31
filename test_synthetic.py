import numpy as np
import pytest
from make_Abc_fast import make_Abc_fast
from estimate_displacement import estimate_displacement
from prepare_displacement_matrices import prepare_displacement_matrices

def test_slingle_square():
    """move a single square by 1 pixel"""
    im0 = np.zeros((64,64))
    im0[30:33, 32:35] = 1
    A1, b1, c1 = make_Abc_fast(im0, 5)
    #c1
    assert np.unravel_index(np.argmax(c1), im0.shape) == (31,33)
    #b1
    assert np.unravel_index(np.argmin(b1[...,0]), im0.shape) == (31,31)
    assert np.unravel_index(np.argmax(b1[...,0]), im0.shape) == (31,34)
    assert np.unravel_index(np.argmin(b1[...,1]), im0.shape) == (29,33)
    assert np.unravel_index(np.argmax(b1[...,1]), im0.shape) == (32,33)
    #A1 diagonal
    assert np.unravel_index(np.argmax(A1[...,0,0]), im0.shape) in [(31,31), (31,35)]
    assert A1[31,31,0,0] == A1[31,35,0,0]
    assert np.unravel_index(np.argmin(A1[...,0,0]), im0.shape) in [(31,32), (31,34)]
    assert A1[31,32,0,0] == A1[31,34,0,0]
    assert np.unravel_index(np.argmax(A1[...,1,1]), im0.shape) in [(29,33), (33,33)]
    assert A1[29,33,1,1] == A1[33,33,1,1]
    assert np.unravel_index(np.argmin(A1[...,1,1]), im0.shape) in [(30,33), (32,33)]
    assert A1[30,33,1,1] == A1[32,33,1,1]
    assert A1[31,31,0,0] == A1[33,33,1,1]
    #A1 nondiagonal
    assert np.all(A1[...,0,1] == A1[...,1,0])
    A101M = [(29,31), (29,32), (30,31), (30,32), (32,34), (32,35), (33, 34), (33,35)]
    assert np.unravel_index(np.argmax(A1[...,0,1]), im0.shape) in A101M
    for i,j in A101M[1:]:
        assert A1[i,j,0,1] == A1[29,31,0,1]
    A101m = [(29,34), (29,35), (30,34), (30,35), (32,31), (32,32), (33, 31), (33,32)]
    assert np.unravel_index(np.argmin(A1[...,0,1]), im0.shape) in A101m
    for i,j in A101m[1:]:
        assert A1[i,j,0,1] == A1[29,34,0,1]


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

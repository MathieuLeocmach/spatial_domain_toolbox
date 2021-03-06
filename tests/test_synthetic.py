import numpy as np
import pytest
from make_Abc_fast import make_Abc_fast
from estimate_displacement import estimate_displacement, get_border
from prepare_displacement_matrices import prepare_displacement_matrices

def test_get_border():
    """test get_border"""
    # 1D
    border = get_border((10,), 2)
    assert np.all(border == [
        True, True,
        False, False, False, False, False, False,
        True, True])
    #2D
    border = get_border((10,5), 2)
    assert border[0,0]
    assert border[-1,-1]
    assert border[1,1]
    assert border[-2,-2]
    assert border[-2,1]
    assert border[1,-2]
    assert np.all(border[2:-2,2:-2] == False)

def test_single_square_fast():
    """move a single square by 1 pixel"""
    im0 = np.zeros((64,64))
    im0[30:33, 32:35] = 1
    A0, b0, c0 = make_Abc_fast(im0, 5)
    #c0
    assert np.unravel_index(np.argmax(c0), im0.shape) == (31,33)
    #b0
    assert np.unravel_index(np.argmin(b0[...,0]), im0.shape) == (32,33)
    assert np.unravel_index(np.argmax(b0[...,0]), im0.shape) == (29,33)
    assert np.unravel_index(np.argmin(b0[...,1]), im0.shape) == (31,34)
    assert np.unravel_index(np.argmax(b0[...,1]), im0.shape) == (31,31)
    #A0 diagonal
    assert np.unravel_index(np.argmin(A0[...,0,0]), im0.shape) in [(30,33), (32,33)]
    assert A0[30,33,0,0] == A0[32,33,0,0]
    assert np.unravel_index(np.argmax(A0[...,0,0]), im0.shape) in [(29,33), (33,33)]
    assert A0[29,33,0,0] == A0[33,33,0,0]
    assert np.unravel_index(np.argmin(A0[...,1,1]), im0.shape) in [(31,32), (31,34)]
    assert A0[31,32,1,1] == A0[31,34,1,1]
    assert np.unravel_index(np.argmax(A0[...,1,1]), im0.shape) in [(31,31), (31,35)]
    assert A0[31,31,1,1] == A0[31,35,1,1]
    assert A0[33,33,0,0] == A0[31,31,1,1]
    #A0 nondiagonal
    assert np.all(A0[...,0,1] == A0[...,1,0])
    A001M = [(29,31), (29,32), (30,31), (30,32), (32,34), (32,35), (33, 34), (33,35)]
    assert np.unravel_index(np.argmax(A0[...,0,1]), im0.shape) in A001M
    for i,j in A001M[1:]:
        assert A0[i,j,0,1] == A0[29,31,0,1]
    A001m = [(29,34), (29,35), (30,34), (30,35), (32,31), (32,32), (33, 31), (33,32)]
    assert np.unravel_index(np.argmin(A0[...,0,1]), im0.shape) in A001m
    for i,j in A001m[1:]:
        assert A0[i,j,0,1] == A0[29,34,0,1]


    #identical images should not detect displacement
    A, Delta_b = prepare_displacement_matrices(A0, b0, A0, b0)
    assert np.all(A==A0)
    assert np.all(Delta_b==0)
    displ, err = estimate_displacement(im0, im0, [5], [15], model="constant", method="fast")
    assert np.all(displ == 0)

    #shift by one pixel on axis 0
    im1 = np.zeros((64,64))
    im1[31:34, 32:35] = 1
    A1, b1, c1 = make_Abc_fast(im1, 5)
    assert np.all(A1[1:] == A0[:-1])
    assert np.all(b1[1:] == b0[:-1])
    assert np.all(c1[1:] == c0[:-1])
    # no initial guess of the displacement
    A, Delta_b = prepare_displacement_matrices(A0, b0, A1, b1)
    assert np.all(Delta_b == -0.5*(b1-b0))
    assert np.all(A == 0.5*(A0+A1))
    displ, err = estimate_displacement(im0, im1, [5], [15], model="constant", method="fast")
    np.testing.assert_almost_equal(displ[31,33,0], 1,1)
    np.testing.assert_almost_equal(displ[31,33,1], 0)
    #initial guess of the displacement
    d0 = np.zeros_like(displ)
    d0[...,0] = 1
    d0 = displ
    displ, err = estimate_displacement(im0, im1, [5], [15], model="constant", method="fast", d0=d0)
    np.testing.assert_almost_equal(displ[31,33,0], 1,0)
    np.testing.assert_almost_equal(displ[31,33,1], 0)

    #shift by one pixel on axis 1
    im1 = np.zeros((64,64))
    im1[30:33, 33:36] = 1
    A1, b1, c1 = make_Abc_fast(im1, 5)
    assert np.all(A1[:,1:] == A0[:,:-1])
    assert np.all(b1[:,1:] == b0[:,:-1])
    assert np.all(c1[:,1:] == c0[:,:-1])
    # no initial guess of the displacement
    A, Delta_b = prepare_displacement_matrices(A0, b0, A1, b1)
    assert np.all(Delta_b == -0.5*(b1-b0))
    assert np.all(A == 0.5*(A0+A1))
    displ, err = estimate_displacement(im0, im1, [5], [15], model="constant", method="accurate")
    np.testing.assert_almost_equal(displ[31,33,0], 0)
    np.testing.assert_almost_equal(displ[31,33,1], 1,1)
    #initial guess of the displacement
    d0 = np.zeros_like(displ)
    d0[...,1] = 1
    d0 = displ
    displ, err = estimate_displacement(im0, im1, [5], [15], model="constant", method="fast", d0=d0)
    np.testing.assert_almost_equal(displ[31,33,0], 0)
    np.testing.assert_almost_equal(displ[31,33,1], 1,0)

def test_single_square_accurate():
    """move a single square by 1 pixel"""
    im0 = np.zeros((64,64))
    im0[30:33, 32:35] = 1
    #identical images should not detect displacement
    displ, err = estimate_displacement(im0, im0, [5], [15], model="constant", method="accurate")
    assert np.all(displ == 0)

    #shift by one pixel on axis 0
    im1 = np.zeros((64,64))
    im1[31:34, 32:35] = 1
    displ, err = estimate_displacement(im0, im1, [5], [15], model="constant", method="accurate")
    assert int(displ[31,33,0]) == 1
    assert int(displ[31,33,1]) == 0

    #shift by one pixel on axis 1
    im1 = np.zeros((64,64))
    im1[30:33, 33:36] = 1
    displ, err = estimate_displacement(im0, im1, [5], [15], model="constant", method="accurate")
    assert int(displ[31,33,0]) == 0
    assert int(displ[31,33,1]) == 1

def test_Correlation_3D():
    """Do not move"""
    im0 = np.zeros((64,64,64), np.float32)
    im0[29:32, 30:33, 32:35] = 1
    # A0, b0, c0 = make_Abc_fast(im0, 5)
    # #c0
    # assert np.unravel_index(np.argmax(c0), im0.shape) == (30,31,33)
    # #b0
    # np.testing.assert_almost_equal(b0[...,0].min(), b0[32,31,33,0])
    # np.testing.assert_almost_equal(b0[...,0].max(), b0[29,31,33,0])
    # np.testing.assert_almost_equal(b0[...,1].min(), b0[30,32,33,1])
    # np.testing.assert_almost_equal(b0[...,1].max(), b0[30,29,33,1])
    # np.testing.assert_almost_equal(b0[...,2].min(), b0[30,31,34,2])
    # np.testing.assert_almost_equal(b0[...,2].max(), b0[30,31,31,2])
    # #A0 diagonal
    # np.testing.assert_almost_equal(A0[...,0,0].min(), A0[29,31,33,0,0])
    # np.testing.assert_almost_equal(A0[...,0,0].max(), A0[32,31,33,0,0])
    # np.testing.assert_almost_equal(A0[...,1,1].min(), A0[30,30,33,1,1])
    # np.testing.assert_almost_equal(A0[...,1,1].max(), A0[30,29,33,1,1])
    # np.testing.assert_almost_equal(A0[...,2,2].min(), A0[30,31,32,2,2])
    # np.testing.assert_almost_equal(A0[...,2,2].max(), A0[30,31,31,2,2])
    # #A0 must be symmetric
    # for i in range(A0.shape[-2]):
    #     for j in range(A0.shape[-1]):
    #         assert np.all(A0[...,i,j]==A0[...,j,i])
    #
    #
    # #identical images should not detect displacement
    # A, Delta_b = prepare_displacement_matrices(A0, b0, A0, b0)
    # assert np.all(A==A0)
    # assert np.all(Delta_b==0)
    displ, err = estimate_displacement(im0, im0, [5], [15], model="constant", method="accurate")
    assert np.all(displ == 0)

def test_Correlation_3D_shift_0():
    """move a single square by 1 pixel along z"""
    im0 = np.zeros((64,64,64), np.float32)
    im0[29:32, 30:33, 32:35] = 1
    #shift by one pixel on axis 0
    im1 = np.zeros_like(im0)
    im1[30:33, 30:33, 32:35] = 1
    displ, err = estimate_displacement(im0, im1, [5], [15], model="constant", method="accurate")
    np.testing.assert_almost_equal(displ[30,31,33,0], 1,1)
    np.testing.assert_almost_equal(displ[30,31,33,1], 0)
    np.testing.assert_almost_equal(displ[30,31,33,2], 0)
    #initial guess of the displacement
    d0 = np.zeros_like(displ)
    d0[...,0] = 1
    d0 = displ
    displ, err = estimate_displacement(im0, im1, [5], [15], model="constant", method="accurate", d0=d0)
    np.testing.assert_almost_equal(displ[30,31,33,0], 1,0)
    np.testing.assert_almost_equal(displ[30,31,33,1], 0)
    np.testing.assert_almost_equal(displ[30,31,33,2], 0)

def test_Correlation_3D_shift_1():
    """move a single square by 1 pixel along y"""
    im0 = np.zeros((64,64,64), np.float32)
    im0[29:32, 30:33, 32:35] = 1
    #shift by one pixel on axis 0
    im1 = np.zeros_like(im0)
    im1[29:32, 31:34, 32:35] = 1
    displ, err = estimate_displacement(im0, im1, [5], [15], model="constant", method="accurate")
    np.testing.assert_almost_equal(displ[30,31,33,0], 0)
    np.testing.assert_almost_equal(displ[30,31,33,1], 1,1)
    np.testing.assert_almost_equal(displ[30,31,33,2], 0)
    #initial guess of the displacement
    d0 = np.zeros_like(displ)
    d0[...,1] = 1
    d0 = displ
    displ, err = estimate_displacement(im0, im1, [5], [15], model="constant", method="accurate", d0=d0)
    np.testing.assert_almost_equal(displ[30,31,33,0], 0)
    np.testing.assert_almost_equal(displ[30,31,33,1], 1,0)
    np.testing.assert_almost_equal(displ[30,31,33,2], 0)

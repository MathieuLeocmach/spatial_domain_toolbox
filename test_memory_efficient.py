import numpy as np
import itertools
import pytest
import memory_efficient

def test_single_square_fast():
    """move a single square by 1 pixel"""
    im0 = np.zeros((64,64))
    im0[30:33, 32:35] = 1

    spatial_size = 5
    n = int((spatial_size - 1) // 2)
    sigma = 0.15 * (spatial_size - 1)
    N = im0.ndim
    basis = np.vstack(list(itertools.product([0, 1, 2], repeat=N))).T
    basis = basis[:,basis.sum(0)<3]
    qAbc = memory_efficient.QuadraticToAbc(N)

    a = np.exp(-np.arange(-n, n+1)**2/(2*sigma**2))
    applicability = [a for dim in range(N)]
    cb = memory_efficient.CorrelationBand(im0.shape, applicability, basis)

    spatial_size2 = 15
    n2 = int((spatial_size2 - 1) // 2)
    sigma2 = 0.15 * (spatial_size2 - 1)
    basis2 = np.zeros((N,1), np.int64)

    a2 = np.exp(-np.arange(-n2, n2+1)**2/(2*sigma2**2))
    applicability2 = [a2 for dim in range(N)]

    cb2 = memory_efficient.CorrelationBand(im0.shape, applicability2, basis2, n_fields=N*(N+3)//2)
    mSNC2 = memory_efficient.metrics_SNC(applicability2, basis2)

    #Separable correlation
    mSC = memory_efficient.metrics_SC(applicability, basis)
    resultsSC = np.zeros(im0.shape+(basis.shape[1],), np.float32)
    for z, r in enumerate(cb.generator(im0)):
        resultsSC[z] = mSC(r)

    #c0
    c0 = qAbc.c(resultsSC)
    assert np.unravel_index(np.argmax(c0), im0.shape) == (31,33)
    #b0
    b0 = qAbc.b(resultsSC)
    assert np.unravel_index(np.argmin(b0[...,0]), im0.shape) == (32,33)
    assert np.unravel_index(np.argmax(b0[...,0]), im0.shape) == (29,33)
    assert np.unravel_index(np.argmin(b0[...,1]), im0.shape) == (31,34)
    assert np.unravel_index(np.argmax(b0[...,1]), im0.shape) == (31,31)
    #A0 diagonal
    A0 = qAbc.A(resultsSC)
    assert np.unravel_index(np.argmin(A0[...,0,0]), im0.shape) in [(30,33), (32,33)]
    np.testing.assert_approx_equal(A0[30,33,0,0], A0[32,33,0,0], 6)
    assert np.unravel_index(np.argmax(A0[...,0,0]), im0.shape) in [(29,33), (33,33)]
    np.testing.assert_approx_equal(A0[29,33,0,0], A0[33,33,0,0], 6)
    assert np.unravel_index(np.argmin(A0[...,1,1]), im0.shape) in [(31,32), (31,34)]
    np.testing.assert_approx_equal(A0[31,32,1,1], A0[31,34,1,1], 6)
    assert np.unravel_index(np.argmax(A0[...,1,1]), im0.shape) in [(31,31), (31,35)]
    np.testing.assert_approx_equal(A0[31,31,1,1], A0[31,35,1,1], 6)
    np.testing.assert_approx_equal(A0[33,33,0,0], A0[31,31,1,1], 6)
    #A0 off diagonal
    assert np.all(A0[...,0,1] == A0[...,1,0])
    A001M = [(29,31), (29,32), (30,31), (30,32), (32,34), (32,35), (33, 34), (33,35)]
    assert np.unravel_index(np.argmax(A0[...,0,1]), im0.shape) in A001M
    for i,j in A001M[1:]:
        assert A0[i,j,0,1] == A0[29,31,0,1]
    A001m = [(29,34), (29,35), (30,34), (30,35), (32,31), (32,32), (33, 31), (33,32)]
    assert np.unravel_index(np.argmin(A0[...,0,1]), im0.shape) in A001m
    for i,j in A001m[1:]:
        assert A0[i,j,0,1] == A0[29,34,0,1]

    #Check normalized separable correlation give the same results here because
    #no signal close to the edge of the image
    mSNC = memory_efficient.metrics_SNC(applicability, basis)
    results = np.empty(im0.shape+(basis.shape[1],), np.float32)
    for z, r in enumerate(cb.generator(im0)):
        results[z] = mSNC(r, z, im0.shape[0])
    np.testing.assert_almost_equal(results, resultsSC, 1)


    #identical images should not detect displacement
    A, Delta_b = memory_efficient.prepare_displacement_matrices_homogeneous(A0, b0, A0, b0)
    assert np.all(A==A0)
    assert np.all(Delta_b==0)
    M = memory_efficient.A_Deltab2G_h(A, Delta_b)
    Gh = np.empty_like(M)
    for z, m in enumerate(cb2.generator(M)):
        Gh[z] = mSNC2(m, z, zlen=len(M), n_fields=N*(N+3)//2)[...,0]
    displ = memory_efficient.Gh2displ(Gh[...,:-N], Gh[...,-N:])
    assert np.all(displ == 0)

    #shift by one pixel on axis 0
    im1 = np.zeros((64,64))
    im1[31:34, 32:35] = 1
    for z, r in enumerate(cb.generator(im1)):
        resultsSC[z] = mSC(r)
    A1 = qAbc.A(resultsSC)
    assert np.all(A1[1:] == A0[:-1])
    b1 = qAbc.b(resultsSC)
    assert np.all(b1[1:] == b0[:-1])
    c1 = qAbc.c(resultsSC)
    assert np.all(c1[1:] == c0[:-1])
    # no initial guess of the displacement
    A, Delta_b = memory_efficient.prepare_displacement_matrices_homogeneous(A0, b0, A1, b1)
    assert np.all(Delta_b == -0.5*(b1-b0))
    assert np.all(A == 0.5*(A0+A1))
    M = memory_efficient.A_Deltab2G_h(A, Delta_b)
    Gh = np.empty_like(M)
    for z, m in enumerate(cb2.generator(M)):
        Gh[z] = mSNC2(m, z, zlen=len(M), n_fields=N*(N+3)//2)[...,0]
    displ = memory_efficient.Gh2displ(Gh[...,:-N], Gh[...,-N:])
    assert int(displ[31,33,0]) == 1
    assert int(displ[31,33,1]) == 0

    #shift by one pixel on axis 1
    im1 = np.zeros((64,64))
    im1[30:33, 33:36] = 1
    for z, r in enumerate(cb.generator(im1)):
        resultsSC[z] = mSC(r)
    A1 = qAbc.A(resultsSC)
    assert np.all(A1[:,1:] == A0[:,:-1])
    b1 = qAbc.b(resultsSC)
    assert np.all(b1[:,1:] == b0[:,:-1])
    c1 = qAbc.c(resultsSC)
    assert np.all(c1[:,1:] == c0[:,:-1])
    # no initial guess of the displacement
    A, Delta_b = memory_efficient.prepare_displacement_matrices_homogeneous(A0, b0, A1, b1)
    assert np.all(Delta_b == -0.5*(b1-b0))
    assert np.all(A == 0.5*(A0+A1))
    M = memory_efficient.A_Deltab2G_h(A, Delta_b)
    Gh = np.empty_like(M)
    for z, m in enumerate(cb2.generator(M)):
        Gh[z] = mSNC2(m, z, zlen=len(M), n_fields=N*(N+3)//2)[...,0]
    displ = memory_efficient.Gh2displ(Gh[...,:-N], Gh[...,-N:])
    assert int(displ[31,33,0]) == 0
    assert int(displ[31,33,1]) == 1

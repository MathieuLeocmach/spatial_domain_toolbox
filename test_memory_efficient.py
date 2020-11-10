from nose.tools import eq_
import numpy as np
import itertools
import memory_efficient


im0 = np.zeros((64,64,64))
im0[29:32, 30:33, 32:35] = 1

spatial_size = 5
N = im0.ndim
qAbc = memory_efficient.QuadraticToAbc(N)
basis = qAbc.basis


applicability = memory_efficient.gaussian_applicability(spatial_size, N)
cb = memory_efficient.CorrelationBand(im0.shape, applicability, basis)

spatial_size2 = 15
basis2 = np.zeros((N,1), np.int64)
applicability2 = memory_efficient.gaussian_applicability(spatial_size2, N)

cb2 = memory_efficient.CorrelationBand(im0.shape, applicability2, basis2, n_fields=N*(N+3)//2)
mSNC2 = memory_efficient.metrics_SNC(applicability2, basis2)

#Separable correlation
mSC = memory_efficient.metrics_SC(applicability, basis)


def test_Separable_Correlation():
    """Separable correlation on known values"""
    resultsSC = np.zeros(im0.shape+(basis.shape[1],), np.float32)
    for z, r in enumerate(cb.generator(im0)):
        resultsSC[z] = mSC(r)

    #c0
    c0 = qAbc.c(resultsSC)
    np.testing.assert_array_equal(resultsSC[...,np.all(cb.basis.T==[0,0,0], axis=1)][...,0], c0)
    assert np.unravel_index(np.argmax(c0), im0.shape) == (30,31,33)
    #b0
    b0 = qAbc.b(resultsSC)
    np.testing.assert_array_equal(resultsSC[...,np.all(cb.basis.T==[1,0,0], axis=1)][...,0], b0[...,0])
    np.testing.assert_array_equal(resultsSC[...,np.all(cb.basis.T==[0,1,0], axis=1)][...,0], b0[...,1])
    np.testing.assert_array_equal(resultsSC[...,np.all(cb.basis.T==[0,0,1], axis=1)][...,0], b0[...,2])
    np.testing.assert_almost_equal(b0[...,0].min(), b0[32,31,33,0])
    np.testing.assert_almost_equal(b0[...,0].max(), b0[29,31,33,0])
    np.testing.assert_almost_equal(b0[...,1].min(), b0[30,32,33,1])
    np.testing.assert_almost_equal(b0[...,1].max(), b0[30,29,33,1])
    np.testing.assert_almost_equal(b0[...,2].min(), b0[30,31,34,2])
    np.testing.assert_almost_equal(b0[...,2].max(), b0[30,31,31,2])
    #A0 diagonal
    A0 = qAbc.A(resultsSC)
    np.testing.assert_array_equal(resultsSC[...,np.all(cb.basis.T==[2,0,0], axis=1)][...,0], A0[...,0,0])
    np.testing.assert_array_equal(resultsSC[...,np.all(cb.basis.T==[0,2,0], axis=1)][...,0], A0[...,1,1])
    np.testing.assert_array_equal(resultsSC[...,np.all(cb.basis.T==[0,0,2], axis=1)][...,0], A0[...,2,2])
    np.testing.assert_almost_equal(A0[...,0,0].min(), A0[29,31,33,0,0])
    np.testing.assert_almost_equal(A0[...,0,0].max(), A0[32,31,33,0,0])
    np.testing.assert_almost_equal(A0[...,1,1].min(), A0[30,30,33,1,1])
    np.testing.assert_almost_equal(A0[...,1,1].max(), A0[30,29,33,1,1])
    np.testing.assert_almost_equal(A0[...,2,2].min(), A0[30,31,32,2,2])
    np.testing.assert_almost_equal(A0[...,2,2].max(), A0[30,31,31,2,2])
    #A0 off diagonal
    np.testing.assert_array_equal(resultsSC[...,np.all(cb.basis.T==[1,1,0], axis=1)][...,0]/2, A0[...,0,1])
    np.testing.assert_array_equal(resultsSC[...,np.all(cb.basis.T==[0,1,1], axis=1)][...,0]/2, A0[...,1,2])
    np.testing.assert_array_equal(resultsSC[...,np.all(cb.basis.T==[1,0,1], axis=1)][...,0]/2, A0[...,0,2])
    #A0 must be symmetric
    for i in range(A0.shape[-2]):
        for j in range(A0.shape[-1]):
            assert np.all(A0[...,i,j]==A0[...,j,i])



def test_Normalized_Separable_Correlation():
    """Check normalized separable correlation give the same results as separable correlation far from the edges"""
    resultsSC = np.zeros(im0.shape+(basis.shape[1],), np.float32)
    for z, r in enumerate(cb.generator(im0)):
        resultsSC[z] = mSC(r)
    #Check normalized separable correlation give the same results here because
    #no signal close to the edge of the image
    mSNC = memory_efficient.metrics_SNC(applicability, basis)
    results = np.empty(im0.shape+(basis.shape[1],), np.float32)
    for z, r in enumerate(cb.generator(im0)):
        results[z] = mSNC(r, z, im0.shape[0])
    np.testing.assert_almost_equal(results, resultsSC, 6)


def test_no_displacement():
    """identical images should not detect displacement"""
    resultsSC = np.zeros(im0.shape+(basis.shape[1],), np.float32)
    for z, r in enumerate(cb.generator(im0)):
        resultsSC[z] = mSC(r)

    A0 = qAbc.A(resultsSC)
    b0 = qAbc.b(resultsSC)
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


def test_1px_0():
    """shift by one pixel on axis 0"""
    resultsSC = np.zeros(im0.shape+(basis.shape[1],), np.float32)
    for z, r in enumerate(cb.generator(im0)):
        resultsSC[z] = mSC(r)

    A0 = qAbc.A(resultsSC)
    b0 = qAbc.b(resultsSC)
    c0 = qAbc.c(resultsSC)
    #shift by one pixel on axis 0
    im1 = np.zeros((64,64,64))
    im1[30:33, 30:33, 32:35] = 1
    for z, r in enumerate(cb.generator(im1)):
        resultsSC[z] = mSC(r)
    A1 = qAbc.A(resultsSC)
    np.testing.assert_array_almost_equal(A1[1:], A0[:-1])
    b1 = qAbc.b(resultsSC)
    np.testing.assert_array_almost_equal(b1[1:], b0[:-1])
    c1 = qAbc.c(resultsSC)
    np.testing.assert_array_almost_equal(c1[1:], c0[:-1])
    # no initial guess of the displacement
    A, Delta_b = memory_efficient.prepare_displacement_matrices_homogeneous(A0, b0, A1, b1)
    assert np.all(Delta_b == -0.5*(b1-b0))
    assert np.all(A == 0.5*(A0+A1))
    #A must be symmetric
    for i in range(A.shape[-2]):
        for j in range(A.shape[-1]):
            assert np.all(A[...,i,j]==A[...,j,i])
    M = memory_efficient.A_Deltab2G_h(A, Delta_b)
    Gh = np.empty_like(M)
    for z, m in enumerate(cb2.generator(M)):
        Gh[z] = mSNC2(m, z, zlen=len(M), n_fields=N*(N+3)//2)[...,0]
    displ = memory_efficient.Gh2displ(Gh[...,:-N], Gh[...,-N:])
    np.testing.assert_almost_equal(displ[30,31,33,0], 1,1)
    np.testing.assert_almost_equal(displ[30,31,33,1], 0)
    np.testing.assert_almost_equal(displ[30,31,33,2], 0)
    #initial guess of the displacement
    d0 = np.array([1,0,0])
    A, Delta_b = memory_efficient.prepare_displacement_matrices_homogeneous(A0, b0, A1, b1, d0)
    assert np.all(A == A0)
    #np.testing.assert_almost_equal(Delta_b, A @ d0, 0)
    M = memory_efficient.A_Deltab2G_h(A, Delta_b)
    Gh = np.empty_like(M)
    for z, m in enumerate(cb2.generator(M)):
        Gh[z] = mSNC2(m, z, zlen=len(M), n_fields=N*(N+3)//2)[...,0]
    displ = memory_efficient.Gh2displ(Gh[...,:-N], Gh[...,-N:])
    np.testing.assert_almost_equal(displ[30,31,33,0], 1)
    np.testing.assert_almost_equal(displ[30,31,33,1], 0)
    np.testing.assert_almost_equal(displ[30,31,33,2], 0)


def test_1px_1():
    """shift by one pixel on axis 1"""
    resultsSC = np.zeros(im0.shape+(basis.shape[1],), np.float32)
    for z, r in enumerate(cb.generator(im0)):
        resultsSC[z] = mSC(r)

    A0 = qAbc.A(resultsSC)
    b0 = qAbc.b(resultsSC)
    c0 = qAbc.c(resultsSC)
    #shift by one pixel on axis 1
    im1 = np.zeros((64,64,64))
    im1[29:32, 31:34, 32:35] = 1
    for z, r in enumerate(cb.generator(im1)):
        resultsSC[z] = mSC(r)
    A1 = qAbc.A(resultsSC)
    np.testing.assert_array_almost_equal(A1[:,1:], A0[:,:-1])
    b1 = qAbc.b(resultsSC)
    np.testing.assert_array_almost_equal(b1[:,1:], b0[:,:-1])
    c1 = qAbc.c(resultsSC)
    np.testing.assert_array_almost_equal(c1[:,1:], c0[:,:-1])
    # no initial guess of the displacement
    A, Delta_b = memory_efficient.prepare_displacement_matrices_homogeneous(A0, b0, A1, b1)
    assert np.all(Delta_b == -0.5*(b1-b0))
    assert np.all(A == 0.5*(A0+A1))
    #A must be symmetric
    for i in range(A.shape[-2]):
        for j in range(A.shape[-1]):
            assert np.all(A[...,i,j]==A[...,j,i])
    M = memory_efficient.A_Deltab2G_h(A, Delta_b)
    Gh = np.empty_like(M)
    for z, m in enumerate(cb2.generator(M)):
        Gh[z] = mSNC2(m, z, zlen=len(M), n_fields=N*(N+3)//2)[...,0]
    displ = memory_efficient.Gh2displ(Gh[...,:-N], Gh[...,-N:])
    np.testing.assert_almost_equal(displ[30,31,33,0], 0)
    np.testing.assert_almost_equal(displ[30,31,33,1], 1,1)
    np.testing.assert_almost_equal(displ[30,31,33,2], 0)
    #initial guess of the displacement
    d0 = np.array([0,1,0])
    A, Delta_b = memory_efficient.prepare_displacement_matrices_homogeneous(A0, b0, A1, b1, d0)
    assert np.all(A == A0)
    #np.testing.assert_almost_equal(Delta_b, A @ d0, 0)
    M = memory_efficient.A_Deltab2G_h(A, Delta_b)
    Gh = np.empty_like(M)
    for z, m in enumerate(cb2.generator(M)):
        Gh[z] = mSNC2(m, z, zlen=len(M), n_fields=N*(N+3)//2)[...,0]
    displ = memory_efficient.Gh2displ(Gh[...,:-N], Gh[...,-N:])
    np.testing.assert_almost_equal(displ[30,31,33,0], 0)
    np.testing.assert_almost_equal(displ[30,31,33,1], 1,1)
    np.testing.assert_almost_equal(displ[30,31,33,2], 0)

def test_Large_Separable_Correlation():
    """Separable correlation on larger array"""
    im3 = np.zeros((512,512,100), np.float32)
    im3[29:32, 30:33, 32:35] = 1

    cb3 = memory_efficient.CorrelationBand(im3.shape, applicability, basis)
    resultsSC3 = np.zeros(im3.shape+(basis.shape[1],), np.float32)
    for z, r in enumerate(cb3.generator(im3)):
        resultsSC3[z] = mSC(r)

    #c0
    c0 = qAbc.c(resultsSC3)
    np.testing.assert_array_equal(resultsSC3[...,np.all(cb.basis.T==[0,0,0], axis=1)][...,0], c0)
    assert np.unravel_index(np.argmax(c0), im3.shape) == (30,31,33)
    #b0
    b0 = qAbc.b(resultsSC3)
    np.testing.assert_array_equal(resultsSC3[...,np.all(cb.basis.T==[1,0,0], axis=1)][...,0], b0[...,0])
    np.testing.assert_array_equal(resultsSC3[...,np.all(cb.basis.T==[0,1,0], axis=1)][...,0], b0[...,1])
    np.testing.assert_array_equal(resultsSC3[...,np.all(cb.basis.T==[0,0,1], axis=1)][...,0], b0[...,2])
    np.testing.assert_almost_equal(b0[...,0].min(), b0[32,31,33,0])
    np.testing.assert_almost_equal(b0[...,0].max(), b0[29,31,33,0])
    np.testing.assert_almost_equal(b0[...,1].min(), b0[30,32,33,1])
    np.testing.assert_almost_equal(b0[...,1].max(), b0[30,29,33,1])
    np.testing.assert_almost_equal(b0[...,2].min(), b0[30,31,34,2])
    np.testing.assert_almost_equal(b0[...,2].max(), b0[30,31,31,2])
    #A0 diagonal
    A0 = qAbc.A(resultsSC3)
    np.testing.assert_array_equal(resultsSC3[...,np.all(cb.basis.T==[2,0,0], axis=1)][...,0], A0[...,0,0])
    np.testing.assert_array_equal(resultsSC3[...,np.all(cb.basis.T==[0,2,0], axis=1)][...,0], A0[...,1,1])
    np.testing.assert_array_equal(resultsSC3[...,np.all(cb.basis.T==[0,0,2], axis=1)][...,0], A0[...,2,2])
    np.testing.assert_almost_equal(A0[...,0,0].min(), A0[29,31,33,0,0])
    np.testing.assert_almost_equal(A0[...,0,0].max(), A0[32,31,33,0,0])
    np.testing.assert_almost_equal(A0[...,1,1].min(), A0[30,30,33,1,1])
    np.testing.assert_almost_equal(A0[...,1,1].max(), A0[30,29,33,1,1])
    np.testing.assert_almost_equal(A0[...,2,2].min(), A0[30,31,32,2,2])
    np.testing.assert_almost_equal(A0[...,2,2].max(), A0[30,31,31,2,2])
    #A0 off diagonal
    np.testing.assert_array_equal(resultsSC3[...,np.all(cb.basis.T==[1,1,0], axis=1)][...,0]/2, A0[...,0,1])
    np.testing.assert_array_equal(resultsSC3[...,np.all(cb.basis.T==[0,1,1], axis=1)][...,0]/2, A0[...,1,2])
    np.testing.assert_array_equal(resultsSC3[...,np.all(cb.basis.T==[1,0,1], axis=1)][...,0]/2, A0[...,0,2])
    #A0 must be symmetric
    for i in range(A0.shape[-2]):
        for j in range(A0.shape[-1]):
            assert np.all(A0[...,i,j]==A0[...,j,i])

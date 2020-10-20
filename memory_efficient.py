import numpy as np
import itertools
from scipy import sparse
from scipy.ndimage import correlate1d
import time
from numba import guvectorize

def quadratic_basis(N):
    """The order of the monomials in quadratic basis functions

N: dimensionality

---
Returns

basis: A (N,B) matrix where B is the number of polynomial basis functions.
`basis[i,j]` is the order of the monomial along dimension i for basis function j.
"""
    basis = np.vstack(list(itertools.product([0, 1, 2], repeat=N))).T
    return basis[:,basis.sum(0)<3]

def gaussian_applicability(spatial_size, N):
    """Gaussian applicability in each dimension"""
    n = int((spatial_size - 1) // 2)
    sigma = 0.15 * (spatial_size - 1)
    a = np.exp(-np.arange(-n, n+1)**2/(2*sigma**2))
    return [a for dim in range(N)]

class CorrelationBand:
    """Iterator that yield plane by plane the correlation results of a signal by
    separable basis and applicability"""

    def __init__(self, shape, applicability, basis, n_fields=None, dtype=np.float32):
        """
shape: Spatial shape of the signal.

applicability: A list containing one 1D vector for each signal dimension.

basis: A (N,B) matrix where N is the signal dimensionality and B is the number
of polynomial basis functions. `basis[i,j]` is the order of the monomial along
dimension i for basis function j.

n_fields: Number of fields to correlate at the same time. Default is a single field.

dtype: Numerical type of the inner storage.
"""
        assert len(shape) == len(applicability)
        assert len(shape) == len(basis)
        #self.signal = signal
        self.applicability = applicability
        self.basis = basis
        self.shape = shape
        self.n_fields = n_fields
        self.dtype = dtype
        # Set up the monomial coordinates.
        self.X = monomials(applicability)
        if n_fields is not None:
            self.X = [X[...,None] for X in self.X]
        # The order in which correlations are performed is crucial
        # We want to perform each calculation only once, and store in memory
        # just what is needed. Therefore, for each dimension, we perform
        # correlation with monomials by descending order. In particular, the
        # zeroth order monomial is correlated last in order not to alter the
        # input of correlation for higher order monomials.
        self.sorted_basis = basis[::-1,np.lexsort(basis)][::-1,::-1]
        # Example for 2D quadratic base:
        # 0 1 0 2 1 0
        # 2 1 1 0 0 0

        # Prepare storage
        self._res = dict()
        # Here, in order to save memory and cache access, we store only a thin
        # band in the slowest varrying dimension, of size len(applicability[0]).
        # Convolutions with monomials in the slowest varrying dimensions will
        # not be stored.
        bandshape = (len(applicability[0]),)+shape[1:]
        if n_fields is not None:
            bandshape = bandshape + (n_fields,)
        # Create storage, fastest varrying dimension first, do not store slowest results
        #t_alloc = time.time()
        for dim in range(len(shape)-1,0,-1):
            last_index = self.sorted_basis[:,0] -1
            for bf in self.sorted_basis.T:
                e = bf[dim]
                index = np.copy(bf)
                index[:dim] = 0
                if np.all(index == last_index):
                    continue
                last_index = index
                indextuple = tuple(index.tolist())
                self._res[indextuple] = np.zeros(bandshape, dtype)
        #print("time for memory allocation: %g ms"%(1e3*(time.time()-t_alloc)))


    def generator(self, signal):
        """Generator that yields correlation results each hyperplane of the
signal, for each basis function.

signal: An iterable of hyperplanes of the signal (arrays of shape `shape[1:]`
(or `shape[1:]+(n_fields,)`)), or an iterator over such an iterable.

---
Yield: An hyperplane of the correlation results perpendicular to the slowest
varrying dimension (e.g. YX plane of a ZYX image) + a last dimension of size B
that contains one coefficient per basis function, in the same order as the basis.
"""
        N = len(self.shape)
        if np.iterable(signal):
            it_signal = iter(signal)
        else:
            it_signal = signal
        # if self.n_fields is None:
        #     assert signal.shape == self.shape
        # else:
        #     assert signal.shape == self.shape + (self.n_fields,)
        thickness = len(self.applicability[0])
        halfth = thickness//2

        #ensures the storage is zero everywhere
        for value in self._res.values():
            value[:] = 0

        #t_hyperplane = 0
        #t_out = 0

        for z in range(self.shape[0]+thickness):
            # Internally, a new hyperplane overwrites the hyperplane that was
            # input thickness-of-the-band planes ago.
            rollingZ = z%thickness

            #t_hp = time.time()
            if z < self.shape[0]:
                # Store the hyperplane in the band
                self._res[(0,)*N][rollingZ] = next(it_signal)

                # Perform correlation on all the dimensions of the hyperplane,
                # fastest varrying dimension first
                for dim in range(N-1,0,-1):
                    #to avoid repeated calculations, we remember what was computed last
                    last_index = self.sorted_basis[:,0] -1
                    for bf in self.sorted_basis.T:
                        e = bf[dim]
                        index = np.copy(bf)
                        index[:dim] = 0
                        if np.all(index == last_index):
                            continue
                        last_index = index
                        prior = np.copy(index)
                        prior[dim] = 0
                        index = tuple(index.tolist())
                        prior = tuple(prior.tolist())
                        # Compute the correlation
                        correlate1d(
                            self._res[prior][rollingZ],
                            (self.applicability[dim] * self.X[dim].ravel()**e).astype(self.dtype),
                            axis = dim-1, #because hyperplane as one dimension less
                            output = self._res[index][rollingZ],
                            mode='constant')
            else:
                # if close to the edge, just erase the current hyperplane everywhere
                for value in self._res.values():
                    value[rollingZ] = 0
            #t_hyperplane += time.time() - t_hp
            if z >= halfth and z-halfth<self.shape[0]:
                #t_o = time.time()
                # Prepare output
                if self.n_fields is None:
                    out = np.zeros(self.shape[1:] + self.basis.shape[1:], dtype=self.dtype)
                else:
                    out = np.zeros(self.shape[1:] + (self.n_fields,) + self.basis.shape[1:], dtype=self.dtype)
                #roll monomial and applicability to be in phase with the current plane
                rollshift = z+1
                X = np.ascontiguousarray(np.roll(self.X[0].ravel(), rollshift))
                app = np.ascontiguousarray(np.roll(self.applicability[0].ravel(), rollshift))
                # Perform correlation in the slowest varrying dimension (along the
                # thinckness of the band), in the order of the basis
                for b, index in enumerate(self.basis.T):
                    prior = np.copy(index)
                    prior[0] = 0
                    prior = tuple(prior.tolist())
                    kernel = (app * X**index[0]).reshape(self.X[0].shape).astype(self.dtype)
                    out[...,b] = np.sum(self._res[prior] * kernel, axis=0)
                #t_out += time.time() - t_o
                yield out
        #print("Time to convolve hyperplanes: %g ms"%(1e3*t_hyperplane))
        #print("Time to convolve in the slowest dimension: %g ms"%(1e3*t_out))

def monomials(applicability):
    """Return the monomial coordinates within the applicability range.
    If we do separable computations, these are vectors, otherwise full arrays.
    Fastest varrying coordinate last"""
    if isinstance(applicability, list):
        ashape = tuple(map(len, applicability))
    else:
        ashape = applicability.shape
    N = len(ashape)
    X = []
    for dim,k in enumerate(ashape):
         n = int((k - 1) // 2)
         X.append(np.arange(-n,n+1).reshape((1,)*dim + (k,) + (1,)*(N-dim-1)))
    return X

def full_app(applicability):
    """Construct the full N-dimensional applicability kernel from a list of N applicabilities"""
    N = len(applicability)
    return np.prod([
        a.reshape((1,)*dim + (len(a),) + (1,)*(N-dim-1))
        for dim, a in enumerate(applicability)
        ], axis=0)

def basis_functions(basis, full_applicability):
    """Basis functions in the applicability range.

basis: A matrix of size NxM, where N is the signal dimensionality and M is the
number of basis functions.

full_applicability: A N dimensional array containing the applicability.

---
Returns

B: A PxM matrix where P is the number of elements in full_applicability (product
of its dimensions)
    """
    N,M = basis.shape
    X = monomials(full_applicability)
    B = np.zeros((np.prod(full_applicability.shape), M))
    for j in range(M):
        b = np.ones(full_applicability.shape)
        for k in range(N):
            b *= X[k]**basis[k,j]
        B[:,j] = b.ravel()
    return B

class metrics_SC:
    """An object function to normalize by the metric in the case of a separable correlation"""
    def __init__(self, applicability, basis, dtype=np.float32):
        full_applicability = full_app(applicability)
        B = basis_functions(basis, full_applicability)
        W = sparse.diags(full_applicability.ravel())
        G = B.T @ W @ B
        self.Ginv = np.linalg.inv(G).astype(dtype)

    def __call__(self, corres):
        """Normalize correlation results by the metric"""
        return (self.Ginv @ corres[...,None]).reshape(corres.shape)

class metrics_SNC:
    """An object function to normalize by the metric in the case of a separable normalized correlation"""
    def __init__(self, applicability, basis, certainty=None, dtype=np.float32):
        N, M = basis.shape
        basis_c = (basis[:,:,None] + basis[:,None,:]).reshape((N, M**2))
        #unicity
        basis_c = basis_c[:,np.lexsort(basis_c[::-1])]
        mask = np.ones(basis_c.shape[1], np.bool)
        mask[1:] = np.any(np.diff(basis_c, axis=1)!=0, axis=0)
        basis_c = basis_c[:,mask]
        self.ij2k = np.zeros((M,M), np.int64)
        for i in range(M):
            for j in range(M):
                self.ij2k[i,j] = np.where(np.all(basis_c.T == basis[:,i]+basis[:,j], axis=1))[0]
        if certainty is None:
            #Compute only on a reduced shape with only a single central pixel that does not touch the edge of the image
            shape = tuple(len(a) for a in applicability)
            certainty = np.ones(shape, dtype=dtype)
        else:
            shape = certainty.shape
        cb_c = CorrelationBand(shape, applicability, basis_c, dtype=dtype)
        self.Ginvs = np.zeros(shape+(M,M), dtype=dtype)
        for z,r_c in enumerate(cb_c.generator(certainty)):
            r_c_ = r_c.reshape((np.prod(r_c.shape[:-1]), r_c.shape[-1]))
            Ginv = np.zeros((r_c_.shape[0],M,M), dtype=dtype)
            for l in range(r_c_.shape[0]):
                G = np.zeros((M,M))
                for i in range(M):
                    for j in range(M):
                        G[i,j] = r_c_[l, self.ij2k[i,j]]
                Ginv[l] = np.linalg.pinv(G, hermitian=True)
            self.Ginvs[z] = Ginv.reshape(self.Ginvs.shape[1:])

    def __call__(self, corres, z=None, zlen=None, n_fields=None):
        """Normalize correlation results by the metric.

If both z and zlen are None, corres is considered to be the whole signal.
Otherwise zlen indicates the total number of planes in the signal and z the
index of the current plane.
        """
        if zlen is None or z is None:
            # We are not doing this plane by plane, but the whole signal at once.
            if n_fields is None:
                assert corres.ndim == self.Ginvs.ndim -1, "Did you mean plane by plane?"
            else:
                assert corres.ndim == self.Ginvs.ndim -2, "Did you mean plane by plane?"
            Ginv = self.Ginvs
        else:
            # Plane by plane
            if self.Ginvs.shape[0] < zlen:
                # Certainty was None, so we pickup the right plane of the inverse metric
                half = self.Ginvs.shape[0]//2
                if z < half:
                    Ginv = self.Ginvs[z]
                elif z+half < zlen:
                    Ginv = self.Ginvs[half]
                else:
                    Ginv = self.Ginvs[z-zlen]
            else:
                # Certainty was not None, the inverse metric is straightforward
                Ginv = self.Ginvs[z]

        #expand the inverse metrics if needed (certainty was None)
        for dim in range(Ginv.ndim-2):
            if Ginv.shape[dim] < corres.shape[dim]:
                half = Ginv.shape[dim]//2
                n_repeat = np.ones(Ginv.shape[dim], np.int64)
                n_repeat[half+1] = corres.shape[dim] - 2*half
                Ginv = np.repeat(Ginv, n_repeat, axis=dim)
        if n_fields is not None:
            Ginv = Ginv[...,None,:,:]
        return (Ginv @ corres[...,None]).reshape(corres.shape)

class QuadraticToAbc:
    """Convert projection coefficients to a quadratic basis into A matrix,
    b vector and c scalar (pixel wise)"""
    def __init__(self, N):
        """N is the dimensionality of the signal"""
        self.N = N
        # generate a quadratic basis as inside function polyexp
        basis = quadratic_basis(N)
        self.index_c = 0
        self.indices_b = np.where(basis.sum(0)==1)[0][::-1]
        self.indices_A = np.where(basis.sum(0)==2)[0]

    def c(self, r):
        """r is the result of metric-normalised correlations with the quadratic
basis. It can be either the total (N+1)-dimensional array, or a N-dimensional
hyperplane."""
        return np.ascontiguousarray(r[...,self.index_c])

    def b(self, r):
        """r is the result of metric-normalised correlations with the quadratic
basis. It can be either the total (N+1)-dimensional array, or a N-dimensional
hyperplane."""
        return np.ascontiguousarray(r[...,self.indices_b])

    def A(self, r):
        """r is the result of metric-normalised correlations with the quadratic
basis. It can be either the total (N+1)-dimensional array, or a N-dimensional
hyperplane."""
        N = self.N
        A = np.zeros(r.shape[:-1]+(N,N), dtype=r.dtype)
        for i,j, k in zip(*np.triu_indices(N), self.indices_A):
            if i==j:
                A[...,N-1-i,N-1-j] = r[...,k]
            else:
                A[...,N-1-i,N-1-j] = r[...,k]
                A[...,N-1-i,N-1-j] *= 0.5
                A[...,N-1-j,N-1-i] = A[...,N-1-i,N-1-j]
        return A

def prepare_displacement_matrices_homogeneous(A1, b1, A2, b2, displacement=None):
    """Compute matrices used for displacement estimation as defined by equations
(7.32) and (7.33) in Gunnar Farnebäck's thesis "Polynomial Expansion for
Orientation and Motion Estimation". Here we suppose an homogenous translation.

A1,b1: Local polynomial expension coefficients at time 1. A1 is a N+2
dimensional array, where the first N indices indicates the position in the
signal and the last two contains the matrix for each point. In the same way, b1
is a N+1 dimensional array. Such arrays can be obtained via QuadraticToAbc.

A2,b2: Local polynomial expension coefficients at time 2.

displacement: The global translation vector from time 1 to time 2.

----
Returns

A: Advected average of A1 and A2 matrices (Eq. 7.32)

Delta_b: advected difference of b2 and b1 (Eq. 7.33)
"""
    assert A1.shape == A2.shape
    assert A1.shape[:-1] == b1.shape
    assert A1.shape[-1] == b1.shape[-1]
    assert b1.shape == b2.shape
    shape = A1.shape[:-2]
    # N is the dimensionality of the signal we consider here (it might be
    # an hyperplane of the original signal), not the rank of the matrices and
    # vectors.
    N = len(shape)
    if displacement is None:
        displacement = np.zeros(N, dtype=A1.dtype)
    assert displacement.shape == (N,)
    # Integral part of the backward displacement vector
    displ = -np.floor(0.5 + displacement).astype(np.int64)
    # Advect back A2 and b2 by rolling
    A = np.roll(A2, displ, axis=tuple(range(N)))
    Delta_b = -0.5*np.roll(b2, displ, axis=tuple(range(N)))
    #take care of the margins by repeating the last available element of A2 or b2
    for dim, d in enumerate(displ):
        if d >= 0:
            # Use only A1 where A2 is not available
            A[(slice(None,None),)*dim + (slice(0,d),)] = A1[(slice(None,None),)*dim + (slice(0,d),)]
            # Use the last availbale element of b2
            Delta_b[(slice(None,None),)*dim + (slice(0,d),)] = -b2[(slice(None,None),)*dim + (slice(0,1),)]
        else:
            # Use only A1 where A2 is not available
            A[(slice(None,None),)*dim + (slice(-d,None),)] = A1[(slice(None,None),)*dim + (slice(-d,None),)]
            # Use the last availbale element of b2
            Delta_b[(slice(None,None),)*dim + (slice(-d,None),)] = -0.5*b2[(slice(None,None),)*dim + (slice(-1,None),)]
    #Advected average for A1 and A2
    A += A1
    A *= 0.5
    # Advected difference for b1 and b2, to which we add back the forward
    # rounded a priori displacement. Here we have to expand the displacement
    # vector to the same rank as the original signal dimension.
    df = np.zeros(A1.shape[-1], A1.dtype)
    df[-N:] = -displ#displacement
    Delta_b += 0.5*b1 + A @ df
    return A, Delta_b

def A_Deltab2G_h(A, Delta_b):
    """Compute the useful coefficients of G=A.T@A and h = A.T * Delta_b

A: A N+2 dimensional array, where the first N indices indicates the position in
the signal and the last two contains the matrix for each point.

Delta_b: A N+1 dimensional array, where the first N indices indicates the
position in the signal and the last contains the vector for each point.

----
Returns

M: A N+1 dimensional array, where the first N indices indicates the position in
the signal and the last contains first the upper tiangular coefficients of G in
the order given by np.triu_indices, and then the coefficients of h, for each
point. That is D(D+3)/2 coefficients per points, with D the dimensionality of
the original signal.
"""
    D = A.shape[-1]
    shape = A.shape[:-2]
    N = len(shape)
    assert A.shape[-2] == D
    assert Delta_b.shape[-1] == D
    assert Delta_b.shape[:-1] == shape

    M = np.zeros(shape+(D*(D+3)//2,), A.dtype)
    #A.T * A, but A is symmetric
    G = A @ A
    #A.T * Delta_b, but A is symmetric
    h = (A @ Delta_b[...,None])[...,0]
    for k, (i,j) in enumerate(zip(*np.triu_indices(D))):
        M[..., k] = G[...,i,j]
        M[...,-D:] = h
    return M

class CorrelationResult:
    """A class to efficiently compute correlation results and generate displacement matrices."""
    def __init__(self, cb, metric):
        """cb: CorrelationBand instance.

metric: metrics_SNC object.
"""
        self.results = np.empty(cb.shape+(cb.basis.shape[1],), cb.dtype)
        self.cb = cb
        self.metric = metric

    def initialize(self, signal):
        """Compute and store the correlation results of the signal, an hyperplane
at the time. Useful only to initialize the time loop."""
        for z, r in enumerate(self.cb.generator(signal)):
            self.results[z] = self.metric(r, z, self.cb.shape[0])

    def displacement_matrices_generator(self, signal, previous, displz=None):
        """Compute and store the correlation results of the signal, an hyperplane
at the time; and yield the displacement matrices.

signal: An iterable of hyperplanes of the signal (arrays of shape `shape[1:]`),
or an iterator over such an iterable.

previous: An other CorrelationResult object sharing the same CorrelationBand 
object. It must have been either `previous.initialize` or
`previous.generate_displacement_matrices`.

displz: A (shape[0],D-1) array, that contains the initial guess of homogeneous
displacement in each hyperplane.

----
Yields

Gh: The displacement matrices in each hyperplane. A (...,D*(D+3)/2) array, where
the first N-1 dimensions are the position in the hyperplane.
"""
        assert previous != self
        assert previous.cb == self.cb
        if displz is None:
            displz = np.zeros((self.cb.shape[0], len(self.cb.shape)-1), np.int64)
        assert len(displz) == self.cb.shape[0]
        qAbc = memory_efficient.QuadraticToAbc(len(self.cb.shape))
        for z, (res0, r, dz) in enumerate(zip(
            previous.results, self.cb.generator(signal), displz
        )):
            res1 = self.metric(r, z, self.cb..shape[0])
            self.results[z] = res1
            A, Delta_b = memory_efficient.prepare_displacement_matrices_homogeneous(
                qAbc.A(res0), qAbc.b(res0),
                qAbc.A(res1), qAbc.b(res1),
                displacement=dz
            )
            yield memory_efficient.A_Deltab2G_h(A, Delta_b)

@guvectorize(['(float32[:], float32[:], float32[:])'], '(m),(n)->(n)', nopython=True, target='parallel')
def Gh2displ(G, h, res):
    """Compute the least square solution to Gx = h, with G a (D,D) symmetric
matrix and h a (D) vector which coefficients are stored in Gh (see A_Deltab2G_h).

G: A (...,D*(D+1)/2) array, containing the upper tiangular coefficients of G in
the order given by np.triu_indices.

h: A (..., D) array, with D the dimensionality of the original signal.

----
Returns

x: The solution, having the same size as h.
"""
    D = len(h)
    assert len(G) == D*(D+1)//2
    GG = np.empty((D,D), dtype=G.dtype)
    for k, (i,j) in enumerate(zip(*np.triu_indices(D))):
        GG[i,j] = G[k]
        GG[j,i] = G[k]
    res[:] = np.linalg.lstsq(GG,h, rcond=1e-3)[0]

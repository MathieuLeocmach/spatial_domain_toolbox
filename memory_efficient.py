import numpy as np
import itertools
from scipy import sparse
from scipy.ndimage import correlate1d
import time

class CorrelationBand:
    """Iterator that yield plane by plane the correlation results of a signal by
    separable basis and applicability"""

    def __init__(self, shape, applicability, basis, dtype=np.float32):
        """
Signal: Signal values. Must be real and nonsparse.

applicability: A list containing one 1D vector for each signal dimension.

basis: A (N,B) matrix where N is the signal dimensionality and B is the number
of polynomial basis functions. `basis[i,j]` is the order of the monomial along
dimension i for basis function j.

dtype: Numerical type of the inner storage.
"""
        assert len(shape) == len(applicability)
        #self.signal = signal
        self.applicability = applicability
        self.basis = basis
        self.shape = shape
        self.dtype = dtype
        # Set up the monomial coordinates.
        self.X = monomials(applicability)
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

---
Yield: An hyperplane of the correlation results perpendicular to the slowest
varrying dimension (e.g. YX plane of a ZYX image) + a last dimension of size B
that contains one coefficient per basis function, in the same order as the basis.
"""
        N = len(self.shape)
        assert signal.shape == self.shape
        thickness = len(self.applicability[0])
        halfth = thickness//2

        #ensures the storage is zero everywhere
        for value in self._res.values():
            value[:] = 0

        #t_hyperplane = 0
        #t_out = 0

        for z in range(self.shape[0]+thickness):
            # Internally, a new hyperplane overwrites the hyperplane that was input
            # thickness-of-the-band planes ago.
            rollingZ = z%thickness

            #t_hp = time.time()
            if z < self.shape[0]:
                # Store the hyperplane in the band
                self._res[(0,)*N][rollingZ] = signal[z]

                # Perform correlation on all the dimensions of the hyperplane, fastest
                # varrying dimension first
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
            if z >= halfth+1 and z-halfth-1<self.shape[0]:
                #t_o = time.time()
                # Prepare output
                out = np.zeros(self.shape[1:] + self.basis.shape[1:], dtype=self.dtype)
                #roll monomial and applicability to be in phase with the current plane
                rollshift = rollingZ-thickness
                X = np.ascontiguousarray(np.roll(self.X[0].ravel(), rollshift))
                app = np.ascontiguousarray(np.roll(self.applicability[0].ravel(), rollshift))
                # Perform correlation in the slowest varraying dimension (along the
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

    def __call__(self, corres, z=None, zlen=None):
        """Normalize correlation results by the metric.

If both z and zlen are None, corres is considered to be the whole signal.
Otherwise zlen indicates the total number of planes in the signal and z the
index of the current plane.
        """
        if zlen is None or z is None:
            # We are not doing this plane by plane, but the whole signal at once.
            assert corres.ndim == self.Ginvs.ndim -1, "Did you mean plane by plane?"
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
        for dim in range(corres.ndim-1):
            if Ginv.shape[dim] < corres.shape[dim]:
                half = Ginv.shape[dim]//2
                n_repeat = np.ones(Ginv.shape[dim], np.int64)
                n_repeat[half+1] = corres.shape[dim] - 2*half
                Ginv = np.repeat(Ginv, n_repeat, axis=dim)
        return (Ginv @ corres[...,None]).reshape(corres.shape)

class QuadraticToAbc:
    """Convert projection coefficients to a quadratic basis into A matrix,
    b vector and c scalar (pixel wise)"""
    def __init__(self, N):
        """N is the dimensionality of the signal"""
        self.N = N
        # generate a quadratic basis as inside function polyexp
        basis = np.vstack(list(itertools.product([0, 1, 2], repeat=N))).T
        basis = basis[:,basis.sum(0)<3]
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
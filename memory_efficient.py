import numpy as np
import itertools
from scipy import sparse
from scipy.ndimage import correlate1d

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

        for z in range(self.shape[0]+thickness):
            # Internally, a new hyperplane overwrites the hyperplane that was input
            # thickness-of-the-band planes ago.
            rollingZ = z%thickness

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
                            self.applicability[dim] * self.X[dim].ravel()**e,
                            axis = dim-1, #because hyperplane as one dimension less
                            output = self._res[index][rollingZ],
                            mode='constant')
            else:
                # if close to the edge, just erase the current hyperplane everywhere
                for value in self._res.values():
                    value[rollingZ] = 0
            if z >= halfth+1 and z+halfth<self.shape[0]:
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
                    kernel = (app * X**index[0]).reshape(self.X[0].shape)
                    out[...,b] = np.sum(self._res[prior] * kernel, axis=0)
                yield out

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

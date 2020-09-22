import numpy as np
import itertools
from scipy import sparse

class AbComputeBand:
    """Container for (partial) correlation results of a signal by separable
    basis and applicability"""

    def __init__(self, shape, applicability, basis, dtype=np.float32):
        """
shape: Signal shape.

applicability: A list containing one 1D vector for each signal dimension.

basis: A (N,M) matrix where N is the signal dimensionality and M is the number
of polynomial basis functions. `basis[i,j]` is the order of the monomial along
dimension i for basis function j.

dtype: Numerical type of the inner storage.
"""
        assert len(shape) == len(applicability)
        #self.signal = signal
        self.applicability = applicability
        self.basis = basis
        self.shape = signal.shape
        # Set up the monomial coordinates.
        self.X = monomials(applicability)
        # The order in which correlations are performed is crucial
        # We want to perform each calculation only once, and store in memory
        # just what is needed. Therefore, for each dimension, we perform
        # correlation with monomials by descending order. In particular, the
        # zeroth order monomial is correlated last in order not to alter the
        # input of correlation for higher order monomials.
        sorted_basis = basis[::-1,np.lexsort(basis)][::-1,::-1]
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
            last_index = sorted_basis[:,0] -1
            for bf in sorted_basis.T:
                e = bf[dim]
                index = np.copy(bf)
                index[:dim] = 0
                if np.all(index == last_index):
                    continue
                last_index = index
                self_res[index] = np.zeros(bandshape, dtype)

    def __call__(plane):
        """Return correlation results for the plane given as input, for each basis function.

plane: An hyperplane of the image perpendicular to the slowest varrying
dimension (e.g. YX plane of a ZYX image). Planes are assumed to be fed in the
order they are in the image.
"""

        # Perform convolution on fastest varrying dimension first
        for dim in range(signal.ndim-1,-1,-1):
            roi = self.get_roi(dim)
            #to avoid repeated calculations, we remember what was computed last
            last_index = sorted_basis[:,0] -1
            for bf in sorted_basis.T:
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
                # Compute the convolution
                self._res[index] = conv3(
                    self._res[prior],
                    self.applicability[dim].reshape(self.X[dim].shape) * self.X[dim]**e,
                    roi
                    )

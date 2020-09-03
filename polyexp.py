"""Author: Gunnar Farnebäck
        Computer Vision Laboratory
        Linköping University, Sweden
        gf@isy.liu.se

Converted to Python by Mathieu Leocmach
"""
import numpy as np
import itertools
from scipy import sparse

from make_Abc_fast import conv3
from compute_displacement import lstsq_ND

def full_app(applicability):
    """Construct the full N-dimensional applicability kernel from a list of N applicabilities"""
    N = len(applicability)
    return np.prod([
        a.reshape((1,)*dim + (len(a),) + (1,)*(N-dim-1))
        for dim, a in enumerate(applicability)
        ], axis=0)

def monomials(applicability):
    """Return the monomial coordinates within the applicability range.
    If we do separable computations, these are vectors, otherwise full arrays."""
    N = len(applicability)
    if isinstance(applicability, list):
        ashape = map(len, applicability)
    else:
        ashape = applicability.shape
    X = []
    for dim,k in enumerate(ashape):
         n = int((k - 1) // 2)
         X.append(np.arange(-n,n+1).reshape((1,)*dim + (k,) + (1,)*(N-dim-1)))

    #it might be OK and faster to use only the first formula. To be tested.
    # if separable_computations:
    #     X = [x.reshape((1,)*dim + (len(x),) + (1,)*(N-dim-1)) for dim,x in enumerate(X)]
    # else:
    #     X = np.meshgrid(*X)
    return X

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
    M = basis.shape[1]
    X = monomials(full_applicability)
    B = np.zeros((np.prod(full_applicability.shape), M))
    for j in range(M):
        b = np.ones(full_applicability.shape)
        for k in range(N):
            b *= X[k]**basis[k,j]
        B[:,j] = b.ravel()
    return B


class ConvResults:
    """Container for (partial) convolution results of a signal by separable
    basis and applicability"""

    def __init__(self, signal, applicability, region_of_interest):
        """
signal: Signal values. Must be real and nonsparse.

applicability: A list containing one 1D vector for each signal dimension.

region_of_interest: An Nx2 matrix where each row contains start and stop indices
along the corresponding dimensions.
"""
        assert signal.ndim == len(applicability)
        #self.signal = signal
        self.applicability = applicability
        self.region_of_interest = region_of_interest
        self._res = {(0,)*signal.ndim:signal}
        self.shape = signal.shape
        # Set up the monomial coordinates.
        self.X = monomials(applicability)


    def __getitem__(self, index):
        """If existing, retrieves the (partial) convolution result
        corresponding to index, otherwise compute and cache it.

        Example:
        Index (0,2,1) returns the convolution by y**2 and x
        Index (2,0,0) returns the convolution by z**3
        """
        assert len(index) == len(self.shape)
        if isinstance(index, np.ndarray):
            index = tuple(index.tolist())
        if index not in self._res:
            #first nonzero index, i.e. the slowest varrying dimension
            k = np.where(np.array(index)>0)[0][0]
            e = index[k]
            prior = np.array(index)
            prior[k] = 0
            # Retrieve or compute recursively the prior partial result where
            # only the convolution on the slowest varrying dimension is missing
            prior = self[prior]
            # Compute the convolution on the slowest varrying dimension
            self._res[index] = conv3(
                prior,
                self.applicability[k] * self.X[k]**e,
                self.get_roi(k)
                )
        return self._res[index]

    def get_roi(self, k):
        # Region of interest must be carefully computed to avoid needless
        # loss of accuracy.
        roi = self.region_of_interest
        N = len(self.shape)
        for l in range(k-1):
            roi[l, 0] += np.min(self.X[l])
            roi[l, 1] += np.max(self.X[l])
        roi[:,0] = np.maximum(0, roi[:,0])
        roi[:,1] = np.minimum(self.shape, roi[:,1])
        # Fetch a computed correlation results on the next slower varrying dimension.
        #index = np.ones(N, np.int64)
        #index[:k+1] = 0
        #prev = self[index]
        #roi = roi[:prev.ndims]
        if k < N-1:
            # We are working on a convolution result that has already been
            # trimmed. So we have to ensure that the roi along the k
            # direction starts at 0
            koffset = roi[k,0] - max(0, roi[k,0] + np.min(self.X[k]))
            roi -= np.repeat(roi[:,0,None], 2, axis=1)
            roi[k] += koffset
        return roi

def polyexp(signal, certainty=None, basis='quadratic', spatial_size=9, sigma=None,
    region_of_interest=None, applicability=None, save_memory=False, verbose=False,
    cout_func=None, cout_data=None
    ):
    """Compute polynomial expansion in up to four dimensions. The expansion
coefficients are computed according to an algorithm described in chapter 4 of
"Polynomial Expansion for Orientation and Motion Estimation" by Gunnar Farnebäck.

signal: Signal values. Must be real and nonsparse and the number of dimensions,
N, must be at most four.

certainty: Certainty values. Must be real and nonsparse, with size identical to
the signal. If None (Default) certainty is set to 1 everywhere, including off
the edge. (Notice that this is principally wrong but computationally much less
demanding.)

basis: Set of basis functions to use. This can be either a string or a matrix.
In the former case valid values are 'constant', 'linear', 'bilinear',
'quadratic', and 'cubic'. In the latter case the matrix must be of size NxM,
where N is the signal dimensionality and M is the number of basis functions.
The meaning of this parameter is further discussed below.

spatial_size: Size of the spatial support of the filters along each dimension.
Default value is 9.

sigma: Standard deviation of a Gaussian applicability. The default value is
0.15(K-1), where K is the spatial_size.

region_of_interest: An Nx2 matrix where each row contains start and stop indices
along the corresponding dimensions. Default value is all of the signal.

applicability: The default applicability is a Gaussian determined by the
spatial_size and sigma parameters. This field overrules those settings. It can
either be an array of the same dimensionality as the signal, or a list
containing one 1D vector for each signal dimension. The latter can be used when
the applicability is separable and allows more efficient computations.

save_memory: If True, do not use the separable methods even if the applicability
is separable. This saves memory while increasing the amount of computations.

verbose: If True, print a report on how the parameters have been interpreted and
what method is going to be used. This is for debugging purposes and can only be
interpreted in the context of the actual code for this function.

cout_func: Function to compute output certainty. The function will be called for
each point where expansion coefficients are computed. The call is of the form
cout_func(G, G0, h, r, cout_data) where G and G0 are as in equation (3.18),
r is a vector with the expansion coefficients, h equals G*r, and
cout_data is specified below. The output from cout_func must be a numeric array
of the same size at all points.

cout_data: Arbitrary data passed on to cout_func.


----
Returns

r: Computed expansion coefficients. R has N+1 dimensions, where the first N
indices indicate the position in the signal and the last dimension holds the
expansion coefficients. In the case that region_of_interest is less than
N-dimensional, the singleton dimensions are removed.

cout: Output certainty. This is only available if cout_func is given.
cout has N+K dimensions, where the first N indices indicate the position in the
signal and the last K dimensions hold the output certainty. In the case that
region_of_interest is less than N-dimensional, the singleton dimensions are
removed.


The purpose of the BASIS parameter is to specify which basis functions are
to be used. These are limited to be monomials. A good example is the
quadratic basis in 2D, consisting of the functions {1, x, y, x^2, y^2,
xy}. Here x is understood to be increasing along the first dimension of
the signal. (Notice that this with the common visalization of images would
mean downwards, not rightwards.) Naturally y then increases along the
second dimension of the signal. Both variables take integer values in the
interval [-(K-1)/2, (K-1)/2], where K is the SPATIAL_SIZE. The ordering of
the expansion coefficients in R of course follows the basis functions.

Since the basis functions are restricted to monomials, they are uniquely
determined by the exponents for each variable. Therefore the quadratic
basis in 2D may be represented by the 2x6 matrix

  0 1 0 2 0 1
  0 0 1 0 2 1

Thus by letting BASIS be an NxM matrix, an arbitrary basis can be
constructed. A special convention is that an empty matrix is interpreted
as the default quadratic basis.

The exact meaning of 'constant', 'linear', 'bilinear', 'quadratic', and
'cubic' in the various dimensionalities is specified in this table of
equivalent basis matrices (4D is not listed but follows the same system):

Dimensionality: 1                   2                   3

                0                   0                   0
'constant'                          0                   0
                                                        0

              0 1                 0 1 0               0 1 0 0
'linear'                          0 0 1               0 0 1 0
                                                      0 0 0 1

              0 1                0 1 0 1          0 1 0 0 1 1 0 1
'bilinear'                       0 0 1 1          0 0 1 0 1 0 1 1
                                                  0 0 0 1 0 1 1 1

             0 1 2             0 1 0 2 0 1      0 1 0 0 2 0 0 1 1 0
'quadratic'                    0 0 1 0 2 1      0 0 1 0 0 2 0 1 0 1
                                                0 0 0 1 0 0 2 0 1 1

            0 1 2 3        0 1 0 2 1 0 3 2 1 0
'cubic'                    0 0 1 0 1 2 0 1 2 3

                              0 1 0 0 2 0 0 1 1 0 3 0 0 2 2 1 0 1 0 1
                              0 0 1 0 0 2 0 1 0 1 0 3 0 1 0 2 2 0 1 1
                              0 0 0 1 0 0 2 0 1 1 0 0 3 0 1 0 1 2 2 1

The name 'bilinear' actually only makes sense for the 2D case. In 1D it
is just linear and in 3D it should rather be called trilinear."""



    N = signal.ndim
    if N==2 & signal.shape[-1] == 1:
        N=1
    assert signal.dtype.char in np.typecodes['Float'], "Signal must have real floating point values"

    if spatial_size<1:
        raise ValueError('What use would such a small kernel be?')
    elif spatial_size%2 != 1:
        spatial_size = int(2*floor((spatial_size-1)//2) + 1)
        warnings.warn('Only kernels of odd size are allowed. Changed the size to %d.'% spatial_size)
    n = int((spatial_size - 1) // 2)

    if sigma is None:
        sigma = 0.15 * (spatial_size - 1);

    if region_of_interest is None:
        if N ==1:
            region_of_interest = np.array([[0, signal.shape[0]]], dtype=int)
        else:
            region_of_interest = np.array([[0,]*N, list(signal.shape)], dtype=int).T

    if applicability is None:
        #Gaussian applicability in each dimension. Fastest varrying last (contrary to MATLAB code).
        a = np.exp(-np.arange(-n, n+1)**2/(2*sigma**2))
        applicability = [
            a#.reshape((1,)*dim + (spatial_size,) + (1,)*(N-dim-1))
            for dim in range(N)
        ]

    # Basis functions. If given as string, convert to matrix form.
    if basis == 'constant':
        basis = np.zeros((N, 1))
    elif basis == 'linear':
        #not the same order as in MATLAB, but dimension agnostic
        basis = np.vstack(list(itertools.product([0, 1], repeat=N))).T
        basis = basis[:,basis.sum(0)<2]
    elif basis == 'bilinear':
        #not the same order as in MATLAB, but dimension agnostic
        basis = np.vstack(list(itertools.product([0, 1], repeat=N))).T
    elif basis == 'quadratic':
        #not the same order as in MATLAB, but dimension agnostic
        basis = np.vstack(list(itertools.product([0, 1, 2], repeat=N))).T
        basis = basis[:,basis.sum(0)<3]
    elif basis == 'cubic':
        #MATLAB was missing the basis element [1,1,1] in 3D
        basis = np.vstack(list(itertools.product([0, 1, 2, 3], repeat=N))).T
        basis = basis[:,basis.sum(0)<4]
    elif isinstance(basis, str):
        raise ValueError('unknown basis name')
    else:
        assert len(basis) == N, 'basis and signal inconsistent'

    # Decide method
    separable_computations = (not save_memory) and isinstance(applicability, list)
    if certainty is None:
        if separable_computations:
            method = 'SC'
        else:
            method = 'C'
    else:
        if separable_computations:
            method = 'SNC'
        else:
            method = 'NC'

    if save_memory and isinstance(applicability, list):
        #we are not going to do separable computations
        #but we have a separable applicability, collapse it to an array.
        applicability = full_app(applicability)


    # Are we expected to compute output certainty?
    cout_needed = cout_func is not None

    # The caller wants a report about what we are doing.
    if verbose:
        print(f"method: {method}")
        print("basis:")
        print(basis)
        print("applicability:")
        print(applicability)
        print("region_of_interest:")
        print(region_of_interest)
        if certainty is None:
            print("constant certainty assumed")


    ### Now over to the actual computations. ###
    M = basis.shape[1]
    if method[0] != "S":
        # Not separable. Set up the basis functions.
        B = basis_functions(basis, applicability)
    if method == 'NC':
        # Normalized Convolution.
        return normconv(
            signal, certainty, B, applicability, region_of_interest,
            cout_func=cout_func, cout_data=cout_data
            )

    elif method == 'C':
        # Convolution. Compute the metric G and the equivalent correlation
        # kernels.
        W = sparse.diags(applicability.ravel())
        G = B.T @ W @ B
        B = W @ B @ np.linalg.inv(G)

        # Now do the correlations to get the expansion coefficients.
        r = np.zeros(list(np.diff(region_of_interest, axis=1)[:,0])+[M,])
        for j in range(M):
            coeff = conv3(signal, B[:,j].reshape(applicability.shape), region_of_interest)
            r[...,j] = coeff
        if not cout_needed:
            return r
        #not optimized, but not used often
        cout = np.zeros((np.prod(r.shape[:-1]),))
        for i, re in enumerate(r.reshape((np.prod(r.shape[:-1]), M))):
            h = G @ re
            cout[i] = cout_func(G, G, h, re, cout_data)
        cout = cout.reshape(r.shape[:-1])
        return r, cout

    elif method == 'SC':
        # Separable Convolution. This implements the generalization of figure
        # 4.9 to any dimensionality and any choice of polynomial basis.
        # Things do become fairly intricate.

        # Compute inverse metric
        full_applicability = full_app(applicability)
        B = basis_functions(basis, full_applicability)
        W = sparse.diags(full_applicability.ravel())
        G = B.T @ W @ B
        Ginv = np.linalg.inv(G)

        # Delegate convolution calculations to ConvResults class
        # Nothing is computed until we ask.
        convres = ConvResults(signal, applicability, region_of_interest)

        # Ready to multiply with the inverse metric.
        r = np.zeros(list(np.diff(region_of_interest, axis=1)[:,0])+[M,])
        for j in range(M):
            for i in range(M):
                index = np.copy(basis[:,i])
                r[...,j] += Ginv[j,i] * convres[index]
        if not cout_needed:
            return r
        #not optimized, but not used often
        cout = np.zeros((np.prod(r.shape[:-1]),))
        for i, re in enumerate(r.reshape((np.prod(r.shape[:-1]), M))):
            h = G @ re
            cout[i] = cout_func(G, G, h, re, cout_data)
        cout = cout.reshape(r.shape[:-1])
        return r, cout

    elif method == 'SNC':
        # Separable Normalized Convolution. This implements the generalization
        # of figure C.1 to any dimensionality and any choice of polynomial
        # basis. Things do become even more intricate.

        # Delegate convolution calculations to ConvResults class
        # Nothing is computed until we ask.
        convres_f = ConvResults(signal*certainty, applicability, region_of_interest)
        convres_c = ConvResults(certainty, applicability, region_of_interest)

        #Compute and store convolution results in one MxM matrix and one
        # M vector per pixel
        h = np.zeros(convres_f.shape+(M,))
        G = np.zeros(convres_f.shape+(M,M))
        for i in range(M):
            h[...,i] = convres_f[basis[:,i]]
            for j in range(M):
                G[...,i,j] = convres_c[basis[:,i]+basis[:,j]]
        # Normalize the convolution
        # Pixelwise least square solution to G @ r = h
        r = lstsq_ND(G,h)

        if not cout_needed:
            return r

        full_applicability = full_app(applicability)
        B = basis_functions(basis, full_applicability)
        W = sparse.diags(full_applicability.ravel())
        G0 = B.T @ W @ B

        #not optimized, but not used often
        cout = np.zeros(r.shape[:-1])
        for index in np.ndindex(cout.shape):
            cout[index] = cout_func(G[index], G0, h[index], r[index], cout_data)
        return r, cout
    else:
        raise ValueError("Unknown method %s"%method)

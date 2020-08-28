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
        assert signal.ndims == len(applicability)
        self.signal = signal
        self.applicability = applicability
        self.region_of_interest = region_of_interest
        self._res = {(0,)*signal.ndims:signal)}
        N = signal.ndims
        # Set up the monomial coordinates.
    	self.X = []
    	for dim,a in enumerate(applicability):
    		 n = int((len(a) - 1) // 2)
    		 self.X.append(np.arange(-n,n+1).reshape((1,)*dim + (len(x),) + (1,)*(N-dim-1)))


    def __getitem__(self, index):
        """If existing, retrieves the (partial) convolution result
        corresponding to index, otherwise compute and cache it.

        Example:
        Index (0,2,1) returns the convolution by y**2 and x
        Index (2,0,0) returns the convolution by z**3
        """
        assert len(index) == signal.ndims
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
        N = self.signal.ndims
        for l in range(k-1):
            roi[l, 0] += np.min(self.X[l])
            roi[l, 1] += np.max(self.X[l])
        roi[:,0] = np.maximum(0, roi[:,0])
        roi[:,1] = np.minimum(self.signal.shape, roi[:,1])
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
			for dim in N
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
	else
	    assert len(basis) = N, 'basis and signal inconsistent'

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
		applicability = np.prod([
			a.reshape((1,)*dim + (len(a),) + (1,)*(N-dim-1))
			for dim, a in enumerate(applicability)
			], axis=0)

	# Set up the monomial coordinates. If we do separable computations, these
	# are vectors, otherwise full arrays.
	if isinstance(applicability, list):
		ashape = map(len, applicability)
	else:
		ashape = applicability.shape
	X = []
	for dim,k in enumerate(ashape):
		 n = int((k - 1) // 2)
		 X.append(np.arange(-n,n+1))

	if separable_computations:
		X = [x.reshape((1,)*dim + (len(x),) + (1,)*(N-dim-1)) for dim,x in enumerate(X)]
	else:
		X = np.meshgrid(*X)


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
		for dim,x in enumerate(X):
			print(f"X{dim}")
			print(x)


	### Now over to the actual computations. ###
	M = basis.shape[1]
	if method[0] != "S":
		# Not separable. Set up the basis functions.
		B = np.zeros((np.prod(applicability.shape), M))
		for j in range(M):
			b = np.ones(applicability.shape)
			for k in range(N):
				b *= X[k]**basis[k,j]
			B[:,j] = b.ravel()
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
        full_applicability = np.prod([
			a.reshape((1,)*dim + (len(a),) + (1,)*(N-dim-1))
			for dim, a in enumerate(applicability)
			], axis=0)
        B = np.zeros((np.prod(full_applicability.shape), M))
		for j in range(M):
			b = np.ones(full_applicability.shape)
			for k in range(N):
				b *= X[k]**basis[k,j]
			B[:,j] = b.ravel()
        W = sparse.diags(applicability.ravel())
		G = B.T @ W @ B
		Ginv = np.linalg.inv(G)

        # Delegate convolution calculations to ConvResults class
        # Nothing is computed until we ask.
        convres = ConvResults(signal, applicability, region_of_interest)

        # Ready to multiply with the inverse metric.
        r = np.zeros(list(np.diff(region_of_interest, axis=1)[:,0])+[M,])
        for j in range(M):
            for i in range(M):
                index = np.copy(basis(:,i))
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



elseif strcmp(method, 'SNC')
    % Separable Normalized Convolution. This implements the generalization
    % of figure C.1 to any dimensionality and any choice of polynomial
    % basis. Things do become even more intricate.

    sorted_basis = sortrows(basis(end:-1:1, :)')';
    sorted_basis = sorted_basis(end:-1:1, end:-1:1);
    convres_f = cell(1 + max(basis'));
    convres_f{1} = signal .* certainty;

    % basis_c is the set of unique pairwise products of the basis functions.
    basis_c = repmat(basis, [1 1 M]) + repmat(permute(basis, [1 3 2]), [1 M]);
    basis_c = reshape(basis_c, [N M^2]);
    basis_c = unique(basis_c(end:-1:1, :)', 'rows')';
    basis_c = basis_c(end:-1:1, end:-1:1);
    convres_c = cell(1 + max(basis_c'));
    convres_c{1} = certainty;

    % We start with the last dimension because this can be assumed to be
    % most likely to have a limited region of interest. If that is true this
    % ordering saves computations. The correct way to do it would be to
    % process the dimensions in an order determined by the region of
    % interest, but we avoid that complexity for now. Notice that the
    % sorting above must be consistent with the ordering used here.
    for k = N:-1:1
	% Region of interest must be carefully computed to avoid needless
	% loss of accuracy.
	roi = region_of_interest;
	for l = 1:k-1
	    roi(l, 1) = roi(l, 1) + min(X{l});
	    roi(l, 2) = roi(l, 2) + max(X{l});
	end
	roi(:, 1) = max(roi(:, 1), ones(N, 1));
	roi(:, 2) = min(roi(:, 2), size(signal)');
	% We need the index to one of the computed correlation results
	% at the previous level.
	index = sorted_basis(:, 1);
	index(1:k) = 0;
	index = num2cell(1 + index);
	% The max(find(size(...)>1)) stuff is a replacement for ndims(). The
        % latter gives a useless result for column vectors.
	roi = roi(1:max(find(size(convres_f{index{:}}) > 1)), :);
	if k < N
	    koffset = roi(k, 1) - max(1, roi(k, 1) + min(X{k}));
	    roi(:, :) = roi(:, :) + 1 - repmat(roi(:, 1), [1 2]);
	    roi(k, :) = roi(k, :) + koffset;
	end

	% Correlations for c*f at this level of the structure.
	last_index = sorted_basis(:, 1) - 1;
	for j = 1:M
	    index = sorted_basis(:, j);
	    e = index(k);
	    index(1:k-1) = 0;
	    if ~all(index == last_index)
		last_index = index;
		index = num2cell(1 + index);
		prior = index;
		prior{k} = 1;
		convres_f{index{:}} = conv3(convres_f{prior{:}}, ...
					    applicability{k} .* X{k}.^e, ...
					    roi);
	    end
	end

	% Correlations for c at this level of the structure.
	last_index = basis_c(:, 1) - 1;
	for j = 1:size(basis_c, 2)
	    index = basis_c(:, j);
	    e = index(k);
	    index(1:k-1) = 0;
	    if ~all(index == last_index)
		last_index = index;
		index = num2cell(1 + index);
		prior = index;
		prior{k} = 1;
		convres_c{index{:}} = conv3(convres_c{prior{:}}, ...
					    applicability{k} .* X{k}.^e, ...
					    roi);
	    end
	end
    end

    % The hierarchical correlation structure results have been computed. Now
    % we need to solve one equation system at each point to obtain the
    % expansion coefficients. This is for performance reasons unreasonable
    % to implement in matlab code (except when there are very few basis
    % functions), so we resort to solving this with a mex file.
    if ~cout_needed
	r = polyexp_solve_system(basis, convres_f, convres_c, isreal(signal));
    else
	full_applicability = applicability{1};
	for k = 2:N
	    full_applicability = outerprod(full_applicability, applicability{k});
	end

	% We need to compute G0.
	B = zeros([prod(size(full_applicability)) M]);
	for j = 1:M
	    b = 1;
	    for k = 1:N
		b = outerprod(b, X{k} .^ basis(k, j));
	    end
	    B(:, j) = b(:);
	end
	W = diag(sparse(full_applicability(:)));
	G0 = B' * W * B;

	if ~isfield(options, 'cout_data')
	    [r, cout] = polyexp_solve_system(basis, convres_f, convres_c, ...
					     isreal(signal), ...
					     options.cout_func, G0);
	else
	    [r, cout] = polyexp_solve_system(basis, convres_f, convres_c, ...
					     isreal(signal), ...
					     options.cout_func, G0, ...
					     options.cout_data);
	end
    end
end

return

%%%% Helper functions. %%%%

% Compute outer product of two arrays. This only works correctly if they
% have mutually exclusive non-singleton dimensions.
function z = outerprod(x, y)
z = repmat(x, size(y)) .* repmat(y, size(x));
return

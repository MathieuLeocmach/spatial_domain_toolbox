"""Author: Gunnar Farnebäck
        Computer Vision Laboratory
        Linköping University, Sweden
        gf@isy.liu.se

Converted to Python by Mathieu Leocmach
"""

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

             		         0 1 0 0 2 0 0 1 1 0 3 0 0 2 2 1 0 1 0
             		         0 0 1 0 0 2 0 1 0 1 0 3 0 1 0 2 2 0 1
             		         0 0 0 1 0 0 2 0 1 1 0 0 3 0 1 0 1 2 2

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
			a.reshape((1,)*dim + (spatial_size,) + (1,)*(N-dim-1))
			for dim in N
		]







% Basis functions. If given as string, convert to matrix form.
% A special convention is that an empty matrix is interpreted as the default
% basis (quadratic).
if isempty(basis)
    basis = 'quadratic';
end

if ischar(basis)
    switch basis
     case 'constant'
      basis = zeros(N, 1);
     case 'linear'
      if N == 1
	  basis = [0 1];
      elseif N == 2
	  basis = [0 1 0
		   0 0 1];
      elseif N == 3
	  basis = [0 1 0 0
		   0 0 1 0
		   0 0 0 1];
      elseif N == 4
	  basis = [0 1 0 0 0
		   0 0 1 0 0
		   0 0 0 1 0
		   0 0 0 0 1];
      end
     case 'bilinear'
      if N == 1
	  basis = [0 1];
      elseif N == 2
	  basis = [0 1 0 1
		   0 0 1 1];
      elseif N == 3
	  basis = [0 1 0 0 1 1 0 1
		   0 0 1 0 1 0 1 1
		   0 0 0 1 0 1 1 1];
      elseif N == 4
	  basis = [0 1 0 0 0 1 1 1 0 0 0 1 1 1 0 1
		   0 0 1 0 0 1 0 0 1 1 0 1 1 0 1 1
		   0 0 0 1 0 0 1 0 1 0 1 1 0 1 1 1
		   0 0 0 0 1 0 0 1 0 1 1 0 1 1 1 1];
      end
     case 'quadratic'
      if N == 1
	  basis = [0 1 2];
      elseif N == 2
	  basis = [0 1 0 2 0 1
		   0 0 1 0 2 1];
      elseif N == 3
	  basis = [0 1 0 0 2 0 0 1 1 0
		   0 0 1 0 0 2 0 1 0 1
		   0 0 0 1 0 0 2 0 1 1];
      elseif N == 4
	  basis = [0 1 0 0 0 2 0 0 0 1 1 1 0 0 0
		   0 0 1 0 0 0 2 0 0 1 0 0 1 1 0
		   0 0 0 1 0 0 0 2 0 0 1 0 1 0 1
		   0 0 0 0 1 0 0 0 2 0 0 1 0 1 1];
      end
     case 'cubic'
      if N == 1
	  basis = [0 1 2 3];
      elseif N == 2
	  basis = [0 1 0 2 1 0 3 2 1 0
		   0 0 1 0 1 2 0 1 2 3];
      elseif N == 3
	  basis = [0 1 0 0 2 0 0 1 1 0 3 0 0 2 2 1 0 1 0
		   0 0 1 0 0 2 0 1 0 1 0 3 0 1 0 2 2 0 1
		   0 0 0 1 0 0 2 0 1 1 0 0 3 0 1 0 1 2 2];
  basis = [0 1 0 0 0 2 0 0 0 1 1 1 0 0 0 3 0 0 0 2 2 2 1 0 0 1 0 0 1 0 0
	   0 0 1 0 0 0 2 0 0 1 0 0 1 1 0 0 3 0 0 1 0 0 2 2 2 0 1 0 0 1 0
	   0 0 0 1 0 0 0 2 0 0 1 0 1 0 1 0 0 3 0 0 1 0 0 1 0 2 2 2 0 0 1
	   0 0 0 0 1 0 0 0 2 0 0 1 0 1 1 0 0 0 3 0 0 1 0 0 1 0 0 1 2 2 2];
      end
     otherwise
      error('unknown basis name')
    end
else
    if size(basis, 1) ~= N
	error('basis and signal inconsistent');
    end
end

% Decide method.
if isempty(certainty)
    constant_certainty = 1;
else
    constant_certainty = 0;
end

if iscell(applicability)
    separable_computations = 1;
else
    separable_computations = 0;
end

if isfield(options, 'save_memory') & options.save_memory ~= 0
    separable_computations = 0;
end

if constant_certainty & separable_computations
    method = 'SC';
elseif constant_certainty & ~separable_computations
    method = 'C';
elseif ~constant_certainty & separable_computations
    method = 'SNC';
else
    method = 'NC';
end

% If we are not going to do separable computations but we have a separable
% applicability, collapse it to an array.
if ~separable_computations & iscell(applicability)
    a = applicability{1};
    for k = 2:N
	a = outerprod(a, applicability{k});
    end
    applicability = a;
end

% Set up the monomial coordinates. If we do separable computations, these
% are vectors, otherwise full arrays.
for k = 1:N
    if iscell(applicability)
	n = (length(applicability{k}) - 1) / 2;
    else
	n = (size(applicability, k) - 1) / 2;
    end
    X{k} = shiftdim((-n:n)', -(k-1));
end

if ~separable_computations
    if N == 2
	x1 = X{1};
	x2 = X{2};
	X{1} = outerprod(x1, ones(size(x2)));
	X{2} = outerprod(ones(size(x1)), x2);
    elseif N == 3
	x1 = X{1};
	x2 = X{2};
	x3 = X{3};
	X{1} = outerprod(outerprod(x1, ones(size(x2))), ones(size(x3)));
	X{2} = outerprod(outerprod(ones(size(x1)), x2), ones(size(x3)));
	X{3} = outerprod(outerprod(ones(size(x1)), ones(size(x2))), x3);
    elseif N == 4
	x1 = X{1};
	x2 = X{2};
	x3 = X{3};
	x4 = X{4};
	X{1} = outerprod(outerprod(outerprod(x1, ones(size(x2))), ...
				   ones(size(x3))), ones(size(x4)));
	X{2} = outerprod(outerprod(outerprod(ones(size(x1)), x2), ...
				   ones(size(x3))), ones(size(x4)));
	X{3} = outerprod(outerprod(outerprod(ones(size(x1)), ...
					     ones(size(x2))), x3), ...
			 ones(size(x4)));
	X{4} = outerprod(outerprod(outerprod(ones(size(x1)), ...
					     ones(size(x2))), ...
				   ones(size(x3))), x4);
    end
end

% Are we expected to compute output certainty?
if nargout == 2 & ~isfield(options, 'cout_func')
    error('Output certainty expected but no function to compute it provided.');
end

cout_needed = 0;
if nargout == 2 & isfield(options, 'cout_func')
    cout_needed = 1;
end


% The caller wants a report about what we are doing.
if isfield(options, 'verbose') & options.verbose ~= 0
    disp(sprintf('method: %s', method));
    disp('basis:');
    disp(basis);
    disp('applicability:');
    disp(applicability);
    disp('region_of_interest:');
    disp(region_of_interest);
    if isempty(certainty)
	disp('constant certainty assumed');
    end
    for k = 1:N
	disp(sprintf('X%d:\n', k));
	disp(X{k});
    end
end


%%%% Now over to the actual computations. %%%%

M = size(basis, 2);

if strcmp(method, 'NC')
    % Normalized Convolution. Set up the basis functions as required by the
    % normconv() function.
    B = zeros([prod(size(applicability)) M]);
    for j = 1:M
	b = ones(size(applicability));
	for k = 1:N
	    b = b .* X{k}.^basis(k, j);
	end
	B(:, j) = b(:);
    end
    appsize = size(applicability);
    % Remove trailing singleton dimensions from appsize.
    appsize = appsize(1:max(find(appsize > 1)));
    B = reshape(B, [appsize M]);

    % Then call normconv.
    if ~cout_needed
	r = normconv(signal, certainty, B, applicability, region_of_interest);
    else
	normconv_options.cout_func = options.cout_func;
	if isfield(options, 'cout_data')
	    normconv_options.cout_data = options.cout_data;
	end
	[r cout] = normconv(signal, certainty, B, applicability, ...
			    region_of_interest, normconv_options);
    end

elseif strcmp(method, 'C')
    % Convolution. Compute the metric G and the equivalent correlation
    % kernels.
    B = zeros([prod(size(applicability)) M]);
    for j = 1:M
	b = ones(size(applicability));
	for k = 1:N
	    b = b .* X{k}.^basis(k, j);
	end
	B(:, j) = b(:);
    end
    W = diag(sparse(applicability(:)));
    G = B' * W * B;
    B = W * B * inv(G);

    % Now do the correlations to get the expansion coefficients.
    r = zeros([prod(1+diff(region_of_interest')) M]);
    for j = 1:M
	coeff = conv3(signal, reshape(B(:,j), size(applicability)),...
		      region_of_interest);
	r(:, j) = coeff(:);
    end
    r = reshape(r, [1+diff(region_of_interest') M]);

    if cout_needed
	cout = arrayloop(N, r, 'polyexp_cout_helper', G, options);
    end

elseif strcmp(method, 'SC')
    % Separable Convolution. This implements the generalization of figure
    % 4.9 to any dimensionality and any choice of polynomial basis.
    % Things do become fairly intricate.
    convres = cell(1 + max(basis'));
    sorted_basis = sortrows(basis(end:-1:1, :)')';
    sorted_basis = sorted_basis(end:-1:1, end:-1:1);
    convres{1} = signal;

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
	roi = roi(1:max(find(size(convres{index{:}}) > 1)), :);
	if k < N
	    koffset = roi(k, 1) - max(1, roi(k, 1) + min(X{k}));
	    roi(:, :) = roi(:, :) + 1 - repmat(roi(:, 1), [1 2]);
	    roi(k, :) = roi(k, :) + koffset;
	end
% 	  roi(k+1:end, :) = (roi(k+1:end, :) + 1 - ...
% 			     repmat(roi(k+1:end, 1), [1 2]));
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
		convres{index{:}} = conv3(convres{prior{:}}, ...
					  applicability{k} .* X{k}.^e, ...
					  roi);
	    end
	end
    end

    % The hierarchical correlation structure results have been computed. Now
    % we need to multiply with the inverse metric. But first we have to
    % compute it.
    full_applicability = applicability{1};
    for k = 2:N
	full_applicability = outerprod(full_applicability, applicability{k});
    end

    B = zeros([prod(size(full_applicability)) M]);
    for j = 1:M
	b = 1;
	for k = 1:N
	    b = outerprod(b, X{k} .^ basis(k, j));
	end
	B(:, j) = b(:);
    end
    W = diag(sparse(full_applicability(:)));
    G = B' * W * B;
    Ginv = inv(G);

    % Ready to multiply with the inverse metric.
    r = zeros([prod(1+diff(region_of_interest')) M]);
    for j = 1:M
	for i = 1:M
	    index = num2cell(1 + basis(:, i));
	    tmp = convres{index{:}};
	    r(:, j) = r(:, j) + Ginv(j, i) * tmp(:);
	end
    end
    r = reshape(r, [1+diff(region_of_interest') M]);

    if cout_needed
	cout = arrayloop(N, r, 'polyexp_cout_helper', G, options);
    end

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

"""Author: Gunnar Farnebäck
        Computer Vision Laboratory
        Linköping University, Sweden
        gf@isy.liu.se

Converted to Python by Mathieu Leocmach
"""

import numpy as np
from scipy.linalg import lstsq
from numba import jit
from make_Abc_fast import conv3
from gaussian_app import gaussian_app

@jit(nopython=True)
def lstsq_ND(A, b):
    """Compute the least square solution to Ax = b at each point in space.

A: A (N+2) dimensional array of (MxK) matrices

b: A (N+1) dimensional array of (M,) vectors

----
Returns

params: A collection of optimal parameters, having the same size as b.
"""
    assert A.shape[-2] == b.shape[-1]
    shape = A.shape[:-2]
    #Create the output array.
    params = np.zeros_like(b)

    for index in np.ndindex(shape):
        #compute least square coefficients so that |A*params -b| is minimized
        params[index] = lstsq(A[index], b[index])[0]
    return params

def normalized_conv(coeff, cin, app, cinaver=None):
	"""Normalized convolution of a scalar.

coeff: N-dimensional array of scalars

cin: Input certainty (N-dimensional array of scalars).

app: applicability, supposed separable (1D kernel)

cinaver: certainty local average weighted by applicability. Can be computed if not given.
"""
	if cinaver is None:
		cinaver = np.copy(cin)
		for dim in range(N):
			#convolution with (separable) applicability in each dimension
			cinaver = conv3(cinaver, app[(np.newaxis,)*dim + (slice(None),)+(np.newaxis,)*(N-1-dim)])
			#machine epsilon (to avoid division by zero)
			cinaver += np.finfo(coeff.dtype).eps
	#normalized convolution of the coefficient to obtain a local average weighted by applicability and certainty
	coeff *= cin
	#convolution with (separable) applicability in each dimension
	for dim in range(N):
		coeff = conv3(coeff, app[(np.newaxis,)*dim + (slice(None),)+(np.newaxis,)*(N-1-dim)])
	#normalizing convolution
	coeff /= cinaver
	return coeff



def compute_displacement(A, Delta_b, kernelsize, sigma, cin, model):
	"""Compute displacement estimates according to equation (7.30) in Gunnar
Farnebäck's thesis "Polynomial Expansion for Orientation and Motion	Estimation".
Optionally also compute output certainty according to equation (7.24) extended
to parameterized displacement fields.

A, Delta_b: Displacement matrices computed by `prepare_displacement_matrices`

kernelsize, sigma: Size and standard deviation for the Gaussian applicability
used in averaging.

cin: Input certainty.

model: Choice of parametric motion model, 'constant', 'affine', or 'eightparam'.

----
Returns

displacement: Computed displacement field.

cout: Computed (reversed) confidence value. Small values indicate more reliable
displacement values.
	"""
	shape = A.shape[:-2]
	N = len(shape)
	#applicability
	app = gaussian_app(kernelsize, 1, sigma) #to do
	#certainty local average weighted by applicability
	cinaver = np.copy(cin)
	for dim in range(N):
		cinaver = conv3(cinaver, app[(np.newaxis,)*dim + (slice(None),) + (np.newaxis,)*(N-1-dim)])
	#machine epsilon (to avoid division by zero)
	eps = np.finfo(A.dtype).eps
	cinaver += eps
	#A.T * A, but A is symmetric
	AA = A @ A
	#A.T * Delta_b, but A is symmetric
	Ab = A @ Delta_b

	# # define base polynomials
	# if model == 'constant':
	# 	S = [1]
	# elif model == 'affine':
	# 	S = [1,] + [np.arange(s)[(slice(None),)+(np.newaxis,)*(N-1-dim)] for dim, s in enumerate(shape)]
	# elif model == 'eightparam':
	# 	pass

	if model == 'constant':
		# 2D code exploiting symmetries
		# bundle together all the useful coefficients of A.T * A
		Q = np.zeros(shape+(5,))
		Q[...,0] = A[...,0,0]**2 + A[...,0,1]**2
		Q[...,1] = A[...,1,1]**2 + A[...,0,1]**2
		Q[...,2] = A[...,0,0] + A[...,1,1]*A[...,0,1]
		# and A.T * Delta_b
		Q[...,3] = A[...,0,0]*Delta_b[...,0] + A[...,0,1]*Delta_b[...,1]
		Q[...,4] = A[...,0,1]*Delta_b[...,0] + A[...,1,1]*Delta_b[...,1]


		#normalized convolution of the coefficients to obtain a local average weighted by applicability and certainty
		for i in range(Q.shape[-1]):
			Q[...,i] = normalized_conv(Q[...,i], cin, app, cinaver)

		# Solve the equation Qv=q. (Eq. 7.23)
		# where v is the unknown displacement, Q = A.T * A and q = A.T * Delta_b
		# with Q and q locally averaged (weighted according to applicability and certainty)
		# (Here * is matrix product)
		a = Q[...,0]
		b = Q[...,1]
		c = Q[...,2]
		d = Q[...,3]
		e = Q[...,4]
		#     [a c]             [b -c]                     [d]
		# Q = [c b]    inv(Q) = [-c a] /(a*b -c**2)    q = [e]
		displacement = np.zeros(shape+(2,))
		displacement[...,0] = d*b - c*e
		displacement[...,1] = a*e - c*d
		displacement /= (a*b - c**2 + eps)[...,None]

		# Compute output certainty (Eq. 7.24)
		# as Delta_b.T * Delta_b - d * q
		q = Delta_b[...,0]**2 + Delta_b[...,1]**2
		q = normalized_conv(q, cin, app, cinaver)
		cout = q - d*displacement[...,0] - e*displacement[...,1]
		return displacement, cout

	elif model == 'affine':
		# define base polynomials (here affine)
		S = [1,] + [np.arange(s)[(slice(None),)+(np.newaxis,)*(N-1-dim)] for dim, s in enumerate(shape)]
		#S.T * A.T * A * S = S.T * AA * S
		Q = np.zeros(shape+(N*len(S),N*len(S)))
		# Q is composed of (N,N) submatrices of shape (len(S),len(S)),
		# each symmetric and corresponding to an element of AA (itself symmetric).
		# Thus there are a lot of repeated coefficients.
		# Since the calculation of each coefficient is heavy (involves spatial convolution)
		# we will compute each coefficient only once and then take advantage of symmetries

		#upper triangular coefficients of AA, and corresponding submatrices of Q
		for k,l in zip(*np.triu_indices(N)):
			a = AA[...,k,l]
			for i,j in zip(*np.triu_indices(len(S))):
				coeff = normalized_conv(a * S[i] * S[j], cin, app, cinaver)
				#fill submatrix of Q with coefficients
				Q[...,len(S)*k+i, len(S)*l+j] = coeff
				if i != j:
					Q[...,len(S)*k+j, len(S)*l+i] = coeff
			#symmetry between submatrices
			if k != l:
				Q[...,len(S)*l:len(S)*(l+1), len(S)*k:len(S)*(k+1)] = Q[..., len(S)*k:len(S)*(k+1), len(S)*l:len(S)*(l+1)]

		# S.T * A.T * Delta_b = S.T * Ab
		q = np.zeros(shape+(N*len(S),))
		# q is composed of (N,) subvectors of shape (len(S),),
		# each corresponding to an element of Ab.
		# Here there is no symmetry involved, but we follow the same procedure as in Q for clarity
		for k in range(N):
			a = Ab[...,k]
			for i in range(len(S)):
				#fill subvector of q with coefficients
				q[...,len(S)*k+i] = normalized_conv(a * S[i], cin, app, cinaver)


		# Solve the equation Qp=q.
		p = lstsq_ND(Q, q)
		#convert solution to displacement by projecting on the base functions
		displacement = np.zeros(shape+(N,))
		for dim in range(N):
			for i in range(len(S)):
				displacement[...,dim] += p[...,len(S)*dim + i] * S[i]

		# Compute output certainty (Eq. 7.24)
		# as Delta_b.T * Delta_b - displacement * q
		coeff = normalized_conv(np.sum(Delta_b**2, -1), cin, app, cinaver)
		cout = coeff - np.sum(p*q, -1)

		return displacement, cout




#
#  case 'eightparam'
#   [x,y] = ndgrid(1:sides(1), 1:sides(2));
#   Q = zeros([sides 39]);
#   Q(:,:,1)  = A(:,:,1,1).^2 + A(:,:,1,2).^2;                % (1,1)
#   Q(:,:,2)  = Q(:,:,1).*x;                                  % (1,2) (2,1)
#   Q(:,:,3)  = Q(:,:,1).*y;                                  % (1,3) (3,1)
#   Q(:,:,4)  = (A(:,:,1,1) + A(:,:,2,2)).*A(:,:,1,2);        % (1,4) (4,1)
#   Q(:,:,5)  = Q(:,:,4).*x;                      % (1,5) (5,1) (2,4) (4,2)
#   Q(:,:,6)  = Q(:,:,4).*y;                      % (1,6) (6,1) (3,4) (4,3)
#   Q(:,:,7)  = Q(:,:,2).*x;				    % (2,2)
#   Q(:,:,8)  = Q(:,:,2).*y;				    % (2,3) (3,2)
#   Q(:,:,9)  = Q(:,:,5).*x;				    % (2,5) (5,2)
#   Q(:,:,10) = Q(:,:,5).*y;                      % (2,6) (6,2) (3,5) (5,3)
#   Q(:,:,11) = Q(:,:,3).*y;				    % (3,3)
#   Q(:,:,12) = Q(:,:,6).*y;				    % (3,6) (6,3)
#   Q(:,:,13) = A(:,:,1,2).^2 + A(:,:,2,2).^2;                % (4,4)
#   Q(:,:,14) = Q(:,:,13).*x;				    % (4,5) (5,4)
#   Q(:,:,15) = Q(:,:,13).*y;				    % (4,6) (6,4)
#   Q(:,:,16) = Q(:,:,14).*x;				    % (5,5)
#   Q(:,:,17) = Q(:,:,14).*y;				    % (5,6) (6,5)
#   Q(:,:,18) = Q(:,:,15).*y;				    % (6,6)
#   Q(:,:,19) = Q(:,:,7) + Q(:,:,10);			    % (1,7) (7,1)
#   Q(:,:,20) = Q(:,:,19).*x;				    % (2,7) (7,2)
#   Q(:,:,21) = Q(:,:,19).*y;		        % (3,7) (7,3) (2,8) (8,2)
#   Q(:,:,22) = Q(:,:,9) + Q(:,:,17);			    % (4,7) (7,4)
#   Q(:,:,23) = Q(:,:,22).*x;				    % (5,7) (7,5)
#   Q(:,:,24) = Q(:,:,22).*y;		        % (6,7) (7,6) (5,8) (8,5)
#   Q(:,:,25) = Q(:,:,8) + Q(:,:,12);			    % (1,8) (8,1)
#   Q(:,:,26) = Q(:,:,25).*y;				    % (3,8) (8,3)
#   Q(:,:,27) = Q(:,:,10) + Q(:,:,18);			    % (4,8) (8,4)
#   Q(:,:,28) = Q(:,:,27).*y;				    % (6,8) (8,6)
#   Q(:,:,29) = (Q(:,:,20) + Q(:,:,24)).*x;		    % (7,7)
#   Q(:,:,30) = (Q(:,:,21) + Q(:,:,28)).*x;		    % (7,8) (8,7)
#   Q(:,:,31) = (Q(:,:,21) + Q(:,:,28)).*y;		    % (8,8)
#
#   Q(:,:,32) = A(:,:,1,1).*b(:,:,1) + A(:,:,1,2).*b(:,:,2);  % (1)
#   Q(:,:,33) = Q(:,:,32).*x;				    % (2)
#   Q(:,:,34) = Q(:,:,32).*y;				    % (3)
#   Q(:,:,35) = A(:,:,1,2).*b(:,:,1) + A(:,:,2,2).*b(:,:,2);  % (4)
#   Q(:,:,36) = Q(:,:,35).*x;				    % (5)
#   Q(:,:,37) = Q(:,:,35).*y;				    % (6)
#   Q(:,:,38) = (Q(:,:,33) + Q(:,:,37)).*x;		    % (7)
#   Q(:,:,39) = (Q(:,:,33) + Q(:,:,37)).*y;                   % (8)
#
#
#   % Compute displacement from eightparam fields in each neighborhood.
#   app = gaussian_app(kernelsize, 1, sigma);
#   cinaver = conv3(conv3(cin, app), app');
#   Q = conv3(conv3(Q.*repmat(cin, [1 1 39]), app), app') ./ ...
#       (eps + repmat(cinaver, [1 1 39]));
#   % We build the equation system as a quadratic form that min_quadform
#   % can solve. Slightly wasteful but effective.
#   Q = reshape(Q(:,:,...
# 		[ 1  2  3  4  5  6 19 25 32
# 		  2  7  8  5  9 10 20 21 33
# 		  3  8 11  6 10 12 21 26 34
# 		  4  5  6 13 14 15 22 27 35
# 		  5  9 10 14 16 17 23 24 36
# 		  6 10 12 15 17 18 24 28 37
# 		 19 20 21 22 23 24 29 30 38
# 		 25 21 26 27 24 28 30 31 39
# 		 32 33 34 35 36 37 38 39 39]),[sides 9 9]);
#
#   % Solve the equation Qv=-q.
#   p = -min_quadform(Q);
#   displacement = zeros([sides 2]);
#   displacement(:,:,1) = sum(p(:,:,[1 2 3 7 8]).*cat(3, ones(sides), ...
# 						    x, y, x.^2, x.*y), 3);
#   displacement(:,:,2) = sum(p(:,:,4:8).*cat(3, ones(sides), ...
# 					    x, y, x.*y, y.^2), 3);
#
#   if nargout > 1
#       q = b(:,:,1).*b(:,:,1) + b(:,:,2).*b(:,:,2);
#       q = conv3(conv3(q.*cin, app), app') ./ (eps + cinaver);
#       cout = q - sum(p .* Q(:,:,1:8,9), 3);
#   end
# end

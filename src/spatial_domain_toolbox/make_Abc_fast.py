"""Author: Gunnar Farnebäck
        Computer Vision Laboratory
        Linköping University, Sweden
        gf@isy.liu.se

Converted to Python by Mathieu Leocmach
"""

import warnings
import math
import numpy as np
import numpy.matlib
from scipy.ndimage import correlate

def conv3(signal, kernel, roi=None):
    """Quick and dirty implementation of Farneback's `convolution` with a ROI.

Note: The MATLAB implementation by Farneback, is not a convolution but a
correlation since the kernel is not mirrored.
    """
    assert signal.dtype.char in np.typecodes['Float'], "Signal must have real floating point values"
    res = correlate(signal, kernel, mode='constant')
    if roi is None:
        return res
    return res[tuple(slice(a,b) for a,b in roi)]

def conv_results2A(conv_results):
    """Convert the N+1 dimensional result of convolution to the N+2 array of A matrices"""
    N = conv_results.ndim -1
    A = np.zeros(conv_results.shape[:-1]+(N,N))
    #diagonal terms
    for dim in range(N):
        A[...,dim,dim] = conv_results[...,dim+1+N]
    #off-diagonal terms
    for k, (i,j) in enumerate(zip(*np.triu_indices(N,1))):
        A[...,i,j] = conv_results[...,k+1+2*N]/2
        A[...,j,i] = A[...,i,j]
    return A

def conv_results2b(conv_results):
    """Convert the N+1 dimensional result of convolution to the N+1 array of b vectors"""
    N = conv_results.ndim -1
    return np.ascontiguousarray(conv_results[...,1:N+1])

def conv_results2c(conv_results):
    """Convert the N+1 dimensional result of convolution to the N array of c scalars"""
    N = conv_results.ndim -1
    return np.ascontiguousarray(conv_results[...,0])


def make_Abc_fast(signal, spatial_size=9, region_of_interest=None, sigma=None, delta=None, certainty=None):
    """Compute A, b, and c parameters in up to four dimensions. The
parameters relate to the local signal model
$f(x) = x^T A x + b^T x + c$
and are determined by a Gaussian weighted least squares fit. This
implementation uses a fast hierarchical scheme of separable filters,
described in chapter 4 of "Polynomial Expansion for Orientation and
Motion Estimation" by Gunnar Farnebäck.

signal: Signal values. Must be real and nonsparse and the number of dimensions,
N, must be at most four.

spatial_size: Size of the spatial support of the filters along each dimension.
Default value is 9.

region_of_interest: An Nx2 matrix where each row contains start and stop indices
along the corresponding dimensions. Default value is all of the signal.

sigma: Standard deviation of a Gaussian applicability. The default value is
0.15(K-1), where K is the spatial_size. However, if delta is set, that value is
used instead.

delta: The value of the gaussian applicability when it reaches the end of the
supporting region along an axis. If both sigma and delta are set, the former is
used.

certainty: Certainty mask. Must be spatially invariant and symmetric with
respect to all axes and have a size compatible with the signal dimension and the
spatial_size parameter. Default value is all ones.

----
Returns

A: Computed A matrices. A has N+2 dimensions, where the first N indices
indicates the position in the signal and the last two contains the matrix for
each point. In the case that region_of_interest is less than N-dimensional, the
singleton dimensions are removed.

b: Computed b vectors. b has N+1 dimensions, where the first N indices indicates
the position in the signal and the last one contains the vector for each point.
In the case that region_of_interest is less than N-dimensional, the singleton
dimensions are removed.

c: Computed c scalars. c has N dimensions. In the case that region_of_interest
is less than N-dimensional, the singleton dimensions are removed.
    """
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

    if region_of_interest is None:
        if N ==1:
            region_of_interest = np.array([[0, signal.shape[0]]], dtype=int)
        else:
            region_of_interest = np.array([[0,]*N, list(signal.shape)], dtype=int).T


    if sigma is None:
        if delta is None:
            sigma = 0.15 * (spatial_size - 1)
        else:
            sigma = n/math.sqrt(-2*math.log(delta))

    if certainty is None:
        certainty = np.ones((spatial_size,)*N)

    # MATLAB version creates a (spatial_size, 1) shaped array
    a = np.exp(-(np.arange(-n,n+1)**2/(2*sigma**2)))


    if N==1:
        # Set up applicability and basis functions.
        applicability = a
        # MATLAB version creates a (spatial_size, 1) shaped array
        x = np.arange(-n,n+1)
        # MATLAB version creates a (spatial_size, 3) shaped array
        b = np.array([np.ones(x.shape), x, x**2])
        nb = b.shape[0]
        #Compute the inverse metric.
        Q = np.zeros((nb, nb))
        for i in range(nb):
            for j in range(i,nb):
                Q[i,j] = np.sum(b[i] * applicability * certainty * b[j])
                Q[j,i] = Q[i,j]
        del b, applicability, x
        Qinv = np.linalg.inv(Q)

        #convolution in the x direction
        # MATLAB version creates three (spatial_size, 1) shaped array
        kernelx0 = a
        kernelx1 = np.arange(-n,n+1) * a
        kernelx2 = np.arange(-n,n+1)**2 * a
        roix = region_of_interest
        roix[:,0] = np.maximum(roix[:,0], 0)
        roix[:,1] = np.minimum(roix[:,1], len(signal))
        conv_results = np.zeros(list(np.diff(region_of_interest, axis=1)[:,0])+[3,])
        for i, kern in enumerate([kernelx0, kernelx1, kernelx2]):
            conv_results[:,i] = conv3(signal, kern, roix)

        #Apply the inverse metric.
        tmp = Qinv[0,0] * conv_results[:,0] + Qinv[0,2] * conv_results[:,2]
        conv_results[:,1] = Qinv[1,1] * conv_results[:,1]
        conv_results[:,2] = Qinv[2,2] * conv_results[:,2] + Qinv[2,0] * conv_results[:,0]
        conv_results[:,0] = tmp
        del tmp

    elif N==2:
        #Set up applicability and basis functions.
        applicability = a[None,:] * a[:,None]
        #fastest varrying index last
        y,x = np.meshgrid(np.arange(-n,n+1), np.arange(-n,n+1))
        b = np.array([np.ones(x.shape), x, y, x**2, y**2, x*y])
        nb = b.shape[0]

        #Compute the inverse metric.
        Q = np.zeros((nb, nb))
        for i in range(nb):
            for j in range(i,nb):
                Q[i,j] = np.sum(b[i] * applicability * certainty * b[j])
                Q[j,i] = Q[i,j]
        del b, applicability, x, y
        Qinv = np.linalg.inv(Q)

        #Convolutions in the y-direction (fastest varrying).
        # MATLAB version creates three (1,spatial_size) shaped array
        kernely0 = a
        kernely1 = np.arange(-n,n+1)*a
        kernely2 = np.arange(-n,n+1)**2 *a
        roiy = region_of_interest + [[-n, n], [0, 0]]
        roiy[:,0] = np.maximum(roiy[:,0], 0)
        roiy[:,1] = np.minimum(roiy[:,1], len(signal))
        convy_results = np.zeros(np.diff(roiy, axis=1)[:,0].astype(int).tolist()+[3])
        for i, kern in enumerate([kernely0, kernely1, kernely2]):
            # Here we ensures convolution along the fastest varrying index, i.e. y
            convy_results[...,i] = conv3(signal, kern[None,:], roiy)

        #Convolutions in the x-direction (slowest varrying).
        kernelx0 = kernely0[:,None]
        kernelx1 = kernely1[:,None]
        kernelx2 = kernely2[:,None]
        roix = region_of_interest
        roix = roix[:convy_results.ndim]
        #ensures the roi along the x direction starts at 0, since we are working on convy_results that has already been trimmed
        roix[1:] = roix[1:] - np.repeat(roix[1:,0,None], 2, axis=1)
        conv_results = np.zeros(np.diff(region_of_interest, axis=1)[:,0].astype(int).tolist()+[6])
        conv_results[...,0] = conv3(convy_results[...,0], kernelx0, roix) # y0x0
        conv_results[...,1] = conv3(convy_results[...,0], kernelx1, roix) # y0x1
        conv_results[...,3] = conv3(convy_results[...,0], kernelx2, roix) # y0x2
        conv_results[...,2] = conv3(convy_results[...,1], kernelx0, roix) # y1x0
        conv_results[...,5] = conv3(convy_results[...,1], kernelx1, roix) # y1x1
        conv_results[...,4] = conv3(convy_results[...,2], kernelx0, roix) # y2x0
        del convy_results

        # Apply the inverse metric.
        # This is just Qinv @ conv_results, but most coefficients of Qinv are zero.
        # Nonzero coefficients are the diagonal and 1*x**2, and 1*y**2
        tmp = Qinv[0,0] * conv_results[...,0] + Qinv[0,3] * conv_results[...,3] + Qinv[0,4] * conv_results[...,4]
        conv_results[...,1] = Qinv[1,1] * conv_results[...,1]
        conv_results[...,2] = Qinv[2,2] * conv_results[...,2]
        conv_results[...,3] = Qinv[3,3] * conv_results[...,3] + Qinv[3,0] * conv_results[...,0]
        conv_results[...,4] = Qinv[4,4] * conv_results[...,4] + Qinv[4,0] * conv_results[...,0]
        conv_results[...,5] = Qinv[5,5] * conv_results[...,5]
        conv_results[...,0] = tmp
        del tmp


#
#     case 3
#     % Set up applicability and basis functions.
#     applicability = outerprod(a, a, a);
#     [x,y,t] = ndgrid(-n:n);
#     b = cat(4, ones(size(x)), x, y, t, x.*x, y.*y, t.*t, x.*y, x.*t, y.*t);
#     nb = size(b,4);
#
#     % Compute the inverse metric.
#     Q = zeros(nb, nb);
#     for i = 1:nb
#         for j = i:nb
#         Q(i,j) = sum(sum(sum(b(:,:,:,i).*applicability.*certainty.*b(:,:,:,j))));
#         Q(j,i) = Q(i,j);
#         end
#     end
#     clear b applicability x y t
#     Qinv = inv(Q);
#
#     % Convolutions in the t-direction
#     kernelt0 = reshape(a, [1 1 spatial_size]);
#     kernelt1 = reshape((-n:n)'.*a, [1 1 spatial_size]);
#     kernelt2 = reshape(((-n:n).^2)'.*a, [1 1 spatial_size]);
#     roit = region_of_interest+[-n n;-n n;0 0];
#     roit(:,1) = max(roit(:,1), ones(3,1));
#     roit(:,2) = min(roit(:,2), size(signal)');
#     conv_t0 = conv3(signal, kernelt0, roit);
#     conv_t1 = conv3(signal, kernelt1, roit);
#     conv_t2 = conv3(signal, kernelt2, roit);
#
#     % Convolutions in the y-direction
#     kernely0 = reshape(kernelt0, [1 spatial_size]);
#     kernely1 = reshape(kernelt1, [1 spatial_size]);
#     kernely2 = reshape(kernelt2, [1 spatial_size]);
#     roiy = region_of_interest+[-n n;0 0;0 0];
#     roiy(:,1) = max(roiy(:,1), ones(3,1));
#     roiy(:,2) = min(roiy(:,2), size(signal)');
#     if diff(roiy(3,:)) == 0
#         roiy = roiy(1:2,:);
#     else
#         roiy(3,:) = roiy(3,:)+1-roiy(3,1);
#     end
#     conv_t0y0 = conv3(conv_t0, kernely0, roiy);
#     conv_t0y1 = conv3(conv_t0, kernely1, roiy);
#     conv_t0y2 = conv3(conv_t0, kernely2, roiy);
#     clear conv_t0
#     conv_t1y0 = conv3(conv_t1, kernely0, roiy);
#     conv_t1y1 = conv3(conv_t1, kernely1, roiy);
#     clear conv_t1
#     conv_t2y0 = conv3(conv_t2, kernely0, roiy);
#     clear conv_t2
#
#     % Convolutions in the x-direction
#     kernelx0 = reshape(kernelt0, [spatial_size 1]);
#     kernelx1 = reshape(kernelt1, [spatial_size 1]);
#     kernelx2 = reshape(kernelt2, [spatial_size 1]);
#     roix = region_of_interest;
#     roix = roix(1:ndims(conv_t0y0),:);
#     roix(2:end,:) = roix(2:end,:)+1-repmat(roix(2:end,1), [1 2]);
#     conv_results = zeros([diff(region_of_interest')+1 10]);
#     conv_results(:,:,:,1) = conv3(conv_t0y0, kernelx0, roix); % t0y0x0
#     conv_results(:,:,:,2) = conv3(conv_t0y0, kernelx1, roix); % t0y0x1
#     conv_results(:,:,:,5) = conv3(conv_t0y0, kernelx2, roix); % t0y0x2
#     clear conv_t0y0
#     conv_results(:,:,:,3) = conv3(conv_t0y1, kernelx0, roix); % t0y1x0
#     conv_results(:,:,:,8) = conv3(conv_t0y1, kernelx1, roix); % t0y1x1
#     clear conv_t0y1
#     conv_results(:,:,:,6) = conv3(conv_t0y2, kernelx0, roix); % t0y2x0
#     clear conv_t0y2
#     conv_results(:,:,:,4) = conv3(conv_t1y0, kernelx0, roix); % t1y0x0
#     conv_results(:,:,:,9) = conv3(conv_t1y0, kernelx1, roix); % t1y0x1
#     clear conv_t1y0
#     conv_results(:,:,:,10) = conv3(conv_t1y1, kernelx0, roix); % t1y1x0
#     clear conv_t1y1
#     conv_results(:,:,:,7) = conv3(conv_t2y0, kernelx0, roix); % t2y0x0
#     clear conv_t2y0
#
#     % Apply the inverse metric.
#     tmp = Qinv(1,1)*conv_results(:,:,:,1) + ...
#           Qinv(1,5)*conv_results(:,:,:,5) + ...
#           Qinv(1,6)*conv_results(:,:,:,6) + ...
#           Qinv(1,7)*conv_results(:,:,:,7);
#     conv_results(:,:,:,2) = Qinv(2,2)*conv_results(:,:,:,2);
#     conv_results(:,:,:,3) = Qinv(3,3)*conv_results(:,:,:,3);
#     conv_results(:,:,:,4) = Qinv(4,4)*conv_results(:,:,:,4);
#     conv_results(:,:,:,5) = Qinv(5,5)*conv_results(:,:,:,5) + ...
#                 Qinv(5,1)*conv_results(:,:,:,1);
#     conv_results(:,:,:,6) = Qinv(6,6)*conv_results(:,:,:,6) + ...
#                 Qinv(6,1)*conv_results(:,:,:,1);
#     conv_results(:,:,:,7) = Qinv(7,7)*conv_results(:,:,:,7) + ...
#                 Qinv(7,1)*conv_results(:,:,:,1);
#     conv_results(:,:,:,8) = Qinv(8,8)*conv_results(:,:,:,8);
#     conv_results(:,:,:,9) = Qinv(9,9)*conv_results(:,:,:,9);
#     conv_results(:,:,:,10) = Qinv(10,10)*conv_results(:,:,:,10);
#     conv_results(:,:,:,1) = tmp;
#     clear tmp;
#
#
#
#     case 4
#     % Set up applicability and basis functions.
#     applicability = outerprod(a, a, a, a);
#     [x,y,z,t] = ndgrid(-n:n);
#     b = cat(5, ones(size(x)), x, y, z, t, x.*x, y.*y, z.*z, t.*t, ...
#         x.*y, x.*z, x.*t, y.*z, y.*t, z.*t);
#     nb = size(b, 5);
#
#     % Compute the inverse metric.
#     Q = zeros(nb, nb);
#     for i = 1:nb
#         for j = i:nb
#         Q(i,j) = sum(sum(sum(sum(b(:,:,:,:,i).*applicability.*certainty.*b(:,:,:,:,j)))));
#         Q(j,i) = Q(i,j);
#         end
#     end
#     clear b applicability x y z t
#     Qinv = inv(Q);
#
#     % Convolutions in the t-direction
#     kernelt0 = reshape(a, [1 1 1 spatial_size]);
#     kernelt1 = reshape((-n:n)'.*a, [1 1 1 spatial_size]);
#     kernelt2 = reshape(((-n:n).^2)'.*a, [1 1 1 spatial_size]);
#     roit = region_of_interest+[-n n;-n n;-n n;0 0];
#     roit(:,1) = max(roit(:,1), ones(4,1));
#     roit(:,2) = min(roit(:,2), size(signal)');
#     conv_t0 = conv3(signal, kernelt0, roit);
#     conv_t1 = conv3(signal, kernelt1, roit);
#     conv_t2 = conv3(signal, kernelt2, roit);
#
#     % Convolutions in the z-direction
#     kernelz0 = reshape(kernelt0, [1 1 spatial_size]);
#     kernelz1 = reshape(kernelt1, [1 1 spatial_size]);
#     kernelz2 = reshape(kernelt2, [1 1 spatial_size]);
#     roiz = region_of_interest+[-n n;-n n;0 0;0 0];
#     roiz(:,1) = max(roiz(:,1), ones(4,1));
#     roiz(:,2) = min(roiz(:,2), size(signal)');
#     if diff(roiz(4,:)) == 0
#         roiz = roiz(1:2,:);
#     else
#         roiz(4,:) = roiz(4,:)+1-roiz(4,1);
#     end
#     conv_t0z0 = conv3(conv_t0, kernelz0, roiz);
#     conv_t0z1 = conv3(conv_t0, kernelz1, roiz);
#     conv_t0z2 = conv3(conv_t0, kernelz2, roiz);
#     clear conv_t0
#     conv_t1z0 = conv3(conv_t1, kernelz0, roiz);
#     conv_t1z1 = conv3(conv_t1, kernelz1, roiz);
#     clear conv_t1
#     conv_t2z0 = conv3(conv_t2, kernelz0, roiz);
#     clear conv_t2
#
#     % Convolutions in the y-direction
#     kernely0 = reshape(kernelt0, [1 spatial_size]);
#     kernely1 = reshape(kernelt1, [1 spatial_size]);
#     kernely2 = reshape(kernelt2, [1 spatial_size]);
#     roiy = region_of_interest+[-n n;0 0;0 0;0 0];
#     roiy(:,1) = max(roiy(:,1), ones(4,1));
#     roiy(:,2) = min(roiy(:,2), size(signal)');
#     roiy = roiy(1:ndims(conv_t0z0),:);
#     roiy(3:end,:) = roiy(3:end,:)+1-repmat(roiy(3:end,1),[1 2]);
#     conv_t0z0y0 = conv3(conv_t0z0, kernely0, roiy);
#     conv_t0z0y1 = conv3(conv_t0z0, kernely1, roiy);
#     conv_t0z0y2 = conv3(conv_t0z0, kernely2, roiy);
#     clear conv_t0z0
#     conv_t0z1y0 = conv3(conv_t0z1, kernely0, roiy);
#     conv_t0z1y1 = conv3(conv_t0z1, kernely1, roiy);
#     clear conv_t0z1
#     conv_t0z2y0 = conv3(conv_t0z2, kernely0, roiy);
#     clear conv_t0z2
#     conv_t1z0y0 = conv3(conv_t1z0, kernely0, roiy);
#     conv_t1z0y1 = conv3(conv_t1z0, kernely1, roiy);
#     clear conv_t1z0
#     conv_t1z1y0 = conv3(conv_t1z1, kernely0, roiy);
#     clear conv_t1z1
#     conv_t2z0y0 = conv3(conv_t2z0, kernely0, roiy);
#     clear conv_t2z0
#
#     % Convolutions in the x-direction
#     kernelx0 = reshape(kernelt0, [spatial_size 1]);
#     kernelx1 = reshape(kernelt1, [spatial_size 1]);
#     kernelx2 = reshape(kernelt2, [spatial_size 1]);
#     roix = region_of_interest;
#     roix = roix(1:ndims(conv_t0z0y0),:);
#     roix(2:end,:) = roix(2:end,:)+1-repmat(roix(2:end,1), [1 2]);
#     conv_results = zeros([diff(region_of_interest')+1 15]);
#     conv_results(:,:,:,:,1) = conv3(conv_t0z0y0, kernelx0, roix); % t0z0y0x0
#     conv_results(:,:,:,:,2) = conv3(conv_t0z0y0, kernelx1, roix); % t0z0y0x1
#     conv_results(:,:,:,:,6) = conv3(conv_t0z0y0, kernelx2, roix); % t0z0y0x2
#     clear conv_t0z0y0
#     conv_results(:,:,:,:,3) = conv3(conv_t0z0y1, kernelx0, roix); % t0z0y1x0
#     conv_results(:,:,:,:,10) = conv3(conv_t0z0y1, kernelx1, roix); % t0z0y1x1
#     clear conv_t0z0y1
#     conv_results(:,:,:,:,7) = conv3(conv_t0z0y2, kernelx0, roix); % t0z0y2x0
#     clear conv_t0z0y2
#     conv_results(:,:,:,:,4) = conv3(conv_t0z1y0, kernelx0, roix); % t0z1y0x0
#     conv_results(:,:,:,:,11) = conv3(conv_t0z1y0, kernelx1, roix); % t0z1y0x1
#     clear conv_t0z1y0
#     conv_results(:,:,:,:,13) = conv3(conv_t0z1y1, kernelx0, roix); % t0z1y1x0
#     clear conv_t0z1y1
#     conv_results(:,:,:,:,8) = conv3(conv_t0z2y0, kernelx0, roix); % t0z2y0x0
#     clear conv_t0z2y0
#     conv_results(:,:,:,:,5) = conv3(conv_t1z0y0, kernelx0, roix); % t1z0y0x0
#     conv_results(:,:,:,:,12) = conv3(conv_t1z0y0, kernelx1, roix); % t1z0y0x1
#     clear conv_t1z0y0
#     conv_results(:,:,:,:,14) = conv3(conv_t1z0y1, kernelx0, roix); % t1z0y1x0
#     clear conv_t1z0y1
#     conv_results(:,:,:,:,15) = conv3(conv_t1z1y0, kernelx0, roix); % t1z1y0x0
#     clear conv_t1z1y0
#     conv_results(:,:,:,:,9) = conv3(conv_t2z0y0, kernelx0, roix); % t2z0y0x0
#     clear conv_t2z0y0
#
#     % Apply the inverse metric.
#     tmp = Qinv(1,1)*conv_results(:,:,:,:,1) + ...
#           Qinv(1,6)*conv_results(:,:,:,:,6) + ...
#           Qinv(1,7)*conv_results(:,:,:,:,7) + ...
#           Qinv(1,8)*conv_results(:,:,:,:,8) + ...
#           Qinv(1,9)*conv_results(:,:,:,:,9);
#     conv_results(:,:,:,:,2) = Qinv(2,2)*conv_results(:,:,:,:,2);
#     conv_results(:,:,:,:,3) = Qinv(3,3)*conv_results(:,:,:,:,3);
#     conv_results(:,:,:,:,4) = Qinv(4,4)*conv_results(:,:,:,:,4);
#     conv_results(:,:,:,:,5) = Qinv(5,5)*conv_results(:,:,:,:,5);
#     conv_results(:,:,:,:,6) = Qinv(6,6)*conv_results(:,:,:,:,6) + ...
#                   Qinv(6,1)*conv_results(:,:,:,:,1);
#     conv_results(:,:,:,:,7) = Qinv(7,7)*conv_results(:,:,:,:,7) + ...
#                   Qinv(7,1)*conv_results(:,:,:,:,1);
#     conv_results(:,:,:,:,8) = Qinv(8,8)*conv_results(:,:,:,:,8) + ...
#                   Qinv(8,1)*conv_results(:,:,:,:,1);
#     conv_results(:,:,:,:,9) = Qinv(9,9)*conv_results(:,:,:,:,9) + ...
#                   Qinv(9,1)*conv_results(:,:,:,:,1);
#     conv_results(:,:,:,:,10) = Qinv(10,10)*conv_results(:,:,:,:,10);
#     conv_results(:,:,:,:,11) = Qinv(11,11)*conv_results(:,:,:,:,11);
#     conv_results(:,:,:,:,12) = Qinv(12,12)*conv_results(:,:,:,:,12);
#     conv_results(:,:,:,:,13) = Qinv(13,13)*conv_results(:,:,:,:,13);
#     conv_results(:,:,:,:,14) = Qinv(14,14)*conv_results(:,:,:,:,14);
#     conv_results(:,:,:,:,15) = Qinv(15,15)*conv_results(:,:,:,:,15);
#     conv_results(:,:,:,:,1) = tmp;
#     clear tmp;
#
#
#
#
#     otherwise
#     error('More than four dimensions are not supported.')
# end
#
# if nargout > 3
#     params.spatial_size = spatial_size;
#     params.region_of_interest = region_of_interest;
#     params.sigma = sigma;
#     params.delta = delta;
#     params.c = certainty;
# end
    # Build A, b, and c from the convolution results
    A = conv_results2A(conv_results)
    b = conv_results2b(conv_results)
    c = conv_results2c(conv_results)
    A = np.squeeze(A)
    b = np.squeeze(b)
    c = np.squeeze(c)
    return A, b, c

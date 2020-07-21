"""Author: Gunnar Farnebäck
        Computer Vision Laboratory
        Linköping University, Sweden
        gf@isy.liu.se

Converted to Python by Mathieu Leocmach
"""

def conv3(signal, kernel, roi):
    """Quick and dirty implementation of convolution with a ROI."""
    return convolve(signal, kernel)[tuple(slice(a,b) for a,b in roi)]

import warnings
import math
import numpy as np
import numpy.matlib
from scipy.signal import convolve

def make_tensors_fast(
    signal, spatial_size=9, region_of_interest=None,
    gamma=None, sigma=None, delta=None, certainty=None
):
    """Compute orientation tensors in up to four dimensions. The tensors are
    computed according to an algorithm described in chapter 5 of Gunnar
    Farnebäck's thesis, "Polynomial Expansion for Orientation and Motion
    Estimation". This implementation uses the "Separable Convolution" method
    with completely separable filters in a hierarchical scheme. The
    applicability (weighting function) is limited to be a Gaussian.

    signal: Signal values. Must be real numpy array and the number of
    dimensions, N, must be at most four.

    spatial_size: Size of the spatial support of the filters along each
    dimension. Default value is 9.

    region_of_interest: A (N,2) numpy array where each row contains start and
    stop indices along the corresponding dimensions. Default value is all of
    the signal.

    gamma: Relation between the contribution to the tensor from the linear and
    the quadratic parts of the signal, as specified in equation (5.19). 0 means
    that only the quadratic part matters while a very large number means that
    only the linear part is used. Default value is 1/(8*sigma^2).

    sigma: Standard deviation of a Gaussian applicability. The default value is
    0.15(K-1), where K is the spatial_size.

    delta: The value of the gaussian applicability when it reaches the end of
    the supporting region along an axis. If both sigma and
    delta are set, the former is used.

    certainty: Certainty mask. Must be spatially invariant and symmetric with respect
    to all axes and have a size compatible with the signal dimension and the
    spatial_size parameter. Default value is all ones. One application of this
    option is in conjunction with interlaced images.

    ----
    Returns

    T: Computed tensor field. T has N+2 dimensions, where the first N indices
    indicate the position in the signal and the last two contain the tensor for
    each point. In the case that region_of_interest is less than N-dimensional,
    the singleton dimensions are removed.

    params: Dictionary containing the parameters that have been used by the
    algorithm.
    """
    N = signal.ndim
    if N==2 & signal.shape[-1] == 1:
        N=1
    if spatial_size<1:
        raise ValueError('What use would such a small kernel be?')
    elif spatial_size%2 != 1:
        spatial_size = int(2*floor((spatial_size-1)//2) + 1)
        warnings.warn('Only kernels of odd size are allowed. Changed the size to %d.'% spatial_size)

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
    if gamma is None:
        gamma = 1 / (8*sigma**2)
    if certainty is None:
        certainty = np.ones((spatial_size,)*N)

    n = int((spatial_size - 1) // 2)
    a = np.exp(-(np.arange(-n,n+1)**2/(2*sigma**2)))

    if N==1:
        #Orientation tensors in 1D are fairly pointless and only included here
        #for completeness.

        #Set up applicability and basis functions.
        applicability = a
        x = np.arange(-n,n+1)
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
        kernelx0 = a
        kernelx1 = np.arange(-n,n+1) * a
        kernelx2 = np.arange(-n,n+1)**2 * a
        roix = region_of_interest[:,0]
        roix[0] = max(roix[0], 0)
        roix[1] = min(roix[1], len(signal))
        conv_results = np.zeros(list(np.diff(region_of_interest, axis=1)[:,0])+[3,])
        for i, kern in enumerate([kernelx0, kernelx1, kernelx2]):
            conv_results[:,i] = conv3(signal, kern, roix)

        #Apply the inverse metric.
        conv_results[:,1] = Qinv[1,1] * conv_results[:,1];
        conv_results[:,2] = Qinv[2,2] * conv_results[:,2] + Qinv[2,0] * conv_results[:,0]

        # Build tensor components.
        # It's more efficient in matlab code to do a small matrix
        # multiplication "manually" in parallell over all the points
        # than doing a multiple loop over the points and computing the
        # matrix products "automatically".
        # The tensor is of the form A*A'+gamma*b*b', where A and b are
        # composed from the convolution results according to
        # A=[3], b=[2].
        # Thus (excluding gamma)
        # T=[3*3+2*2].
        T = np.zeros(list(np.diff(region_of_interest, axis=1)[:,0])+[1, 1])
        T[...,0,0] = conv_results[:,2]**2 + gamma * conv_results[:,1]**2
        T = np.squeeze(T)

    elif N==2:
        #Set up applicability and basis functions.
        applicability = a[None,:] * a[:,None]
        x, y = np.meshgrid(np.arange(-n,n+1), np.arange(-n,n+1))
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

        #Convolutions in the y-direction.
        kernely0 = a
        kernely1 = np.arange(-n,n+1)*a
        kernely2 = np.arange(-n,n+1)**2 *a
        roiy = region_of_interest + [[-n, n], [0, 0]]
        roiy[:,0] = np.maximum(roiy[:,0], 0)
        roiy[:,1] = np.minimum(roiy[:,1], len(signal))
        convy_results = np.zeros(np.diff(roiy, axis=1)[:,0].astype(int).tolist()+[3])
        for i, kern in enumerate([kernely0, kernely1, kernely2]):
            convy_results[...,i] = conv3(signal, kern[:,None], roiy) #find what conv3 is doing

        #Convolutions in the x-direction.
        kernelx0 = kernely0[None,:]
        kernelx1 = kernely0[None,:]
        kernelx2 = kernely0[None,:]
        #roix = roix(1:ndims(conv_y0),:);
    	#roix(2:end,:) = roix(2:end,:) + 1 - repmat(roix(2:end,1), [1 2]);
        roix = region_of_interest
        roix = roix[:convy_results.ndim]
        #ensures the roi along the x direction starts at 0, since we are working on convy_results that has already been trimmed
        roix[1:] = roix[1:] - np.repeat(roix[1:,0,None], 2, axis=1)
        conv_results = np.zeros(np.diff(region_of_interest, axis=1)[:,0].astype(int).tolist()+[6])

        conv_results[...,0] = conv3(convy_results[...,0], kernelx0, roix) # y0x0
        conv_results[...,1] = conv3(convy_results[...,0], kernelx1, roix) # y0x1
        conv_results[...,3] = conv3(convy_results[...,0], kernelx2, roix) # y0x2
        conv_results[...,2] = conv3(convy_results[...,1], kernelx0, roix) # y1x0
        conv_results[...,5] = conv3(convy_results[...,1], kernelx1, roix) / 2 # y1x1
        conv_results[...,4] = conv3(convy_results[...,2], kernelx0, roix) # y2x0
        del convy_results
        # Apply the inverse metric.
        conv_results[...,1] = Qinv[1,1] * conv_results[...,1]
        conv_results[...,2] = Qinv[2,2] * conv_results[...,2]
        conv_results[...,3] = Qinv[3,3] * conv_results[...,3] + Qinv[3,0] * conv_results[...,0]
        conv_results[...,4] = Qinv[4,4] * conv_results[...,4] + Qinv[4,0] * conv_results[...,0]
        conv_results[...,5] = Qinv[5,5] * conv_results[...,5]
        # Build tensor components.
        # It's more efficient in matlab code to do a small matrix
        # multiplication "manually" in parallell over all the points
        # than doing a multiple loop over the points and computing the
        # matrix products "automatically".
        # The tensor is of the form A*A'+gamma*b*b', where A and b are
        # composed from the convolution results according to
        #   [4  6]    [2]
        # A=[6  5], b=[3].
        # Thus (excluding gamma)
        #   [4*4+6*6+2*2 4*6+5*6+2*3]
        # T=[4*6+5*6+2*3 6*6+5*5+3*3].
        T = np.zeros(list(np.diff(region_of_interest, axis=1)[:,0])+[2,2])
        T[...,0,0] = conv_results[...,3] ** 2 + conv_results[...,5] ** 2 + gamma * conv_results[...,1] ** 2
        T[...,1,1] = conv_results[...,5] ** 2 + conv_results[...,4] ** 2 + gamma * conv_results[...,2] ** 2
        T[...,0,1] = (conv_results[...,3] + conv_results[...,4]) * conv_results[...,5] + gamma * conv_results[...,1] * conv_results[...,2]
        T[...,1,0] = T[...,0,1]
        T = np.squeeze(T)
    # elif N == 3:
    #     # Set up applicability and basis functions.
    #     applicability = outerprod(a,a,a)
    #     x,y,t = ndgrid(np.arange(- n,n+1))
    #     b = cat(4,np.ones((x.shape,x.shape)),x,y,t,np.multiply(x,x),np.multiply(y,y),np.multiply(t,t),np.multiply(x,y),np.multiply(x,t),np.multiply(y,t))
    #     nb = b.shape[4-1]
    #     # Compute the inverse metric.
    #     Q = np.zeros((nb,nb))
    #     for i in np.arange(1,nb+1).reshape(-1):
    #         for j in np.arange(i,nb+1).reshape(-1):
    #             Q[i,j-1] = sum(sum(sum(np.multiply(np.multiply(np.multiply(b(:,:,:,i),applicability),certainty),b(:,:,:,j)))))
    #             Q[j,i-1] = Q(i,j)
    #     clear('b','applicability','x','y','t')
    #     Qinv = inv(Q)
    #     # Convolutions in the t-direction
    #     kernelt0 = np.reshape(a, tuple(np.array([1,1,spatial_size])), order="F")
    #     kernelt1 = np.reshape(np.multiply(np.transpose((np.arange(- n,n+1))),a), tuple(np.array([1,1,spatial_size])), order="F")
    #     kernelt2 = np.reshape(np.multiply(np.transpose(((np.arange(- n,n+1)) ** 2)),a), tuple(np.array([1,1,spatial_size])), order="F")
    #     roit = region_of_interest + np.array([[- n,n],[- n,n],[0,0]])
    #     roit[:,1-1] = np.amax(roit(:,1),np.ones((3,1)))
    #     roit[:,2-1] = np.amin(roit(:,2),np.transpose(signal.shape))
    #     conv_t0 = conv3(signal,kernelt0,roit)
    #     conv_t1 = conv3(signal,kernelt1,roit)
    #     conv_t2 = conv3(signal,kernelt2,roit)
    #     # Convolutions in the y-direction
    #     kernely0 = np.reshape(kernelt0, tuple(np.array([1,spatial_size])), order="F")
    #     kernely1 = np.reshape(kernelt1, tuple(np.array([1,spatial_size])), order="F")
    #     kernely2 = np.reshape(kernelt2, tuple(np.array([1,spatial_size])), order="F")
    #     roiy = region_of_interest + np.array([[- n,n],[0,0],[0,0]])
    #     roiy[:,1-1] = np.amax(roiy(:,1),np.ones((3,1)))
    #     roiy[:,2-1] = np.amin(roiy(:,2),np.transpose(signal.shape))
    #     if diff(roiy(3,:)) == 0:
    #         roiy = roiy(np.arange(1,2+1),:)
    #     else:
    #         roiy[3,:-1] = roiy(3,:) + 1 - roiy(3,1)
    #     conv_t0y0 = conv3(conv_t0,kernely0,roiy)
    #     conv_t0y1 = conv3(conv_t0,kernely1,roiy)
    #     conv_t0y2 = conv3(conv_t0,kernely2,roiy)
    #     clear('conv_t0')
    #     conv_t1y0 = conv3(conv_t1,kernely0,roiy)
    #     conv_t1y1 = conv3(conv_t1,kernely1,roiy)
    #     clear('conv_t1')
    #     conv_t2y0 = conv3(conv_t2,kernely0,roiy)
    #     clear('conv_t2')
    #     # Convolutions in the x-direction
    #     kernelx0 = np.reshape(kernelt0, tuple(np.array([spatial_size,1])), order="F")
    #     kernelx1 = np.reshape(kernelt1, tuple(np.array([spatial_size,1])), order="F")
    #     kernelx2 = np.reshape(kernelt2, tuple(np.array([spatial_size,1])), order="F")
    #     roix = region_of_interest
    #     roix = roix(np.arange(1,np.asarray(conv_t0y0).ndim+1),:)
    #     roix[np.arange[2,end()+1],:-1] = roix(np.arange(2,end()+1),:) + 1 - np.matlib.repmat(roix(np.arange(2,end()+1),1),np.array([1,2]))
    #     conv_results = np.zeros((np.array([diff(np.transpose(region_of_interest)) + 1,10]),np.array([diff(np.transpose(region_of_interest)) + 1,10])))
    #     conv_results[:,:,:,1-1] = conv3(conv_t0y0,kernelx0,roix)
    #     conv_results[:,:,:,2-1] = conv3(conv_t0y0,kernelx1,roix)
    #     conv_results[:,:,:,5-1] = conv3(conv_t0y0,kernelx2,roix)
    #     clear('conv_t0y0')
    #     conv_results[:,:,:,3-1] = conv3(conv_t0y1,kernelx0,roix)
    #     conv_results[:,:,:,8-1] = conv3(conv_t0y1,kernelx1,roix) / 2
    #     clear('conv_t0y1')
    #     conv_results[:,:,:,6-1] = conv3(conv_t0y2,kernelx0,roix)
    #     clear('conv_t0y2')
    #     conv_results[:,:,:,4-1] = conv3(conv_t1y0,kernelx0,roix)
    #     conv_results[:,:,:,9-1] = conv3(conv_t1y0,kernelx1,roix) / 2
    #     clear('conv_t1y0')
    #     conv_results[:,:,:,10-1] = conv3(conv_t1y1,kernelx0,roix) / 2
    #     clear('conv_t1y1')
    #     conv_results[:,:,:,7-1] = conv3(conv_t2y0,kernelx0,roix)
    #     clear('conv_t2y0')
    #     # Apply the inverse metric.
    #     conv_results[:,:,:,2-1] = Qinv(2,2) * conv_results(:,:,:,2)
    #     conv_results[:,:,:,3-1] = Qinv(3,3) * conv_results(:,:,:,3)
    #     conv_results[:,:,:,4-1] = Qinv(4,4) * conv_results(:,:,:,4)
    #     conv_results[:,:,:,5-1] = Qinv(5,5) * conv_results(:,:,:,5) + Qinv(5,1) * conv_results(:,:,:,1)
    #     conv_results[:,:,:,6-1] = Qinv(6,6) * conv_results(:,:,:,6) + Qinv(6,1) * conv_results(:,:,:,1)
    #     conv_results[:,:,:,7-1] = Qinv(7,7) * conv_results(:,:,:,7) + Qinv(7,1) * conv_results(:,:,:,1)
    #     conv_results[:,:,:,8-1] = Qinv(8,8) * conv_results(:,:,:,8)
    #     conv_results[:,:,:,9-1] = Qinv(9,9) * conv_results(:,:,:,9)
    #     conv_results[:,:,:,10-1] = Qinv(10,10) * conv_results(:,:,:,10)
    #     # Build tensor components.
    #     # It's more efficient in matlab code to do a small matrix
    #     # multiplication "manually" in parallell over all the points
    #     # than doing a multiple loop over the points and computing the
    #     # matrix products "automatically".
    #     # The tensor is of the form T=A*A'+gamma*b*b', where A and b are
    #     # composed from the convolution results according to
    #     #   [5  8  9]    [2]
    #     # A=[8  6 10], b=[3].
    #     #   [9 10  7]    [4]
    #     # Thus (excluding gamma)
    #     #   [5*5+8*8+9*9+2*2  5*8+6*8+9*10+2*3  5*9+8*10+7*9+2*4 ]
    #     # T=[5*8+6*8+9*10+2*3 8*8+6*6+10*10+3*3 8*9+6*10+7*10+3*4].
    #     #   [5*9+8*10+7*9+2*4 8*9+6*10+7*10+3*4 9*9+10*10+7*7+4*4]
    #     T = np.zeros((np.array([diff(np.transpose(region_of_interest)) + 1,3,3]),np.array([diff(np.transpose(region_of_interest)) + 1,3,3])))
    #     T[:,:,:,1,1-1] = conv_results(:,:,:,5) ** 2 + conv_results(:,:,:,8) ** 2 + conv_results(:,:,:,9) ** 2 + gamma * conv_results(:,:,:,2) ** 2
    #     T[:,:,:,2,2-1] = conv_results(:,:,:,8) ** 2 + conv_results(:,:,:,6) ** 2 + conv_results(:,:,:,10) ** 2 + gamma * conv_results(:,:,:,3) ** 2
    #     T[:,:,:,3,3-1] = conv_results(:,:,:,9) ** 2 + conv_results(:,:,:,10) ** 2 + conv_results(:,:,:,7) ** 2 + gamma * conv_results(:,:,:,4) ** 2
    #     T[:,:,:,1,2-1] = np.multiply((conv_results(:,:,:,5) + conv_results(:,:,:,6)),conv_results(:,:,:,8)) + np.multiply(conv_results(:,:,:,9),conv_results(:,:,:,10)) + np.multiply(gamma * conv_results(:,:,:,2),conv_results(:,:,:,3))
    #     T[:,:,:,2,1-1] = T(:,:,:,1,2)
    #     T[:,:,:,1,3-1] = np.multiply((conv_results(:,:,:,5) + conv_results(:,:,:,7)),conv_results(:,:,:,9)) + np.multiply(conv_results(:,:,:,8),conv_results(:,:,:,10)) + np.multiply(gamma * conv_results(:,:,:,2),conv_results(:,:,:,4))
    #     T[:,:,:,3,1-1] = T(:,:,:,1,3)
    #     T[:,:,:,2,3-1] = np.multiply((conv_results(:,:,:,6) + conv_results(:,:,:,7)),conv_results(:,:,:,10)) + np.multiply(conv_results(:,:,:,8),conv_results(:,:,:,9)) + np.multiply(gamma * conv_results(:,:,:,3),conv_results(:,:,:,4))
    #     T[:,:,:,3,2-1] = T(:,:,:,2,3)
    #     T = squeeze(T)
    # elif N == 4:
    #     # Set up applicability and basis functions.
    #     applicability = outerprod(a,a,a,a)
    #     x,y,z,t = ndgrid(np.arange(- n,n+1))
    #     b = cat(5,np.ones((x.shape,x.shape)),x,y,z,t,np.multiply(x,x),np.multiply(y,y),np.multiply(z,z),np.multiply(t,t),np.multiply(x,y),np.multiply(x,z),np.multiply(x,t),np.multiply(y,z),np.multiply(y,t),np.multiply(z,t))
    #     nb = b.shape[5-1]
    #     # Compute the inverse metric.
    #     Q = np.zeros((nb,nb))
    #     for i in np.arange(1,nb+1).reshape(-1):
    #         for j in np.arange(i,nb+1).reshape(-1):
    #             Q[i,j-1] = sum(sum(sum(sum(np.multiply(np.multiply(np.multiply(b(:,:,:,:,i),applicability),certainty),b(:,:,:,:,j))))))
    #             Q[j,i-1] = Q(i,j)
    #     clear('b','applicability','x','y','z','t')
    #     Qinv = inv(Q)
    #     # Convolutions in the t-direction
    #     kernelt0 = np.reshape(a, tuple(np.array([1,1,1,spatial_size])), order="F")
    #     kernelt1 = np.reshape(np.multiply(np.transpose((np.arange(- n,n+1))),a), tuple(np.array([1,1,1,spatial_size])), order="F")
    #     kernelt2 = np.reshape(np.multiply(np.transpose(((np.arange(- n,n+1)) ** 2)),a), tuple(np.array([1,1,1,spatial_size])), order="F")
    #     roit = region_of_interest + np.array([[- n,n],[- n,n],[- n,n],[0,0]])
    #     roit[:,1-1] = np.amax(roit(:,1),np.ones((4,1)))
    #     roit[:,2-1] = np.amin(roit(:,2),np.transpose(signal.shape))
    #     conv_t0 = conv3(signal,kernelt0,roit)
    #     conv_t1 = conv3(signal,kernelt1,roit)
    #     conv_t2 = conv3(signal,kernelt2,roit)
    #     # Convolutions in the z-direction
    #     kernelz0 = np.reshape(kernelt0, tuple(np.array([1,1,spatial_size])), order="F")
    #     kernelz1 = np.reshape(kernelt1, tuple(np.array([1,1,spatial_size])), order="F")
    #     kernelz2 = np.reshape(kernelt2, tuple(np.array([1,1,spatial_size])), order="F")
    #     roiz = region_of_interest + np.array([[- n,n],[- n,n],[0,0],[0,0]])
    #     roiz[:,1-1] = np.amax(roiz(:,1),np.ones((4,1)))
    #     roiz[:,2-1] = np.amin(roiz(:,2),np.transpose(signal.shape))
    #     if diff(roiz(4,:)) == 0:
    #         roiz = roiz(np.arange(1,2+1),:)
    #     else:
    #         roiz[4,:-1] = roiz(4,:) + 1 - roiz(4,1)
    #     conv_t0z0 = conv3(conv_t0,kernelz0,roiz)
    #     conv_t0z1 = conv3(conv_t0,kernelz1,roiz)
    #     conv_t0z2 = conv3(conv_t0,kernelz2,roiz)
    #     clear('conv_t0')
    #     conv_t1z0 = conv3(conv_t1,kernelz0,roiz)
    #     conv_t1z1 = conv3(conv_t1,kernelz1,roiz)
    #     clear('conv_t1')
    #     conv_t2z0 = conv3(conv_t2,kernelz0,roiz)
    #     clear('conv_t2')
    #     # Convolutions in the y-direction
    #     kernely0 = np.reshape(kernelt0, tuple(np.array([1,spatial_size])), order="F")
    #     kernely1 = np.reshape(kernelt1, tuple(np.array([1,spatial_size])), order="F")
    #     kernely2 = np.reshape(kernelt2, tuple(np.array([1,spatial_size])), order="F")
    #     roiy = region_of_interest + np.array([[- n,n],[0,0],[0,0],[0,0]])
    #     roiy[:,1-1] = np.amax(roiy(:,1),np.ones((4,1)))
    #     roiy[:,2-1] = np.amin(roiy(:,2),np.transpose(signal.shape))
    #     roiy = roiy(np.arange(1,np.asarray(conv_t0z0).ndim+1),:)
    #     roiy[np.arange[3,end()+1],:-1] = roiy(np.arange(3,end()+1),:) + 1 - np.matlib.repmat(roiy(np.arange(3,end()+1),1),np.array([1,2]))
    #     conv_t0z0y0 = conv3(conv_t0z0,kernely0,roiy)
    #     conv_t0z0y1 = conv3(conv_t0z0,kernely1,roiy)
    #     conv_t0z0y2 = conv3(conv_t0z0,kernely2,roiy)
    #     clear('conv_t0z0')
    #     conv_t0z1y0 = conv3(conv_t0z1,kernely0,roiy)
    #     conv_t0z1y1 = conv3(conv_t0z1,kernely1,roiy)
    #     clear('conv_t0z1')
    #     conv_t0z2y0 = conv3(conv_t0z2,kernely0,roiy)
    #     clear('conv_t0z2')
    #     conv_t1z0y0 = conv3(conv_t1z0,kernely0,roiy)
    #     conv_t1z0y1 = conv3(conv_t1z0,kernely1,roiy)
    #     clear('conv_t1z0')
    #     conv_t1z1y0 = conv3(conv_t1z1,kernely0,roiy)
    #     clear('conv_t1z1')
    #     conv_t2z0y0 = conv3(conv_t2z0,kernely0,roiy)
    #     clear('conv_t2z0')
    #     # Convolutions in the x-direction
    #     kernelx0 = np.reshape(kernelt0, tuple(np.array([spatial_size,1])), order="F")
    #     kernelx1 = np.reshape(kernelt1, tuple(np.array([spatial_size,1])), order="F")
    #     kernelx2 = np.reshape(kernelt2, tuple(np.array([spatial_size,1])), order="F")
    #     roix = region_of_interest
    #     roix = roix(np.arange(1,np.asarray(conv_t0z0y0).ndim+1),:)
    #     roix[np.arange[2,end()+1],:-1] = roix(np.arange(2,end()+1),:) + 1 - np.matlib.repmat(roix(np.arange(2,end()+1),1),np.array([1,2]))
    #     conv_results = np.zeros((np.array([diff(np.transpose(region_of_interest)) + 1,15]),np.array([diff(np.transpose(region_of_interest)) + 1,15])))
    #     conv_results[:,:,:,:,1-1] = conv3(conv_t0z0y0,kernelx0,roix)
    #     conv_results[:,:,:,:,2-1] = conv3(conv_t0z0y0,kernelx1,roix)
    #     conv_results[:,:,:,:,6-1] = conv3(conv_t0z0y0,kernelx2,roix)
    #     clear('conv_t0z0y0')
    #     conv_results[:,:,:,:,3-1] = conv3(conv_t0z0y1,kernelx0,roix)
    #     conv_results[:,:,:,:,10-1] = conv3(conv_t0z0y1,kernelx1,roix) / 2
    #     clear('conv_t0z0y1')
    #     conv_results[:,:,:,:,7-1] = conv3(conv_t0z0y2,kernelx0,roix)
    #     clear('conv_t0z0y2')
    #     conv_results[:,:,:,:,4-1] = conv3(conv_t0z1y0,kernelx0,roix)
    #     conv_results[:,:,:,:,11-1] = conv3(conv_t0z1y0,kernelx1,roix) / 2
    #     clear('conv_t0z1y0')
    #     conv_results[:,:,:,:,13-1] = conv3(conv_t0z1y1,kernelx0,roix) / 2
    #     clear('conv_t0z1y1')
    #     conv_results[:,:,:,:,8-1] = conv3(conv_t0z2y0,kernelx0,roix)
    #     clear('conv_t0z2y0')
    #     conv_results[:,:,:,:,5-1] = conv3(conv_t1z0y0,kernelx0,roix)
    #     conv_results[:,:,:,:,12-1] = conv3(conv_t1z0y0,kernelx1,roix) / 2
    #     clear('conv_t1z0y0')
    #     conv_results[:,:,:,:,14-1] = conv3(conv_t1z0y1,kernelx0,roix) / 2
    #     clear('conv_t1z0y1')
    #     conv_results[:,:,:,:,15-1] = conv3(conv_t1z1y0,kernelx0,roix) / 2
    #     clear('conv_t1z1y0')
    #     conv_results[:,:,:,:,9-1] = conv3(conv_t2z0y0,kernelx0,roix)
    #     clear('conv_t2z0y0')
    #     # Apply the inverse metric.
    #     conv_results[:,:,:,:,2-1] = Qinv(2,2) * conv_results(:,:,:,:,2)
    #     conv_results[:,:,:,:,3-1] = Qinv(3,3) * conv_results(:,:,:,:,3)
    #     conv_results[:,:,:,:,4-1] = Qinv(4,4) * conv_results(:,:,:,:,4)
    #     conv_results[:,:,:,:,5-1] = Qinv(5,5) * conv_results(:,:,:,:,5)
    #     conv_results[:,:,:,:,6-1] = Qinv(6,6) * conv_results(:,:,:,:,6) + Qinv(6,1) * conv_results(:,:,:,:,1)
    #     conv_results[:,:,:,:,7-1] = Qinv(7,7) * conv_results(:,:,:,:,7) + Qinv(7,1) * conv_results(:,:,:,:,1)
    #     conv_results[:,:,:,:,8-1] = Qinv(8,8) * conv_results(:,:,:,:,8) + Qinv(8,1) * conv_results(:,:,:,:,1)
    #     conv_results[:,:,:,:,9-1] = Qinv(9,9) * conv_results(:,:,:,:,9) + Qinv(9,1) * conv_results(:,:,:,:,1)
    #     conv_results[:,:,:,:,10-1] = Qinv(10,10) * conv_results(:,:,:,:,10)
    #     conv_results[:,:,:,:,11-1] = Qinv(11,11) * conv_results(:,:,:,:,11)
    #     conv_results[:,:,:,:,12-1] = Qinv(12,12) * conv_results(:,:,:,:,12)
    #     conv_results[:,:,:,:,13-1] = Qinv(13,13) * conv_results(:,:,:,:,13)
    #     conv_results[:,:,:,:,14-1] = Qinv(14,14) * conv_results(:,:,:,:,14)
    #     conv_results[:,:,:,:,15-1] = Qinv(15,15) * conv_results(:,:,:,:,15)
    #     # Build tensor components.
    #     # It's more efficient in matlab code to do a small matrix
    #     # multiplication "manually" in parallell over all the points
    #     # than doing a multiple loop over the points and computing the
    #     # matrix products "automatically".
    #     # The tensor is of the form T=A*A'+gamma*b*b', where A and b are
    #     # composed from the convolution results according to
    #     #   [6  10 11 12]    [2]
    #     #   [10  7 13 14]    [3]
    #     # A=[11 13  8 15], b=[4].
    #     #   [12 14 15  9]    [5]
    #     # Thus (excluding gamma)
    #     #   [6*6+10*10+11*11+12*12+2*2 6*10+7*10+11*13+12*14+2*3
    #     #   [6*10+7*10+11*13+12*14+2*3 10*10+7*7+13*13+14*14+3*3
    #     # T=[6*11+10*13+8*11+12*15+2*4 10*11+7*13+8*13+14*15+3*4
    #     #   [6*12+10*14+11*15+9*12+2*5 10*12+7*14+13*15+9*14+3*5
    #     #    6*11+10*13+8*11+12*15+2*4 6*12+10*14+11*15+9*12+2*5]
    #     #    10*11+7*13+8*13+14*15+3*4 10*12+7*14+13*15+9*14+3*5]
    #     #    11*11+13*13+8*8+15*15+4*4 11*12+13*14+8*15+9*15+4*5].
    #     #    11*12+13*14+8*15+9*15+4*5 12*12+14*14+15*15+9*9+5*5]
    #     T = np.zeros((np.array([diff(np.transpose(region_of_interest)) + 1,3,3]),np.array([diff(np.transpose(region_of_interest)) + 1,3,3])))
    #     T[:,:,:,:,1,1-1] = conv_results(:,:,:,:,6) ** 2 + conv_results(:,:,:,:,10) ** 2 + conv_results(:,:,:,:,11) ** 2 + conv_results(:,:,:,:,12) ** 2 + gamma * conv_results(:,:,:,:,2) ** 2
    #     T[:,:,:,:,2,2-1] = conv_results(:,:,:,:,10) ** 2 + conv_results(:,:,:,:,7) ** 2 + conv_results(:,:,:,:,13) ** 2 + conv_results(:,:,:,:,14) ** 2 + gamma * conv_results(:,:,:,:,3) ** 2
    #     T[:,:,:,:,3,3-1] = conv_results(:,:,:,:,11) ** 2 + conv_results(:,:,:,:,13) ** 2 + conv_results(:,:,:,:,8) ** 2 + conv_results(:,:,:,:,15) ** 2 + gamma * conv_results(:,:,:,:,4) ** 2
    #     T[:,:,:,:,4,4-1] = conv_results(:,:,:,:,11) ** 2 + conv_results(:,:,:,:,13) ** 2 + conv_results(:,:,:,:,8) ** 2 + conv_results(:,:,:,:,15) ** 2 + gamma * conv_results(:,:,:,:,4) ** 2
    #     T[:,:,:,:,1,2-1] = np.multiply((conv_results(:,:,:,:,6) + conv_results(:,:,:,:,7)),conv_results(:,:,:,:,10)) + np.multiply(conv_results(:,:,:,:,11),conv_results(:,:,:,:,13)) + np.multiply(conv_results(:,:,:,:,12),conv_results(:,:,:,:,14)) + np.multiply(gamma * conv_results(:,:,:,:,2),conv_results(:,:,:,:,3))
    #     T[:,:,:,:,2,1-1] = T(:,:,:,:,1,2)
    #     T[:,:,:,:,1,3-1] = np.multiply((conv_results(:,:,:,:,6) + conv_results(:,:,:,:,8)),conv_results(:,:,:,:,11)) + np.multiply(conv_results(:,:,:,:,10),conv_results(:,:,:,:,13)) + np.multiply(conv_results(:,:,:,:,12),conv_results(:,:,:,:,15)) + np.multiply(gamma * conv_results(:,:,:,:,2),conv_results(:,:,:,:,4))
    #     T[:,:,:,:,3,1-1] = T(:,:,:,:,1,3)
    #     T[:,:,:,:,1,4-1] = np.multiply((conv_results(:,:,:,:,6) + conv_results(:,:,:,:,9)),conv_results(:,:,:,:,12)) + np.multiply(conv_results(:,:,:,:,10),conv_results(:,:,:,:,14)) + np.multiply(conv_results(:,:,:,:,11),conv_results(:,:,:,:,15)) + np.multiply(gamma * conv_results(:,:,:,:,2),conv_results(:,:,:,:,5))
    #     T[:,:,:,:,4,1-1] = T(:,:,:,:,1,4)
    #     T[:,:,:,:,2,3-1] = np.multiply((conv_results(:,:,:,:,7) + conv_results(:,:,:,:,8)),conv_results(:,:,:,:,13)) + np.multiply(conv_results(:,:,:,:,10),conv_results(:,:,:,:,11)) + np.multiply(conv_results(:,:,:,:,14),conv_results(:,:,:,:,15)) + np.multiply(gamma * conv_results(:,:,:,:,3),conv_results(:,:,:,:,4))
    #     T[:,:,:,:,3,2-1] = T(:,:,:,:,2,3)
    #     T[:,:,:,:,2,4-1] = np.multiply((conv_results(:,:,:,:,7) + conv_results(:,:,:,:,9)),conv_results(:,:,:,:,14)) + np.multiply(conv_results(:,:,:,:,10),conv_results(:,:,:,:,12)) + np.multiply(conv_results(:,:,:,:,13),conv_results(:,:,:,:,15)) + np.multiply(gamma * conv_results(:,:,:,:,3),conv_results(:,:,:,:,5))
    #     T[:,:,:,:,4,2-1] = T(:,:,:,:,2,4)
    #     T[:,:,:,:,3,4-1] = np.multiply((conv_results(:,:,:,:,8) + conv_results(:,:,:,:,9)),conv_results(:,:,:,:,15)) + np.multiply(conv_results(:,:,:,:,11),conv_results(:,:,:,:,12)) + np.multiply(conv_results(:,:,:,:,13),conv_results(:,:,:,:,14)) + np.multiply(gamma * conv_results(:,:,:,:,4),conv_results(:,:,:,:,5))
    #     T[:,:,:,:,4,3-1] = T(:,:,:,:,3,4)
    #     T = squeeze(T)
    else:
        raise Exception('More than four dimensions are not supported.')

    params = {
        'spatial_size' : spatial_size,
        'region_of_interest' : region_of_interest,
        'gamma' : gamma,
        'sigma' : sigma,
        'delta' : delta,
        'certainty' : certainty,
    }

    return T,params

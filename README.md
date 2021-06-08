# Spatial domain toolbox

This is the reimplementaiton in Python of the Matlab and C package `spatial_domain_toolbox` written by Gunnar Farnebäck.
It contains the implementations of some of the algorithms described in his Ph.D. thesis
"Polynomial Expansion for Orientation and Motion Estimation" but has subsequently been extended with other algorithms.
The original package can be found at [https://github.com/GunnarFarneback/spatial_domain_toolbox]

A common theme is that the algorithms work on multidimensional signals and are designed
and used directly in the spatial (or spatiotemporal) domain, without
involving e.g. Fourier or wavelet transforms. Some functions even more
generally apply to  multidimensional arrays.

This ability to work with arrays of arbitrary dimensions has been extended during the Python reimplementation by Mathieu Leocmach.

Although most of the files follow the structure of the original implementation,
the file `memory_efficient.py` refactors the code in an object oriented way.
This file is standalone and contains all the necessary classes and functions 
to perform optical flow on N-dimensional images while limitting memory footprint.

Numba is used to compile critical parts of the code, however this implementation will not beat more modern GPU acceleration in pure speed.
What this implementation provides is 
 - arbitrary dimensionality
 - access to intermediate data
 - low memory use 

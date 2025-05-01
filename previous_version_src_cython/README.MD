This is my previous naive implementation of cython to run cholesky on the gpu. The next step is to bind directly to the function "MPSMatrixDecompositionCholesky" from Metal Perfomance Shaders using cython.

The file cholesky_cython.pyx has the (bad) implementation of cholesky that runs on the gpu. Only the files cholesky_cython.pyx and setup.py are required to compile. The build command is
 python setup.py build_ext --inplace

! Without changes, The file example_cython.py can be run to compare the calculation of numpy and cython !
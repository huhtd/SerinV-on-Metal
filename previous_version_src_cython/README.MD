This is my previous naive implementation of cython to run cholesky on the gpu. The next step is to bind directly to the function "MPSMatrixDecompositionCholesky" from Metal Perfomance Shaders using cython.

The file cholesky_cython.pyx has the (bad) implementation of cholesky that runs on the gpu. 



1. Run the command "python setup.py build_ext --inplace"
2. Run the file "example_cython.py"

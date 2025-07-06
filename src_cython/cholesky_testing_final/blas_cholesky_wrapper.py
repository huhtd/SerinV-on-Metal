import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

# Load shared library
blas = ctypes.CDLL('./libblascholesky.dylib')

# Define function signature
blas.cpu_blas_cholesky.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_uint]
blas.cpu_blas_cholesky.restype = None

def cpu_blas_cholesky_decompose(A: np.ndarray) -> np.ndarray:
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    A = np.array(A, dtype=np.float32, order='C')
    blas.cpu_blas_cholesky(A, A.shape[0])
    return A

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

# Load the shared library
mps = ctypes.CDLL('./libmpscholesky.dylib')

# Define the function signature
mps.gpu_mps_cholesky.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_uint
]
mps.gpu_mps_cholesky.restype = None

def gpu_mps_cholesky_decompose(A: np.ndarray) -> np.ndarray:
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    A = np.array(A, dtype=np.float32, order='C')  # Ensure memory layout is correct
    mps.gpu_mps_cholesky(A, A.shape[0])
    return A

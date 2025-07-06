import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

# Load shared library
mps = ctypes.CDLL('./libmpsmul.dylib')

# Declare function prototype
mps.gpu_mps_multiply.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # A
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # B
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # C
    ctypes.c_uint                                        # n
]
mps.gpu_mps_multiply.restype = None

def gpu_mps_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape == B.shape and A.shape[0] == A.shape[1], "Matrices must be square and same size"
    n = A.shape[0]
    C = np.zeros_like(A, dtype=np.float32)
    mps.gpu_mps_multiply(A.astype(np.float32), B.astype(np.float32), C, ctypes.c_uint(n))
    return C

import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

# Load shared libraries
mps = ctypes.CDLL('./libmpsmul.dylib')
blas = ctypes.CDLL('./libblasmul.dylib')

# Define MPS function signature
mps.gpu_mps_multiply.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_uint
]
mps.gpu_mps_multiply.restype = None

# Define BLAS function signature
blas.cpu_blas_multiply.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_uint
]
blas.cpu_blas_multiply.restype = None

def gpu_mps_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    C = np.zeros_like(A, dtype=np.float32)
    mps.gpu_mps_multiply(A, B, C, n)
    return C

def cpu_blas_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    C = np.zeros_like(A, dtype=np.float32)
    blas.cpu_blas_multiply(A, B, C, n)
    return C

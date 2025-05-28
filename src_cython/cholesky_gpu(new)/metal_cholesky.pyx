# distutils: language = c
# distutils: extra_compile_args = -fobjc-arc
# distutils: extra_link_args = -framework Metal -framework MetalPerformanceShaders

import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound

cdef extern from "mps_cholesky.h":
    void mps_cholesky(float *A, int N)      # in-place API

@boundscheck(False)
@wraparound(False)
def cholesky_gpu_inplace(cnp.ndarray[cnp.float32_t, ndim=2] A not None):
    """
    In-place Cholesky (lower-triangular) on Apple-silicon GPUs/AMX.
    The input array **A** is overwritten with its factor **L**.
    Returns the same array for convenience.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if A.dtype != np.float32:
        raise ValueError("dtype must be float32")
    if not A.flags.c_contiguous:
        raise ValueError("array must be C-contiguous")

    cdef int N = A.shape[0]
    mps_cholesky(<float *>A.data, N)
    return A           # modified in place

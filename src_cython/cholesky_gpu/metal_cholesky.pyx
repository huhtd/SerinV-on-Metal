# distutils: language = c
# distutils: extra_compile_args = -fobjc-arc
# distutils: extra_link_args = -framework Metal -framework MetalPerformanceShaders

import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound

cdef extern from "mps_cholesky.h":      # same base name, no extra header needed
    void mps_cholesky(const float*, float*, int)

@boundscheck(False)
@wraparound(False)
def cholesky_gpu(cnp.ndarray[cnp.float32_t, ndim=2] A):
    """
    Cholesky on Apple GPU (Metal / MPS).  Returns lower-triangular L.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if A.dtype != np.float32:
        A = np.ascontiguousarray(A, dtype=np.float32)

    cdef int N = A.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=2] L = np.empty_like(A, order='C')
    mps_cholesky(<const float*>A.data, <float*>L.data, N)
    return L

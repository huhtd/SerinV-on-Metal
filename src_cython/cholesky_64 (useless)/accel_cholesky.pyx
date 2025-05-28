# distutils: language = c
# distutils: extra_compile_args = -fobjc-arc
# distutils: extra_link_args = -framework Accelerate

import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound

cdef extern from "mps_cholesky.h":
    void mps_cholesky(double *A, int N)

@boundscheck(False)
@wraparound(False)
def cholesky_cpu_inplace(cnp.ndarray[cnp.float64_t, ndim=2] A not None):
    """
    In-place Cholesky (lower) in float64 via Accelerate/AMX.
    Overwrites A and returns it for convenience.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if not A.flags.c_contiguous:
        raise ValueError("Array must be C-contiguous")

    cdef int N = A.shape[0]
    mps_cholesky(<double *>A.data, N)
    return A

#build with python setup.py build_ext --inplace (in this directory)
# cholesky_cython.pyx

import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound
# Import the float version of sqrt (sqrtf) from the C math library
from libc.math cimport sqrtf
#np.import_array()

@boundscheck(False)
@wraparound(False)
def cholesky_cython(cnp.float32_t[:, ::1] A):
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i, j, k
    cdef float temp_sum
    cdef cnp.float32_t[:, ::1] L = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i+1):
            temp_sum = A[i, j]
            for k in range(j):
                temp_sum -= L[i, k] * L[j, k]
            if i == j:
                if temp_sum <= 0.0:
                    raise ValueError("Matrix is not positive definite.")
                # Use sqrtf instead of np.sqrtf
                L[i, j] = sqrtf(temp_sum)
            else:
                L[i, j] = temp_sum / L[j, j]

    return L

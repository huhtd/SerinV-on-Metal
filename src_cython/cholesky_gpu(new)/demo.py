import time
import numpy as np
from Metal import MTLCreateSystemDefaultDevice
import scipy

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count()))

# --- GPU sanity check ----------------------------------------------------
if MTLCreateSystemDefaultDevice() is None:
    raise RuntimeError("No Metal GPU device found.")

# in-place GPU routine
from metal_cholesky import cholesky_gpu_inplace   # <- new wrapper name

# -------------------------------------------------------------------------

# ------------------------------------------------------------------ util
def generate_posdef_matrix(n: int, dtype=np.float32) -> np.ndarray:
    """
    Why not use np.random as before? because it is way too slow for large matrices.
    The decomposition time is not influenced by the complexity of the matrix, but the allocation time is.
    Symmetric positive-definite tridiagonal Toeplitz matrix:
        2 on the diagonal, -1 on the first sub- and super-diagonal.
    Eigenvalues:  2-2·cos(k·π/(n+1))  > 0   → guaranteed PD.
    Construction is O(n) and uses < 3·n memory while building.
    """
    M = 2 * np.eye(n,   dtype=dtype)           # main diagonal
    M += -1 * np.eye(n, k=1, dtype=dtype)      # super-diagonal
    M += -1 * np.eye(n, k=-1, dtype=dtype)     # sub-diagonal
    return M

if __name__ == "__main__":
    mode      = 0          # 0 = both, 1 = GPU only, 2 = CPU only
    N         = 8000       # matrix size
    num_mats  = 4         # how many matrices to factor




    # ------------------------------------------------------------------ GPU
    if mode in (0, 1):

        gpu_mats = [generate_posdef_matrix(N) for _ in range(num_mats)]

        t0 = time.perf_counter()
        for M_gpu in gpu_mats:
            cholesky_gpu_inplace(M_gpu)
        gpu_total = time.perf_counter() - t0
        gpu_avg   = gpu_total / num_mats

    # ------------------------------------------------------------------ CPU
    if mode in (0, 2):


        cpu_mats = [generate_posdef_matrix(N) for _ in range(num_mats)]

        t1 = time.perf_counter()
        cpu_results = []
        for M_cpu in cpu_mats:
            cpu_results.append(np.linalg.cholesky(M_cpu))         # returns new array
        '''
        if we use scipy.linalg.cholesky it is much faster, but the AMX code is still 
        several magnitudes faster especially for large matrices
        '''
        cpu_total = time.perf_counter() - t1
        cpu_avg   = cpu_total / num_mats

    # ---------------------------------------------------------------- stats
    print(f"Size Matrix = {N},  num_mats = {num_mats}")
    if mode in (0, 1):
        print(f"Avg AMX time : {gpu_avg*1e3:8.1f} ms")
    if mode in (0, 2):
        print(f"Avg CPU time : {cpu_avg*1e3:8.1f} ms")

    # speed-up & correctness check
    if mode == 0:
        print(f"Speed-up AMX / CPU : {cpu_avg / gpu_avg:5.2f}×")

        # compare last pair of results
        #in this version the comparison is wrong, since we calculate on different matrices. 
        diff = np.linalg.norm(M_gpu - cpu_results[-1])
        print(f"‖L_gpu − L_np‖_F = {diff:.2e}")
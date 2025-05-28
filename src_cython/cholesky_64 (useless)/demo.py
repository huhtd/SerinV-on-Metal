import time
import numpy as np
from accel_cholesky import cholesky_cpu_inplace
import scipy

# ---------------------------------------------------------------- utility
def generate_posdef(n: int) -> np.ndarray:
    """Return an n×n positive-definite float64 matrix (row-major)."""
    A = np.random.rand(n, n)
    M = A @ A.T
    M[np.diag_indices_from(M)] += 1.0
    return M.astype(np.float64, copy=False)

# ---------------------------------------------------------------- params
N          = 5000   # matrix dimension
BATCHES    = 10     # how many matrices to factor
mats       = [generate_posdef(N) for _ in range(BATCHES)]

# ---------------------------------------------------------------- AMX / Accelerate
# warm-up on a tiny matrix (builds LAPACK thread pool etc.)
cholesky_cpu_inplace(mats[0][:128, :128].copy())

t0 = time.perf_counter()
amx_results = []
for M in mats:
    A = M.copy()                     # writable copy; will be overwritten
    cholesky_cpu_inplace(A)
    amx_results.append(A)            # keep for later diff check
amx_total = time.perf_counter() - t0
amx_avg   = amx_total / BATCHES

# ---------------------------------------------------------------- NumPy / OpenBLAS
_ = np.linalg.cholesky(mats[0][:128, :128])        # warm-up

t1 = time.perf_counter()

np_results = [scipy.linalg.cholesky(M, lower=True, overwrite_a=True, check_finite=False) for M in mats]  # allocates new array
np_total   = time.perf_counter() - t1
np_avg     = np_total / BATCHES

# ---------------------------------------------------------------- report
print(f"Matrix size : {N} × {N} (float64)")
print(f"Batches     : {BATCHES}")
print(f"Accelerate/AMX avg : {amx_avg*1e3:8.1f} ms")
print(f"NumPy/OpenBLAS avg : {np_avg*1e3:8.1f} ms")
print(f"Speed-up AMX / NumPy : {np_avg/amx_avg:5.2f}×")

# spot-check last pair
diff = np.linalg.norm(amx_results[-1] - np_results[-1])
print(f"‖L_AMX − L_NumPy‖_F = {diff:.2e}")

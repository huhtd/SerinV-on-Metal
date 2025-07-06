import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
import os

from blas_cholesky_wrapper import cpu_blas_cholesky_decompose
from gpu_mps_cholesky_wrapper import gpu_mps_cholesky_decompose


# Ensure output folder exists
os.makedirs("figures", exist_ok=True)

# Load precomputed matrices
data = np.load("precomputed_matrices_big.npz")
sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 7000, 10000, 15000]

def get_matrix(n: int) -> np.ndarray:
    return data[str(n)].copy()

numpy_times = []
blas_times = []
mps_times = []

for N in sizes:
    print(f"\nRunning Cholesky for size: {N}x{N}")
    A = get_matrix(N)

    # NumPy timing
    start = time.perf_counter_ns()
    L_numpy = scipy.linalg.cholesky(A).astype(np.float32)
    numpy_time = time.perf_counter_ns() - start
    numpy_times.append(numpy_time)
    print(f"NumPy took {numpy_time:.2f} ns")

    # BLAS Cholesky timing
    A_blas = A.copy()
    start = time.perf_counter_ns()
    L_blas = cpu_blas_cholesky_decompose(A_blas)
    blas_time = time.perf_counter_ns() - start
    blas_times.append(blas_time)
    print(f"BLAS (Accelerate) took {blas_time:.2f} ns")

    # MPS Cholesky (Metal GPU)
    A_mps = A.copy()
    start = time.perf_counter_ns()
    L_mps = gpu_mps_cholesky_decompose(A_mps)
    mps_time = time.perf_counter_ns() - start
    mps_times.append(mps_time)
    print(f"MPS (GPU) took {mps_time:.2f} ns")

    # Relative error
    rel_err_blas = np.linalg.norm(L_numpy - L_blas) / np.linalg.norm(L_numpy)
    rel_err_mps = np.linalg.norm(L_numpy - L_mps) / np.linalg.norm(L_numpy)
    print(f"Rel. error (NumPy vs BLAS): {rel_err_blas:.6e}")
    print(f"Rel. error (NumPy vs MPS):  {rel_err_mps:.6e}")

# Convert to seconds
numpy_times_s = np.array(numpy_times) * 1e-9
blas_times_s  = np.array(blas_times)  * 1e-9
mps_times_s   = np.array(mps_times)   * 1e-9
sizes_np = np.array(sizes)

# Reference curve (1/3 N^3 scaling for Cholesky)
N_ref = np.linspace(min(sizes_np), max(sizes_np), 100)
T_ref = (1/3) * 1e-12 * N_ref**3

### 1. Linear Plot ###
plt.figure(figsize=(10, 6))
plt.plot(sizes_np, numpy_times_s, 'r-o', label='NumPy (CPU)')
plt.plot(sizes_np, blas_times_s, 'b-o', label='BLAS (Accelerate)')
plt.plot(sizes_np, mps_times_s,  'g-o', label='MPS (Metal GPU)')
plt.plot(N_ref, T_ref, 'k--', label='Reference $\\frac{1}{3}N^3$')
plt.title("Cholesky Timing: NumPy vs BLAS vs MPS")
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Time (seconds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/cholesky_timing.pdf", format='pdf')
plt.show()

### 2. Log-Log Plot ###
plt.figure(figsize=(10, 6))
plt.loglog(sizes_np, numpy_times_s, 'r-o', label='NumPy (CPU)')
plt.loglog(sizes_np, blas_times_s, 'b-o', label='BLAS (Accelerate)')
plt.loglog(sizes_np, mps_times_s,  'g-o', label='MPS (Metal GPU)')
plt.loglog(N_ref, T_ref, 'k--', label='Reference $\\frac{1}{3}N^3$')
plt.title("Cholesky Timing (Log-Log) with $\\frac{1}{3}N^3$ Reference")
plt.xlabel("Matrix Size (log N)")
plt.ylabel("Time (log seconds)")
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig("figures/cholesky_timing_loglog.pdf", format='pdf')
plt.show()

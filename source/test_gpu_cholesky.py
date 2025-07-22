import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
import os

from matplotlib.ticker import ScalarFormatter
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

    j = 5  # Number of averaged runs

    # NumPy timing
    current_average_numpy = []
    # warm up run
    _ = scipy.linalg.cholesky(A).astype(np.float32)
    for i in range(j):
        start = time.perf_counter_ns()
        L_numpy = scipy.linalg.cholesky(A).astype(np.float32)
        numpy_time_current = time.perf_counter_ns() - start
        current_average_numpy.append(numpy_time_current)
    numpy_time = np.mean(current_average_numpy)
    numpy_times.append(numpy_time)
    print(f"NumPy took {numpy_time:.2f} ns on average over {j} runs")

    # BLAS Cholesky timing
    current_average_blas = []
    # warm up run
    _ = cpu_blas_cholesky_decompose(A.copy())
    for i in range(j):
        A_blas = A.copy()
        start = time.perf_counter_ns()
        L_blas = cpu_blas_cholesky_decompose(A_blas)
        blas_time_current = time.perf_counter_ns() - start
        current_average_blas.append(blas_time_current)
    blas_time = np.mean(current_average_blas)
    blas_times.append(blas_time)
    print(f"BLAS (Accelerate) took {blas_time:.2f} ns on average over {j} runs")

    # MPS Cholesky (Metal GPU)
    current_average_mps = []
    # warm up run
    _ = gpu_mps_cholesky_decompose(A.copy())
    for i in range(j):
        A_mps = A.copy()
        start = time.perf_counter_ns()
        L_mps = gpu_mps_cholesky_decompose(A_mps)
        mps_time_current = time.perf_counter_ns() - start
        current_average_mps.append(mps_time_current)
    mps_time = np.mean(current_average_mps)
    mps_times.append(mps_time)
    print(f"MPS (GPU) took {mps_time:.2f} ns on average over {j} runs")

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

# Trend Line curve (1/3 N^3 scaling for Cholesky)
N_ref = np.linspace(min(sizes_np), max(sizes_np), 100)
T_ref = (1/3) * 1e-12 * N_ref**3

### 1. Linear Plot ###
plt.figure(figsize=(10, 6))
plt.plot(sizes_np, numpy_times_s, 'r-o', label='NumPy (CPU)')
plt.plot(sizes_np, blas_times_s, 'b-o', label='BLAS (Accelerate)')
plt.plot(sizes_np, mps_times_s,  'g-o', label='MPS (Metal GPU)')
plt.title("Cholesky Timing: NumPy vs BLAS vs MPS")
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Time (seconds)")
plt.xticks(sizes_np, rotation=45)
plt.ylim(bottom=0)
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
plt.loglog(N_ref, T_ref, 'k--', label='Trend Line $\\frac{1}{3}N^3$')
plt.title("Cholesky Timing (Log-Log) with $\\frac{1}{3}N^3$ Trend Line")
plt.xlabel("Matrix Size (log N)")
plt.ylabel("Time (log seconds)")
plt.xticks(sizes_np, sizes_np, rotation=45)
plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig("figures/cholesky_timing_loglog.pdf", format='pdf')
plt.show()

import numpy as np
import time
import matplotlib.pyplot as plt
import os
from gpu_cpu_wrapper import gpu_mps_multiply, cpu_blas_multiply
from matplotlib.ticker import MultipleLocator, ScalarFormatter

# Ensure output folder exists
os.makedirs("figures", exist_ok=True)

# Load matrices and define sizes
data = np.load("precomputed_matrices_big.npz")
sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 7000, 10000, 15000]

def get_matrix(n: int) -> np.ndarray:
    return data[str(n)].copy()

# Store timings in nanoseconds
blas_times = []
mps_times = []
numpy_times = []

# Benchmark loop
for N in sizes:
    print(f"\nRunning size: {N}x{N}")
    A = get_matrix(N)
    B = get_matrix(N)
    
    j = 5  # Number of averaged runs

    # NumPy
    current_average_numpy = []
    #warm up run
    _ = A @ B
    for i in range (j):
        # Measure time for NumPy multiplication
        start = time.perf_counter_ns()
        C_numpy = A @ B
        numpy_time_current = time.perf_counter_ns() - start
        current_average_numpy.append(numpy_time_current)
    numpy_time = np.mean(current_average_numpy)
    numpy_times.append(numpy_time)
    print(f"NumPy took {numpy_time:.2f} ns on average over {j} runs")

    # CPU BLAS
    current_average_blas = []
    # warm up run
    _ = cpu_blas_multiply(A, B)
    for i in range(j):
        # Measure time for BLAS multiplication
        start = time.perf_counter_ns()
        C_cpu = cpu_blas_multiply(A, B)
        blas_time_current = time.perf_counter_ns() - start
        current_average_blas.append(blas_time_current)
    blas_time = np.mean(current_average_blas)
    blas_times.append(blas_time)
    print(f"CPU (BLAS) took {blas_time:.2f} ns on average over {j} runs")

    # GPU MPS
    current_average_mps = []
    # warm up run
    _ = gpu_mps_multiply(A, B)
    for i in range(j):
        # Measure time for MPS multiplication
        start = time.perf_counter_ns()
        C_gpu = gpu_mps_multiply(A, B)
        mps_time_current = time.perf_counter_ns() - start
        current_average_mps.append(mps_time_current)
    mps_time = np.mean(current_average_mps)
    mps_times.append(mps_time)
    print(f"GPU (MPS) took {mps_time:.2f} ns on average over {j} runs")

    # Relative error check
    rel_err_blas = np.linalg.norm(C_numpy - C_cpu) / np.linalg.norm(C_numpy)
    rel_err_mps  = np.linalg.norm(C_numpy - C_gpu) / np.linalg.norm(C_numpy)
    print(f"Rel. error (NumPy vs BLAS): {rel_err_blas:.6e}")
    print(f"Rel. error (NumPy vs MPS):  {rel_err_mps:.6e}")

# Convert to seconds
sizes_np = np.array(sizes)
numpy_times_s = np.array(numpy_times) * 1e-9
blas_times_s  = np.array(blas_times)  * 1e-9
mps_times_s   = np.array(mps_times)   * 1e-9

### 1. Linear Plot with rotated ticks and 0.5s y-tick ###
plt.figure(figsize=(10, 6))
plt.plot(sizes_np, blas_times_s, 'b-o', label='CPU (BLAS - Accelerate)')
plt.plot(sizes_np, mps_times_s, 'g-o', label='GPU (MPS)')
plt.plot(sizes_np, numpy_times_s, 'r-o', label='CPU (NumPy)')
plt.title("Matrix Multiplication Timing: NumPy vs BLAS vs MPS")
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Time (seconds)")
plt.xticks(sizes_np, rotation=45)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))  # <- Y-axis every 0.5s
plt.ylim(bottom=0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/matmul_timing.pdf", format='pdf')
plt.show()

### 2. Log-Log Plot with readable x-ticks and rotated labels ###
N_ref = np.linspace(min(sizes_np), max(sizes_np), 100)
T_ref = 2e-12 * N_ref**3  # Trend Line for GEMM ~ 2N^3 FLOPs

plt.figure(figsize=(10, 6))
plt.loglog(sizes_np, blas_times_s, 'b-o', label='CPU (BLAS - Accelerate)')
plt.loglog(sizes_np, mps_times_s, 'g-o', label='GPU (MPS)')
plt.loglog(sizes_np, numpy_times_s, 'r-o', label='CPU (NumPy)')
plt.loglog(N_ref, T_ref, 'k--', label='Trend Line $2N^3$')
plt.title("Matrix Multiplication Timing (Log-Log) with $2N^3$ Trend Line")
plt.xlabel("Matrix Size (log N)")
plt.ylabel("Time (log seconds)")
plt.xticks(sizes_np, sizes_np, rotation=45)  # <- Custom tick locations and labels
plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())  # <- Disable sci notation
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig("figures/matmul_timing_loglog.pdf", format='pdf')
plt.show()

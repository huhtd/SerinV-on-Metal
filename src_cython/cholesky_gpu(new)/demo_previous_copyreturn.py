import numpy as np
import matplotlib.pyplot as plt
import time
import scipy

from Metal import MTLCreateSystemDefaultDevice
device = MTLCreateSystemDefaultDevice()
if device is None:
    raise RuntimeError("No Metal GPU device found. Exiting before cholesky_gpu call.")

# Import the GPU-accelerated function
from metal_cholesky import cholesky_gpu 

def generate_positive_definite_matrix_32int(N):
    """
    Generate a random NxN positive definite matrix in float32.
    """
    A = np.random.rand(N, N).astype(np.float32)
    M = np.dot(A, A.T)
    M[np.diag_indices_from(M)] += 1.0
    return M  # M is positive-definite (strict)

if __name__ == "__main__":
    mode = 0  # 0 = both, 1 = GPU only, 2 = CPU only

    N         = 5000       # matrix size
    num_mats  = 10          # how many matrices to factor

    mats = [generate_positive_definite_matrix_32int(N) for _ in range(num_mats)]

    if mode in (0, 1):
        # Warm-up GPU
        cholesky_gpu(mats[0][:128, :128].copy())
        #cholesky_gpu(np.empty((N,N), dtype=np.float32))
        #cholesky_gpu(np.eye(N, dtype=np.float32))

        # GPU timed loop
        t0 = time.perf_counter()
        for A in mats:
            Lg = cholesky_gpu(A)
        gpu_total = time.perf_counter() - t0
        gpu_avg   = gpu_total / num_mats

    if mode in (0, 2):
        # Warm-up CPU
        _ = np.linalg.cholesky(mats[0][:128, :128]).astype(np.float32)

        # CPU timed loop
        t1 = time.perf_counter()
        for A in mats:
            Ln = np.linalg.cholesky(A).astype(np.float32)
            #Ln = scipy.linalg.cholesky(A, lower=True, overwrite_a=True, check_finite=False)
        cpu_total = time.perf_counter() - t1
        cpu_avg   = cpu_total / num_mats

    print(f"Size Matrix = {N},  batches = {num_mats}")
    
    if mode in (0, 1):
        print(f"Avg GPU time : {gpu_avg*1e3:8.1f} ms")
    if mode in (0, 2):
        print(f"Avg CPU time : {cpu_avg*1e3:8.1f} ms")
    if mode == 0 and cpu_avg > 0:
        print(f"Speed-up GPU / CPU : {cpu_avg / gpu_avg:5.2f}×")

        # Spot check result
        diff = np.linalg.norm(Lg - Ln)
        print(f"‖L_gpu − L_np‖_F = {diff:.2e}")

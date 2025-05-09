import numpy as np
import matplotlib.pyplot as plt
import time

# Import the GPU-accelerated function
from metal_cholesky import cholesky_gpu 

def generate_positive_definite_matrix_32int(N):
    """
    Generate a random NxN positive definite matrix in float32.
    """
    A = np.random.rand(N, N).astype(np.float32)
    M = np.dot(A,A.T)

    # add a diagonal of 1.0 so λ_min ≥ 1, wide safety margin
    M[np.diag_indices_from(M)] += 1.0

    return M  # M is positive-definite (strict)

if __name__ == "__main__":
    N         = 5000        # matrix size
    num_mats  = 10          # how many matrices to factor

    # create all matrices up-front so generation time isn't measured
    mats = [generate_positive_definite_matrix_32int(N) for _ in range(num_mats)]

    # --------------- warm-up GPU once (JIT & allocations) -------------
    cholesky_gpu(mats[0][:128, :128].copy())

    # --------------- GPU timed loop -----------------------------------
    t0 = time.perf_counter()
    for A in mats:
        Lg = cholesky_gpu(A)
    gpu_total = time.perf_counter() - t0
    gpu_avg   = gpu_total / num_mats

    # --------------- CPU (NumPy) timed loop ---------------------------
    t1 = time.perf_counter()
    for A in mats:
        Ln = np.linalg.cholesky(A).astype(np.float32)
    cpu_total = time.perf_counter() - t1
    cpu_avg   = cpu_total / num_mats

    # --------------- summary ------------------------------------------
    print(f"Size Matrix = {N},  batches = {num_mats}")
    print(f"Avg GPU time : {gpu_avg*1e3:8.1f} ms")
    print(f"Avg CPU time : {cpu_avg*1e3:8.1f} ms")
    if cpu_avg > 0:
        print(f"Speed-up GPU / CPU : {cpu_avg / gpu_avg:5.2f}×")

    # quick correctness spot-check on last pair
    diff = np.linalg.norm(Lg - Ln)
    print(f"‖L_gpu − L_np‖_F = {diff:.2e}")

    # ----- visualisation ---------------------------------------------
    #fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    #axs[0].matshow(L_gpu)
    #axs[0].set_title("L (GPU / Metal)")
    #axs[1].matshow(L_np)
    #axs[1].set_title("L (NumPy)")
    #plt.show()

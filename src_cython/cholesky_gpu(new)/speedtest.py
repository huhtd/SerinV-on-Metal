import time
import numpy as np
from metal_cholesky import cholesky_gpu_inplace   # in-place FP32 routine
from Metal import MTLCreateSystemDefaultDevice

if MTLCreateSystemDefaultDevice() is None:
    raise RuntimeError("No Metal GPU device found.")

# ------------------------------------------------------------------ util
def generate_posdef_matrix(n: int, dtype=np.float32) -> np.ndarray:
    """
    Why not use np.random as before? because it is way too slow for large matrices
    Symmetric positive-definite tridiagonal Toeplitz matrix:
        2 on the diagonal, -1 on the first sub- and super-diagonal.
    Eigenvalues:  2-2·cos(k·π/(n+1))  > 0   → guaranteed PD.
    Construction is O(n) and uses < 3·n memory while building.
    """
    M = 2 * np.eye(n,   dtype=dtype)           # main diagonal
    M += -1 * np.eye(n, k=1, dtype=dtype)      # super-diagonal
    M += -1 * np.eye(n, k=-1, dtype=dtype)     # sub-diagonal
    return M


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    N = 40000
    print(f"AMX Cholesky, N = {N}")

    A = generate_posdef_matrix(N)   # master copy
    t0 = time.perf_counter()
    cholesky_gpu_inplace(A)         # overwrites A with its factor L
    elapsed = time.perf_counter() - t0



    print(f"Factorisation time : {elapsed*1e3:8.1f} ms")

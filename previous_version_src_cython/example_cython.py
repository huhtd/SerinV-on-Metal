import numpy as np
import matplotlib.pyplot as plt

# Import the Cython-accelerated function
from cholesky_cython import cholesky_cython

def generate_positive_definite_matrix_32int(N):
    """
    Generate a random NxN positive definite matrix in float32.
    """
    A = np.random.rand(N, N).astype(np.float32)
    return np.dot(A, A.T)  # A * A^T is positive-definite

if __name__ == "__main__":
    N = 50
    A = generate_positive_definite_matrix_32int(N)

    # use cython
    L_cy = cholesky_cython(A)

    # compare to numpy
    L_np = np.linalg.cholesky(A.astype(np.float64)).astype(np.float32)

    # get difference
    diff = np.linalg.norm(L_cy - L_np)
    print(f"Difference between Cython L and NumPy L: {diff}")


    fig, axs = plt.subplots(1, 2)
    axs[0].matshow(L_cy)
    axs[0].set_title("L (Cython)")
    axs[1].matshow(L_np)
    axs[1].set_title("L (NumPy)")

    plt.show()

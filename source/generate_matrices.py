import numpy as np

def generate_random_posdef_matrix(n: int, dtype=np.float32) -> np.ndarray:
    """
    Generates a full (dense) random positive-definite matrix of size n x n.
    """
    A = np.random.randn(n, n).astype(dtype)
    M = np.dot(A, A.T)  # Symmetric positive semi-definite
    # Add n * ε * I to push it to strictly positive-definite
    epsilon = 1e-3
    M += epsilon * np.eye(n, dtype=dtype)
    return M

sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 7000, 10000, 15000]
matrices = {}

print("Generating dense positive-definite matrices...")
for n in sizes:
    print(f"Generating {n}x{n} matrix...")
    matrices[str(n)] = generate_random_posdef_matrix(n)

np.savez_compressed("precomputed_matrices_big.npz", **matrices)
print("All matrices saved to precomputed_matrices_big.npz.")

import numpy as np

def generate_random_posdef_matrix(n: int, dtype=np.float32) -> np.ndarray:
    """Generates a normalized positive-definite matrix of size n x n."""
    A = np.random.randn(n, n).astype(dtype)
    M = np.dot(A, A.T)  # Symmetric positive semi-definite
    M /= np.max(np.abs(M))  # Normalize to max(abs()) = 1
    M += 1e-3 * np.eye(n, dtype=dtype)  # Ensure positive definiteness
    return M

sizes = [500, 1000, 1500, 2000, 2500, 3000,3500,4000,4500,5000,7000,10000,15000]
matrices = {}

print("Generating normalized positive-definite matrices...")
for n in sizes:
    print(f"Generating {n}x{n} matrix...")
    matrices[str(n)] = generate_random_posdef_matrix(n)

np.savez_compressed("precomputed_matrices_big.npz", **matrices)
print("All matrices saved to precomputed_matrices.npz.")

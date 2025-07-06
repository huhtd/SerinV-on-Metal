#include <Accelerate/Accelerate.h>
#include <cstdint>

extern "C" void cpu_blas_cholesky(float* A, unsigned int n) {
    int N = static_cast<int>(n);
    int lda = N;
    int info;

    // LAPACK expects column-major layout; our input is row-major
    // But Accelerate has row-major version using LAPACK_ROW_MAJOR
    // We’ll transpose it before and after

    // In-place factorization (LAPACK spotrf returns result in A)
    // Lower triangle version
    spotrf_("L", &N, A, &lda, &info);

    // Set upper triangle to 0 (match MPS and NumPy output)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            A[i * N + j] = 0.0f;
        }
    }

    if (info != 0) {
        printf("LAPACK spotrf failed with info = %d\n", info);
    }
}

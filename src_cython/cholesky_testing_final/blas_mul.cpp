#include <Accelerate/Accelerate.h>
#include <cstdint>

// Expose function as C to prevent name mangling
extern "C" void cpu_blas_multiply(const float* left, const float* right, float* out, unsigned int nu) {
    int n = static_cast<int>(nu);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                1.0f, left, n,
                      right, n,
                0.0f, out, n);
}

// mps_cholesky.m  – Objective-C ARC • AMX / Accelerate 64-bit
#import "mps_cholesky.h"
#import <Accelerate/Accelerate.h>
#import <stdlib.h>
#import <string.h>

void mps_cholesky(double *A, int N)
{
    const size_t bytes = (size_t)N * (size_t)N * sizeof(double);

    /* column-major scratch (Accelerate uses Fortran layout) */
    double *col = (double *)malloc(bytes);
    if (!col) return;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            col[i + (size_t)j * N] = A[(size_t)i * N + j];

    /* AMX-optimised dpotrf_ (lower) */
    char              uplo = 'L';
    __CLPK_integer    n    = (__CLPK_integer)N;
    __CLPK_integer    lda  = (__CLPK_integer)N;
    __CLPK_integer    info = 0;

    dpotrf_(&uplo, &n, col, &lda, &info);
    if (info) { free(col); return; }    /* not positive-definite */

    /* copy lower triangle back, zero upper */
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j)
            A[(size_t)i * N + j] = col[i + (size_t)j * N];
        for (int j = i + 1; j < N; ++j)
            A[(size_t)i * N + j] = 0.0;
    }
    free(col);
}

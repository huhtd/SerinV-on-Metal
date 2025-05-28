#ifndef MPS_CHOLESKY_H
#define MPS_CHOLESKY_H
/* In-place Cholesky factorisation (lower).  
   A is an N×N row-major float64 matrix that will be overwritten with L. */
void mps_cholesky(double *A, int N);
#endif

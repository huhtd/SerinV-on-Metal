// mps_cholesky.m  –  Objective-C, ARC • in-place variant
#import "mps_cholesky.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

void mps_cholesky(float *A, int N)           /* ← single pointer, no return */
{
    /* ---------- one-time objects ---------- */
    static id<MTLDevice>       dev  = nil;
    static id<MTLCommandQueue> q    = nil;
    static MPSMatrixDecompositionCholesky *chol = nil;
    static dispatch_once_t     once;

    dispatch_once(&once, ^{
        dev  = MTLCreateSystemDefaultDevice();
        q    = [dev newCommandQueue];
        chol = [[MPSMatrixDecompositionCholesky alloc] initWithDevice:dev
                                                               lower:true
                                                               order:16];    // build once for max N
    });

    const size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    /* ---------- wrap caller’s buffer ---------- */
    id<MTLBuffer> buf =
        [dev newBufferWithBytesNoCopy:(void *)A
                               length:bytes
                              options:MTLResourceStorageModeShared
                          deallocator:nil];

    /* ---------- command buffer ---------- */
    id<MTLCommandBuffer> cb = [q commandBuffer];

    /* ---------- encode Cholesky (in-place) ---------- */
    MPSMatrixDescriptor *desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                              columns:N
                                             rowBytes:N * sizeof(float)
                                             dataType:MPSDataTypeFloat32];

    MPSMatrix *M = [[MPSMatrix alloc] initWithBuffer:buf descriptor:desc];

    [chol encodeToCommandBuffer:cb sourceMatrix:M resultMatrix:M status:nil];

    /* ---------- zero upper triangle after GPU ---------- */
    [cb addCompletedHandler:^(__unused id<MTLCommandBuffer>){
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j)
                A[i * N + j] = 0.0f;
    }];

    [cb commit];
    [cb waitUntilCompleted];
}

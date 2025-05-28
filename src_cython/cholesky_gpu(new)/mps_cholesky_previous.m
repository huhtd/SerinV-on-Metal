// mps_cholesky.m  ── Objective-C, ARC enabled
#import "mps_cholesky.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

void mps_cholesky(const float *A_in, float *L_out, int N)
{
    /* ---------- one-time static objects ---------- */
    static id<MTLDevice>  dev  = nil;
    static id<MTLCommandQueue> q = nil;
    static MPSMatrixDecompositionCholesky *chol = nil;
    static dispatch_once_t once;

    dispatch_once(&once, ^{
        dev  = MTLCreateSystemDefaultDevice();
        /* id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); GIVES SEGMENTATION FAULT*/
        q    = [dev newCommandQueue];
        chol = [[MPSMatrixDecompositionCholesky alloc] initWithDevice:dev
                                                               lower:true
                                                               order:N];
    });

    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    /* ---------- buffers ---------- */
    id<MTLBuffer> sA = [dev newBufferWithBytes:A_in          /* staging (shared) */
                                         length:bytes
                                        options:MTLResourceStorageModeShared];

    id<MTLBuffer> dA = [dev newBufferWithLength:bytes        /* private for speed */
                                        options:MTLResourceStorageModePrivate];

    id<MTLBuffer> dL = [dev newBufferWithLength:bytes        /* shared for read-back */
                                        options:MTLResourceStorageModeShared];
    memset(dL.contents, 0, bytes);                           /* zero entire result */

    /* ---------- command buffer ---------- */
    id<MTLCommandBuffer> cb = [q commandBuffer];

    /* blit host→device : sA → dA */
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromBuffer:sA
            sourceOffset:0
                toBuffer:dA
       destinationOffset:0
                    size:bytes];
    [bl endEncoding];

    /* run Cholesky */
    MPSMatrixDescriptor *desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                              columns:N
                                             rowBytes:N * sizeof(float)
                                             dataType:MPSDataTypeFloat32];

    MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:dA descriptor:desc];
    MPSMatrix *L = [[MPSMatrix alloc] initWithBuffer:dL descriptor:desc];

    [chol encodeToCommandBuffer:cb sourceMatrix:A resultMatrix:L status:nil];

    /* copy back after GPU finishes */
    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        /* blank upper triangle the kernel left untouched */
        float *p = (float *)dL.contents;
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j)
                p[i * N + j] = 0.0f;
        memcpy(L_out, p, bytes);
    }];

    [cb commit];
    [cb waitUntilCompleted];
}

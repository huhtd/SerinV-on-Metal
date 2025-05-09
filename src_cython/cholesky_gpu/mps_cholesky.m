#import "mps_cholesky.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

void mps_cholesky(const float *A_in, float *L_out, int N)
{
    /* ---------- one-time static objects ---------- */
    static id<MTLDevice>                 dev   = nil;
    static id<MTLCommandQueue>           q     = nil;
    static MPSMatrixDecompositionCholesky *chol = nil;
    static dispatch_once_t once;

    dispatch_once(&once, ^{
        dev  = MTLCreateSystemDefaultDevice();
        q    = [dev newCommandQueue];
        chol = [[MPSMatrixDecompositionCholesky alloc] initWithDevice:dev
                                                               lower:true
                                                               order:N];
    });

    /* ---------- per-call buffers (private mem) ---------- */
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    id<MTLBuffer> dA = [dev newBufferWithLength:bytes
                                        options:MTLResourceStorageModePrivate];
    id<MTLBuffer> dL = [dev newBufferWithLength:bytes
                                        options:MTLResourceStorageModePrivate];

    /* upload A -> dA */
    id<MTLCommandBuffer> cb = [q commandBuffer];
    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        memcpy(L_out, dL.contents, bytes);      /* copy back after GPU done */
    }];
    [dA didModifyRange:NSMakeRange(0, bytes)];  /* hint for unified mem */

    /* blit host → device */
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromBytes:A_in
              sourceBytesPerRow:N*sizeof(float)
             sourceBytesPerImage:0
                       sourceSize:MTLSizeMake(N, N, 1)
                        toTexture:nil
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:MTLOriginMake(0,0,0)];
    [bl endEncoding];

    /* run Cholesky */
    MPSMatrixDescriptor *desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                              columns:N
                                             rowBytes:N*sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:dA descriptor:desc];
    MPSMatrix *L = [[MPSMatrix alloc] initWithBuffer:dL descriptor:desc];

    [chol encodeToCommandBuffer:cb sourceMatrix:A resultMatrix:L status:nil];
    [cb commit];
    [cb waitUntilCompleted];
}

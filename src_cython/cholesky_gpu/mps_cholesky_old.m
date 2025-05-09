// Compile as Objective-C (.m), ARC enabled
#import "mps_cholesky.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#this file also works, however slightly slower.

void mps_cholesky(const float *A_in, float *L_out, int N)
/* Exposed with C linkage → easy for Cython */
{
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];

    NSUInteger bytes = (NSUInteger)N * (NSUInteger)N * sizeof(float);

    id<MTLBuffer> dA = [dev newBufferWithBytes:A_in length:bytes
                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> dL = [dev newBufferWithLength:bytes
                                        options:MTLResourceStorageModeShared];
    memset(dL.contents, 0, bytes);

    MPSMatrixDescriptor *desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                              columns:N
                                             rowBytes:N*sizeof(float)
                                             dataType:MPSDataTypeFloat32];

    MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:dA descriptor:desc];
    MPSMatrix *L = [[MPSMatrix alloc] initWithBuffer:dL descriptor:desc];

    MPSMatrixDecompositionCholesky *chol =
        [[MPSMatrixDecompositionCholesky alloc] initWithDevice:dev
                                                        lower:true      // true ⇒ compute lower-triangular L
                                                        order:N];       // matrix size


    id<MTLCommandBuffer> cb = [q commandBuffer];
    [chol encodeToCommandBuffer:cb sourceMatrix:A resultMatrix:L status:nil];
    [cb commit];
    [cb waitUntilCompleted];

    float *ptr = (float *)dL.contents;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            ptr[i * N + j] = 0.0f;

    memcpy(L_out, dL.contents, bytes);           // copy GPU result back
}

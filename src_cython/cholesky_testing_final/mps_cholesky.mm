#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cassert>
#include <cstdio>

extern "C" void gpu_mps_cholesky(float* A, unsigned int n) {
    static id<MTLDevice> device = nil;
    static id<MTLCommandQueue> queue = nil;

    if (!device) {
        device = MTLCreateSystemDefaultDevice();
        queue = [device newCommandQueue];
    }

    size_t matrix_size_bytes = n * n * sizeof(float);
    id<MTLBuffer> bufA = [device newBufferWithBytesNoCopy:A
                                                   length:matrix_size_bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];

    MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                                       columns:n
                                                                      rowBytes:n * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:desc];

    // Create cholesky kernel dynamically per matrix size (no hardcoded 16)
    MPSMatrixDecompositionCholesky *cholesky =
        [[MPSMatrixDecompositionCholesky alloc] initWithDevice:device
                                                         lower:YES
                                                         order:n];

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

    [cholesky encodeToCommandBuffer:commandBuffer
                       sourceMatrix:matA
                      resultMatrix:matA
                             status:nil];


    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

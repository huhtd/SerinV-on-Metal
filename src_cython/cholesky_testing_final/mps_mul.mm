#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cassert>

extern "C" void gpu_mps_multiply(const float* A, const float* B, float* C, unsigned int n) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    assert(device != nullptr);

    size_t memory_length = n * n * sizeof(float);

    id<MTLBuffer> bufA = [device newBufferWithBytesNoCopy:(void*)A length:memory_length options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bufB = [device newBufferWithBytesNoCopy:(void*)B length:memory_length options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bufC = [device newBufferWithBytesNoCopy:(void*)C length:memory_length options:MTLResourceStorageModeShared deallocator:nil];

    MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:n rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:desc];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:desc];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:desc];

    MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:device resultRows:n resultColumns:n interiorColumns:n];

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];

    [matmul encodeToCommandBuffer:buffer leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [buffer commit];
    [buffer waitUntilCompleted];
}

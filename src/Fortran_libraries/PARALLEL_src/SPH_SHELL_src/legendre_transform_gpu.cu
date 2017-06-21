#include "legendre_transform_gpu.h"
#include <cuda_runtime.h>
#include <iostream>

__host__ void LegendreTransform::initialize_gpu()
{
  int device_count, device;
  cudaGetDeviceCount(&device_count);
  cudaGetDevice(&device);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  size_t devMemory = prop.totalGlobalMem;
  std::cout << "Device Memory = " << devMemory << std::endl;
  cudaFree(0);
}

__host__ void LegendreTransform::finalize_gpu()
{
  cudaFree(0);
}

extern "C" {
  void initialize_gpu_() {
    LegendreTransform x;
    x.initialize_gpu();
  }
}

#include "cuda.h"                                                                  
#include "cuda_profiler_api.h"
#include <iostream>

/* The conclusion of this benchmarck is that:

 */ 

using namespace std;

/*__device__ __forceinline__ void prefetchASMREG( double *data, int offset ){
  data += offset;
  asm("prefetch.global.L2 [%0];"::"l"(data) );
}*/

__global__
void vecMul(double *a, double *b) {
  int idx = blockIdx.x*blockDim.x;
  double reg = 1;
 for(int i=0; i<100000; i++) {
  reg *= a[idx + threadIdx.x];
  reg *= a[idx + (threadIdx.x+4)%32];
  reg *= a[idx + (threadIdx.x+8)%32];
  reg *= a[idx + (threadIdx.x+12)%32];
 }
 
  b[threadIdx.x] = reg; 
}

int main() {
  cudaProfilerStart();
  double *a, *b; 

  cudaMalloc((void**)&a, 229376*sizeof(double));
  cudaMalloc((void**)&b, 32*sizeof(double));
  cudaMemset(a, 1.0, 229376*sizeof(double));

  vecMul<<<7168,32>>>(a, b); 

  double *h_b = new double[32];
  cudaMemcpy(h_b, b, 32*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(a);
  cudaFree(b);
  cudaProfilerStop();
}

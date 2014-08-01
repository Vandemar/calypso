#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_cublas.h"
//#include "cuda_profiler_api.h"

//Using Cuda Registers
__constant__ int indx_rtm;
__constant__ int indx_rlm;


void cublasStatusCheck(cublasStatus_t stat) {
  if ( strcmp(_cublasGetErrorEnum(stat),"cublasSuccess") != 0 ) {
    printf("%s\n", _cublasGetErrorEnum(stat));
  }
  return;
}
void cudaErrorCheck(cudaError_t error) {
  if ( strcmp(_cudaGetErrorEnum(error), "cudaSuccess") != 0 ) {
    printf ("%s\n", _cudaGetErrorEnum(error));
  }
  return;
}

__global__ void scaleBy_GaussSphMatrix(cublasStatus_t *ptrStat, double* Pws_l_d, double* g_sph_rlm_d) {
  __shared__ double scale;

  int me = blockIdx.x;
  int index = indx_rtm*(me);

  scale = g_sph_rlm_d[me];

  cublasStatus_t stat;
  cublasHandle_t coHandle;
  stat = cublasCreate(&coHandle);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    ptrStat = &stat;
    return;
  }
  
  stat = cublasDscal(coHandle, indx_rtm, &scale, Pws_l_d + index, 1);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    ptrStat = &stat;
    return;
  }

  cublasDestroy(coHandle);
  ptrStat = &stat;
  return;
}

__global__ void scaleBy_Weights(cublasStatus_t *ptrStat, double* Pws_l_d, double* weight_rtm_d) {
  __shared__ double scale;

  int me = blockIdx.x;
  scale = weight_rtm_d[me];

  cublasStatus_t stat;
  cublasHandle_t coHandle;
  stat = cublasCreate(&coHandle);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    ptrStat = &stat;
    return;
  }

  stat = cublasDscal(coHandle, indx_rlm, &scale, Pws_l_d + me, indx_rtm);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    ptrStat = &stat;
    return;
  }

  cublasDestroy(coHandle);
  ptrStat = &stat;
  return;
}

 
extern "C" void spectral_to_grid_(int* nidx_rtm, int* nidx_rlm, double* P_rtm, double* g_sph_rlm, double* weight_rtm, double* Pws) {
  //Declaring device matrices 
  double *P_rtm_d, *g_sph_rlm_d, *weight_rtm_d;

  //cudaProfilerStart();
  //Defining cuda and cublas debugging variables
  cudaError_t error;
  cublasStatus_t stat;

  //Testing cublas
  cublasHandle_t handle;
  stat = cublasCreate(&handle);
  cublasStatusCheck(stat);

  // Initializing 2 cuda constant variables:
  // 	1). number of modes 
  //    2). number of meridians

  error = cudaMemcpyToSymbol(indx_rtm, nidx_rtm, sizeof(int));
  cudaErrorCheck(error);
  error = cudaMemcpyToSymbol(indx_rlm, nidx_rlm, sizeof(int));
  cudaErrorCheck(error);

  //allocating space for device matrices:
  //P_rtm_d
  error = cudaMalloc((void**)&P_rtm_d, sizeof(*P_rtm)*(*nidx_rtm)*(*nidx_rlm));
  cudaErrorCheck(error);
  //Moving data from host to device.
  stat = cublasSetMatrixAsync(*nidx_rtm, *nidx_rlm, sizeof(*P_rtm), P_rtm, *nidx_rtm, P_rtm_d, *nidx_rtm, 0);
  cublasStatusCheck(stat);

  //allocating space 
  //g_sph_rlm_d
  error = cudaMalloc((void**)&g_sph_rlm_d, sizeof(*g_sph_rlm)*(*nidx_rlm));
  cudaErrorCheck(error);
  //Moving data from host to device
  stat = cublasSetVectorAsync(*nidx_rlm, sizeof(double), g_sph_rlm, 1, g_sph_rlm_d, 1, 0);
  cublasStatusCheck(stat);

  //call to kernel
  scaleBy_GaussSphMatrix<<<*nidx_rlm,1>>> (&stat, P_rtm_d, g_sph_rlm_d);
  //checking if kernel exited safely.
  cublasStatusCheck(stat);
 
  //allocating space
  //weight_rtm_d
  error = cudaMalloc((void**)&weight_rtm_d, sizeof(*weight_rtm)*(*nidx_rtm));
  cudaErrorCheck(error);
  //Moving data from host to device
  stat = cublasSetVectorAsync(*nidx_rtm, sizeof(double), weight_rtm, 1, weight_rtm_d, 1, 0);
  cublasStatusCheck(stat);

  //Synching work from previous call to kernel
  cudaThreadSynchronize();
   
  //call to kernel
  scaleBy_Weights<<<*nidx_rtm, 1>>> (&stat, P_rtm_d, weight_rtm_d);
  cublasStatusCheck(stat);

  //synching threads once more.
  cudaThreadSynchronize();

  //moving device data into a host matrix
  stat = cublasGetMatrix(*nidx_rtm, *nidx_rlm, sizeof(double), P_rtm_d, *nidx_rtm, Pws, *nidx_rtm);
  cublasStatusCheck(stat);

  cublasDestroy(handle);

  cudaFree(P_rtm_d);
  cudaFree(g_sph_rlm_d);  
  cudaFree(weight_rtm_d);
//  cudaProfilerStop();
//  cudaDeviceReset();
  return; 
} 


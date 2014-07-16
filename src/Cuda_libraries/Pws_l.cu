#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

extern "C" void spectral_to_grid_(int* nidx_rlm, int* nidx_rtm, double* P_rtm, double* g_sph_rlm, double* weight_rtm, double* Pws) {
  
  int i;
  //device matrices 
  double *P_rtm_d, *g_sph_rlm_d, *weight_rtm_d;
  cublasHandle_t handle;
        
  //allocating space for device matrices
  cudaMalloc((void**)&P_rtm_d, sizeof(*P_rtm)*(*nidx_rtm)*(*nidx_rlm));
 
  cublasSetMatrix(*nidx_rtm, *nidx_rlm, sizeof(*P_rtm), P_rtm, *nidx_rtm, P_rtm_d, *nidx_rtm);
  
  cublasCreate(&handle);
  double *tmp;
  for( i=1; i<(*nidx_rlm)+1; i++) {
    tmp = g_sph_rlm+(i-1);
    cublasDscal(handle, *nidx_rtm, tmp, P_rtm_d + (*nidx_rtm)*(i-1), 1); 
  } 

  for( i=1; i<(*nidx_rtm)+1; i++) {
    tmp = weight_rtm+(i-1);
    cublasDscal(handle, *nidx_rlm, tmp, P_rtm_d + (i-1), *nidx_rtm);    
  }

  //moving device data into a host matrix
  cublasGetMatrix(*nidx_rtm, *nidx_rlm, sizeof(*Pws), P_rtm_d, *nidx_rtm, Pws, *nidx_rtm);   
  
  cublasDestroy(handle);

  cudaFree(P_rtm_d);
  return; 
} 

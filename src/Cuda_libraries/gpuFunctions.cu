#include <cuda_runtime.h>
#include <math.h>
#include <unistd.h>

#include "legendre_poly.h"

#include "math_functions.h"
#include "math_constants.h"

cudaDeviceProp prop;
Parameters_s deviceInput;
Debug h_debug, d_debug;
Geometry_c constants;
References hostData;
Logger cudaPerformance("Metrics.log", 7);

Timer movData2GPU;
Timer movData2Host;

int countFT=0, countBT=0;
int minGridSize=0, blockSize=0;
size_t devMemory = 0;
cudaStream_t *streams;
int nStreams=0;

// **** lstack_rlm resides in global memory as well as constant memory
// ** Pick one or the other
__constant__ int lstack_rlm_cmem[1000];

//CUDA Unbound - part of device reduce example
bool g_verbose = false; // Whether to display input/output to console
cub::CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory

void initialize_gpu_() {

//Required because, Template parameters need to be evaluated by compile time
//#if __cplusplus > 199711L
//   #error c++ 11 standard or greater REQUIRED!
// #endif

  int device_count, device;
  cudaDeviceReset();
  // Gets number of GPU devices
  cudaGetDeviceCount(&device_count);
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  devMemory = prop.totalGlobalMem;
  cudaErrorCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  cudaErrorCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
  cudaFree(0);
  #if defined(CUDA_TIMINGS)
    cudaProfilerStart();
  #endif
  //registerAllTimers();
  movData2GPU.setWhatAmI("Transfer data from Host to GPU");
  movData2Host.setWhatAmI("Transfer data from GPU to Host");

  cudaPerformance.registerTimer(&movData2GPU);
  cudaPerformance.registerTimer(&movData2Host);
}

void registerAllTimers() {
  //If more timers are registered than the amount specified in the constructor of logger, program will 
  // segfault.
  // TO BE DEPRECATED
  // Timer transBwdVec("Bwd Vector Transform");
  // Timer transBwdScalar("Bwd Scalar Transform");
  // Timer transF_s("Fwd scalar transform with cached schmidt");
  // Timer transF_reduce("Fwd Vector Reduction Algorithm");
  // cudaPerformance.registerTimer(&transBwdVec);
  // cudaPerformance.registerTimer(&transBwdScalar);
  // cudaPerformance.registerTimer(&transF_reduce);
  // cudaPerformance.registerTimer(&transF_s);
} 

void set_constants_(int *nnod_rtp, int *nnod_rtm, int *nnod_rlm, int nidx_rtm[], int nidx_rlm[], int istep_rtm[], int istep_rlm[], int *trunc_lvl, int *np_smp) {

  cudaPerformance.recordProblemDescription(*trunc_lvl, nidx_rtm[0], nidx_rtm[1]);

  //For best occupancy
  /*cudaErrorCheck(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize,
																&blockSize,
																transF_vec_reduction< 10, 
																  cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
																		double>, 
																computeSharedMemory));  
  */

  for(int i=0; i<3; i++) { 
    constants.nidx_rtm[i] = nidx_rtm[i];
    constants.istep_rtm[i] = istep_rtm[i];
  }

  for(int i=0; i<2; i++) {
    constants.nidx_rlm[i] = nidx_rlm[i];
    constants.istep_rlm[i] = istep_rlm[i];
  }

  constants.nnod_rtp = *nnod_rtp;
  constants.nnod_rtm = *nnod_rtm;
  constants.nnod_rlm = *nnod_rlm;
  constants.t_lvl = *trunc_lvl; 

  constants.np_smp = *np_smp;



//  #if defined(CUDA_TIMINGS)
/*    t_1 = MPI_Wtime();
    char name[15];
    gethostname(name, 15);
    string str(name);
    std::cout<<"Host: " << str << "\t Memory Allocation Time: " << t_1-t_0 << "\t Device Initialization Time: " << t_3-t_2 << std::endl;*/
//  #endif

}

void setptrs_(int *idx_gl_1d_rlm_j) {
  //Necessary to filter harmonic modes across MPI nodes.
  h_debug.idx_gl_1d_rlm_j = idx_gl_1d_rlm_j;
}

/*void setptrs_(int *idx_gl_1d_rlm_j, double *P_smdt, double *dP_smdt) {
  h_debug.idx_gl_1d_rlm_j = idx_gl_1d_rlm_j;
  //h_debug.P_smdt = P_smdt;
  //h_debug.dP_smdt = dP_smdt;
}*/


void initialize_leg_trans_gpu_() {
  size_t memAllocation = 0;
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.g_colat_rtm), constants.nidx_rtm[1]*sizeof(double))); 
  memAllocation -= constants.nidx_rtm[1]*sizeof(double);
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.a_r_1d_rlm_r), constants.nidx_rtm[0]*sizeof(double))); 
  memAllocation -= constants.nidx_rtm[0]*sizeof(double);
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.asin_theta_1d_rtm), constants.nidx_rtm[1]*sizeof(double))); 
  memAllocation -= constants.nidx_rtm[1]*sizeof(double);
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.lstack_rlm), (constants.nidx_rtm[2]+1)*sizeof(int))); 
  
  memAllocation -= (constants.nidx_rtm[2]+1)*sizeof(int);
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.g_sph_rlm), constants.nidx_rlm[1]*sizeof(double))); 
  memAllocation -= constants.nidx_rlm[1]*sizeof(double);
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.g_sph_rlm_7), constants.nidx_rlm[1]*sizeof(double))); 
  memAllocation -= constants.nidx_rlm[1]*sizeof(double);
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.idx_gl_1d_rlm_j), constants.nidx_rlm[1]*3*sizeof(int))); 
  memAllocation -= constants.nidx_rlm[1]*3*sizeof(int);
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.radius_1d_rlm_r), constants.nidx_rtm[0]*sizeof(double))); 
  memAllocation -= constants.nidx_rtm[0]*sizeof(double);
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.weight_rtm), constants.nidx_rtm[1]*sizeof(double))); 
  memAllocation -= constants.nidx_rtm[1]*sizeof(double);
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.mdx_p_rlm_rtm), constants.nidx_rlm[1]*sizeof(int))); 
  memAllocation -= constants.nidx_rlm[1]*sizeof(int);
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.mdx_n_rlm_rtm), constants.nidx_rlm[1]*sizeof(int))); 
  memAllocation -= constants.nidx_rlm[1]*sizeof(int);
//#ifndef CUDA_OTF
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.p_jl), sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1]));
  memAllocation -= constants.nidx_rtm[1]*constants.nidx_rlm[1] * sizeof(double);
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.dP_jl), sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1]));
  memAllocation -= constants.nidx_rtm[1]*constants.nidx_rlm[1] * sizeof(double);
//#endif
//OTF has yet to be implemented for fwd transform
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.p_rtm), sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1]));
  memAllocation -= constants.nidx_rtm[1]*constants.nidx_rlm[1] * sizeof(double);
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.dP_rtm), sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1]));
  memAllocation -= constants.nidx_rtm[1]*constants.nidx_rlm[1] * sizeof(double);

// Question, is loading from DRAM faster than actual calculation? 
//since m=0,l=0 is the trivial case, this is excluded. All others i.e, m=1 upto t_lvl (inclusive) is allocated 
//  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.leg_poly_m_eq_l), sizeof(double)*(constants.t_lvl)));
//  memAllocation += sizeof(double)*(constants.t_lvl);

// A variable amount of memory
  // dim3 grid(1,1,1);
  // dim3 block(64,1,1);
  // set_leg_poly_m_ep_l<<<grid,block,0>>>(deviceInput.leg_poly_m_eq_l);
  
  #if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
    h_debug.P_smdt = (double*) malloc (sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1]);
    h_debug.dP_smdt = (double*) malloc (sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1]);
    cudaErrorCheck(cudaMalloc((void**)&(d_debug.P_smdt), sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1]));
    memAllocation -= sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1];
    cudaErrorCheck(cudaMemset(d_debug.P_smdt, -1, sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1]));
    cudaErrorCheck(cudaMalloc((void**)&(d_debug.dP_smdt), sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1]));
    memAllocation -= sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1];
    cudaErrorCheck(cudaMemset(d_debug.dP_smdt, -1, sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1]));
  #endif

  unsigned int numberOfDoubles = memAllocation/(sizeof(double));
  unsigned int numberOfReductionSpaces = min(numberOfDoubles/(constants.nidx_rtm[1]*3), constants.nidx_rtm[1]);
  //streams = (cudaStream_t*) malloc (sizeof(cudaStream_t) * numberOfReductionSpaces);
  streams = new cudaStream_t[numberOfReductionSpaces];
  for(int i=0; i<numberOfReductionSpaces; i++) {
    cudaErrorCheck(cudaStreamCreate(&streams[i]));
    nStreams++;
  }
  cudaErrorCheck(cudaMalloc((void**)&(deviceInput.reductionSpace), sizeof(double)*numberOfReductionSpaces*3*constants.nidx_rtm[1]));
  memAllocation -= sizeof(double)*numberOfReductionSpaces*3*constants.nidx_rtm[1];
  if(memAllocation <= 0) {
    exit(-1);
  }
}
 
void alloc_space_on_gpu_(int *ncmp, int *nvector, int *nscalar) {
  int ncomp = constants.ncomp = *ncmp;
  constants.nvector = *nvector;
  constants.nscalar = *nscalar;

  #if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
    if(!h_debug.vr_rtm)
      h_debug.vr_rtm = (double*) malloc (sizeof(double)*constants.nnod_rtm*constants.ncomp);
    if(!h_debug.sp_rlm)
      h_debug.sp_rlm = (double*) malloc (sizeof(double)*constants.nnod_rlm*constants.ncomp);
  #endif

  // Current: 0 = vr_rtm, 1 = sp_rlm, 2 = g_sph_rlm 
  if(!deviceInput.vr_rtm) {
    cudaErrorCheck(cudaMalloc((void**)&(deviceInput.vr_rtm), constants.nnod_rtm*ncomp*sizeof(double))); 
    cudaErrorCheck(cudaMemset(deviceInput.vr_rtm, 0, constants.nnod_rtm*ncomp*sizeof(double)));
  }
  if(!deviceInput.sp_rlm) {
    cudaErrorCheck(cudaMalloc((void**)&(deviceInput.sp_rlm), constants.nnod_rlm*ncomp*sizeof(double))); 
    cudaErrorCheck(cudaMemset(deviceInput.sp_rlm, 0, constants.nnod_rlm*ncomp*sizeof(double)));
  }
}

void memcpy_h2d_(int *lstack_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *g_sph_rlm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, int *idx_gl_1d_rlm_j, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm) {
   
    hostData.mdx_p_rlm_rtm = mdx_p_rlm_rtm;
    hostData.mdx_n_rlm_rtm = mdx_n_rlm_rtm;
    hostData.idx_gl_1d_rlm_j = idx_gl_1d_rlm_j;
    hostData.radius_1d_rlm_r = radius_1d_rlm_r;
    hostData.g_sph_rlm_7= g_sph_rlm_7;

    h_debug.lstack_rlm = lstack_rlm;
 #ifdef CUDA_DEBUG 
    h_debug.g_colat_rtm = g_colat_rtm;
    h_debug.g_sph_rlm = g_sph_rlm;
 #endif

  cudaErrorCheck(cudaMemcpy(deviceInput.a_r_1d_rlm_r, a_r_1d_rlm_r , constants.nidx_rtm[0]*sizeof(double), cudaMemcpyHostToDevice)); 
  cudaErrorCheck(cudaMemcpy(deviceInput.asin_theta_1d_rtm, asin_theta_1d_rtm, constants.nidx_rtm[1]*sizeof(double), cudaMemcpyHostToDevice)); 
  cudaErrorCheck(cudaMemcpy(deviceInput.g_colat_rtm, g_colat_rtm, constants.nidx_rtm[1]*sizeof(double), cudaMemcpyHostToDevice)); 
  cudaErrorCheck(cudaMemcpy(deviceInput.lstack_rlm, lstack_rlm, (constants.nidx_rtm[2]+1)*sizeof(int), cudaMemcpyHostToDevice)); 
 cudaErrorCheck(cudaMemcpyToSymbol(lstack_rlm_cmem, lstack_rlm, sizeof(int) * (constants.nidx_rtm[2]+1), 0, cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpy(deviceInput.g_sph_rlm, g_sph_rlm, constants.nidx_rlm[1]*sizeof(double), cudaMemcpyHostToDevice)); 
  cudaErrorCheck(cudaMemcpy(deviceInput.g_sph_rlm_7, g_sph_rlm_7, constants.nidx_rlm[1]*sizeof(double), cudaMemcpyHostToDevice)); 
  cudaErrorCheck(cudaMemcpy(deviceInput.idx_gl_1d_rlm_j, idx_gl_1d_rlm_j, constants.nidx_rlm[1]*3*sizeof(int), cudaMemcpyHostToDevice)); 
  cudaErrorCheck(cudaMemcpy(deviceInput.radius_1d_rlm_r, radius_1d_rlm_r, constants.nidx_rtm[0]*sizeof(double), cudaMemcpyHostToDevice)); 
  cudaErrorCheck(cudaMemcpy(deviceInput.weight_rtm, weight_rtm, constants.nidx_rtm[1]*sizeof(double), cudaMemcpyHostToDevice)); 
  cudaErrorCheck(cudaMemcpy(deviceInput.mdx_p_rlm_rtm, mdx_p_rlm_rtm, constants.nidx_rlm[1]*sizeof(int), cudaMemcpyHostToDevice)); 
  cudaErrorCheck(cudaMemcpy(deviceInput.mdx_n_rlm_rtm, mdx_n_rlm_rtm, constants.nidx_rlm[1]*sizeof(int), cudaMemcpyHostToDevice)); 
}

void cpy_schmidt_2_gpu_(double *P_jl, double *dP_jl, double *P_rtm, double *dP_rtm) {
  //#ifndef CUDA_OTF
    cudaErrorCheck(cudaMemcpy(deviceInput.p_jl, P_jl, sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1], cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(deviceInput.dP_jl, dP_jl, sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1], cudaMemcpyHostToDevice));
  //#endif
//FWD trans OTF has yet to be implemented
    cudaErrorCheck(cudaMemcpy(deviceInput.p_rtm, P_rtm, sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1], cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(deviceInput.dP_rtm, dP_rtm, sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1], cudaMemcpyHostToDevice));
}
 
void cpy_field_dev2host_4_debug_() {
  #if defined(CUDA_OTF)
    cudaErrorCheck(cudaMemcpy(h_debug.P_smdt, d_debug.P_smdt, sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1], cudaMemcpyDeviceToHost)); 
    cudaErrorCheck(cudaMemcpy(h_debug.dP_smdt, d_debug.dP_smdt, sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1], cudaMemcpyDeviceToHost)); 
  #endif
  cudaErrorCheck(cudaMemcpy(h_debug.vr_rtm, deviceInput.vr_rtm, constants.nnod_rtm*constants.ncomp*sizeof(double), cudaMemcpyDeviceToHost)); 
//  cudaErrorCheck(cudaMemcpy(d_data->g_sph_rlm, h_data->g_sph_rlm, sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1], cudaMemcpyDeviceToHost)); 
}

void cpy_spec_dev2host_4_debug_() {
  #if defined(CUDA_OTF)
    cudaErrorCheck(cudaMemcpy(h_debug.P_smdt, d_debug.P_smdt, sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1], cudaMemcpyDeviceToHost)); 
    cudaErrorCheck(cudaMemcpy(h_debug.dP_smdt, d_debug.dP_smdt, sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1], cudaMemcpyDeviceToHost)); 
  #endif
  cudaErrorCheck(cudaMemcpy(h_debug.sp_rlm, deviceInput.sp_rlm, constants.nnod_rlm*constants.ncomp*sizeof(double), cudaMemcpyDeviceToHost)); 
//  cudaErrorCheck(cudaMemcpy(d_data->g_sph_rlm, h_data->g_sph_rlm, sizeof(double)*constants.nidx_rtm[1]*constants.nidx_rlm[1], cudaMemcpyDeviceToHost)); 
}

void set_spectrum_data_(double *sp_rlm, int *ncomp) {
  // Current: 0 = vr_rtm, 1 = sp_rlm, 2 = g_sph_rlm 
  movData2GPU.startTimer();
  cudaErrorCheck(cudaMemcpy(deviceInput.sp_rlm, sp_rlm, constants.nnod_rlm*(*ncomp)*sizeof(double), cudaMemcpyHostToDevice)); 
  movData2GPU.endTimer();
}

void set_physical_data_(double *vr_rtm, int *ncomp) {
  // Current: 0 = vr_rtm, 1 = sp_rlm, 2 = g_sph_rlm 
  movData2GPU.startTimer();
  cudaErrorCheck(cudaMemcpy(deviceInput.vr_rtm, vr_rtm, constants.nnod_rtm*(*ncomp)*sizeof(double), cudaMemcpyHostToDevice)); 
  movData2GPU.endTimer();
}

void retrieve_spectrum_data_(double *sp_rlm, int *ncomp) {
  // Current: 0 = vr_rtm, 1 = sp_rlm, 2 = g_sph_rlm 
  movData2Host.startTimer();
  cudaErrorCheck(cudaMemcpy(sp_rlm, deviceInput.sp_rlm, constants.nnod_rlm*(*ncomp)*sizeof(double), cudaMemcpyDeviceToHost)); 
  movData2Host.endTimer();
}

void retrieve_physical_data_(double *vr_rtm, int *ncomp) {
  // Current: 0 = vr_rtm, 1 = sp_rlm, 2 = g_sph_rlm 
  movData2Host.startTimer();
  cudaErrorCheck(cudaMemcpy(vr_rtm, deviceInput.vr_rtm, constants.nnod_rtm*(*ncomp)*sizeof(double), cudaMemcpyDeviceToHost)); 
  movData2Host.endTimer();
}

//How should these functions be timed?
void clear_spectrum_data_(int *ncomp) {
  cudaErrorCheck(cudaMemset(deviceInput.sp_rlm, 0, constants.nnod_rlm*(*ncomp)*sizeof(double)));
}

void clear_field_data_(int *ncomp) {
  cudaErrorCheck(cudaMemset(deviceInput.vr_rtm, 0, constants.nnod_rtm*(*ncomp)*sizeof(double)));
}

void deAllocMemOnGPU() {
  // Current: 0 = vr_rtm, 1 = sp_rlm, 2 = g_sph_rlm 
    cudaErrorCheck(cudaFree(deviceInput.vr_rtm));
    cudaErrorCheck(cudaFree(deviceInput.sp_rlm));
    cudaErrorCheck(cudaFree(deviceInput.g_colat_rtm));
    cudaErrorCheck(cudaFree(deviceInput.g_sph_rlm));
    cudaErrorCheck(cudaFree(deviceInput.g_sph_rlm_7));
    cudaErrorCheck(cudaFree(deviceInput.a_r_1d_rlm_r));
    cudaErrorCheck(cudaFree(deviceInput.lstack_rlm));
    cudaErrorCheck(cudaFree(deviceInput.idx_gl_1d_rlm_j));
    cudaErrorCheck(cudaFree(deviceInput.radius_1d_rlm_r));
    cudaErrorCheck(cudaFree(deviceInput.weight_rtm));
    cudaErrorCheck(cudaFree(deviceInput.mdx_p_rlm_rtm));
    cudaErrorCheck(cudaFree(deviceInput.mdx_n_rlm_rtm));
    cudaErrorCheck(cudaFree(deviceInput.asin_theta_1d_rtm));
  #ifndef CUDA_OTF
    cudaErrorCheck(cudaFree(deviceInput.p_jl));
    cudaErrorCheck(cudaFree(deviceInput.dP_jl));
    cudaErrorCheck(cudaFree(deviceInput.p_rtm));
    cudaErrorCheck(cudaFree(deviceInput.dP_rtm));
  #endif
    cudaErrorCheck(cudaFree(deviceInput.reductionSpace));    
}

void deAllocDebugMem() {
  #if defined(CUDA_OTF) 
    free(h_debug.P_smdt);
    free(h_debug.dP_smdt);
  #endif
    free(h_debug.vr_rtm);
    free(h_debug.sp_rlm);
//  free(h_debug.g_sph_rlm);
  #if defined(CUDA_OTF) 
    cudaErrorCheck(cudaFree(d_debug.P_smdt));
    cudaErrorCheck(cudaFree(d_debug.dP_smdt));
  #endif
//  cudaErrorCheck(cudaFree(d_debug.g_sph_rlm));
}

void cleangpu_() {
  deAllocMemOnGPU();
  deAllocDebugMem();
  for(int i=0; i<nStreams; i++)
    cudaErrorCheck(cudaStreamDestroy(streams[i]));
  #if defined(CUDA_TIMINGS)
    cudaProfilerStop();
  #endif

  //Write performance metrics
  cudaPerformance.echoAllClocks();
  cudaPerformance.closeStream();
}

//Fortran wrapper function
void cuda_sync_device_() {
  cudaErrorCheck(cudaDeviceSynchronize());
}

void cudaDevSync() {
  cudaErrorCheck(cudaDeviceSynchronize());
}

size_t computeSharedMemory(int blockSize) {
  return blockSize * sizeof(double);
} 

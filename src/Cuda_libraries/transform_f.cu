#include <cuda_runtime.h>
#include <assert.h>

#include "legendre_poly.h"
#include "math_functions.h"
#include "math_constants.h"

void find_optimal_algorithm_(int *ncomp, int *nvector, int *nscalar) {
  constants.ncomp = *ncomp;
  constants.nscalar= *nscalar;
  constants.nvector = *nvector;

  dim3 grid(constants.nidx_rlm[1],constants.nidx_rtm[0],1);
  dim3 block(constants.nvector, constants.nidx_rtm[0],1);

  transFwd_vector algorithm;
//{originaAlgorithm, reductionAlgorithm};

  Timer wallClock;
  double elapsedTime=0;

  cout << "\tCUDA Fwd vector transform: \n"; 

  for(int i=0; i<2; i++) {
    wallClock.startTimer();
    switch (i) {
    case originaAlgorithm:
      cout << "\t\t static original Algorthim: ";
	  transF_vec<<<grid, block, 0>>> (deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.radius_1d_rlm_r, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.mdx_n_rlm_rtm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, deviceInput.p_rtm, deviceInput.dP_rtm, deviceInput.g_sph_rlm_7, deviceInput.asin_theta_1d_rtm, constants);
      break;
    case reductionAlgorithm:
	  cout << "\t\t static reduction Algorithm: ";
	  transF_vec_reduction< 9, 1,
                  cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                      double>
            <<<grid, 9>>> (deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, 
						deviceInput.sp_rlm, deviceInput.radius_1d_rlm_r, 
						deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, 
						deviceInput.mdx_n_rlm_rtm, deviceInput.a_r_1d_rlm_r, 
                        deviceInput.g_colat_rtm, deviceInput.p_rtm, 
						deviceInput.dP_rtm, deviceInput.g_sph_rlm_7, 
						deviceInput.asin_theta_1d_rtm, 
                        constants);
  
	  break;
    }
    cudaErrorCheck(cudaDeviceSynchronize());
    wallClock.endTimer();
    elapsedTime = wallClock.elapsedTime();
    cout << elapsedTime << "\n"; 
  }
}

__global__
void transF_vec(int kst, int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants) {
  //dim3 grid(constants.nidx_rlm[1],1,1);
  //dim3 block(constants.nidx_rtm[0],1,1);
  int k_rtm = threadIdx.x+kst-1;
  //int j_rlm = blockIdx.x;

// 3 for m-1, m, m+1
  unsigned int ip_rtm, in_rtm;

  double reg0, reg1, reg2, reg3, reg4;
  double sp1, sp2, sp3; 

  int order = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + blockIdx.x];
//  int degree = idx_gl_1d_rlm_j[constants.nidx_rlm[1] + blockIdx.x];
  double gauss_norm = g_sph_rlm_7[blockIdx.x];
  int nTheta = constants.nidx_rtm[1];
  int nVector = constants.nvector;
  int nComp = constants.ncomp;
  int istep_rtm_t = constants.istep_rtm[1];
  int istep_rtm_m = constants.istep_rtm[2];
  int istep_rlm_j = constants.istep_rlm[1];
  int istep_rlm_r = constants.istep_rlm[0];

  int mdx_p = mdx_p_rlm_rtm[blockIdx.x] - 1;
  ip_rtm = k_rtm * constants.istep_rtm[0];
  int mdx_n = mdx_n_rlm_rtm[blockIdx.x] - 1;
  mdx_p *= constants.istep_rtm[2];
  mdx_n *= constants.istep_rtm[2];
  mdx_p += ip_rtm;
  mdx_n += ip_rtm;

  int idx;
  int idx_p_rtm = blockIdx.x*nTheta; 
 
  double r_1d_rlm_r = radius_1d_rlm_r[k_rtm]; 
  int idx_sp = nComp * ( blockIdx.x*istep_rlm_j + k_rtm*istep_rlm_r); 

  for(int t=1; t<=nVector; t++) {
    sp1=sp2=sp3=0;
    for(int l_rtm=0; l_rtm<nTheta; l_rtm++) {
      ip_rtm = 3*t + nComp * (l_rtm * istep_rtm_t + mdx_p); 
      in_rtm = 3*t + nComp * (l_rtm * istep_rtm_t + mdx_n); 

      idx = idx_p_rtm + l_rtm; 
      reg0 = __dmul_rd(gauss_norm, weight_rtm[l_rtm]);
      reg1 = __dmul_rd(reg0, P_rtm[idx]);
      reg2 = __dmul_rd(reg0, dP_rtm[idx]);
      reg4 = __dmul_rd(P_rtm[idx], (double) order);
      reg1 = __dmul_rd(asin_theta_1d_rtm[l_rtm], reg0);
      reg3 = __dmul_rd(reg4, reg1);         

      sp1 += __dmul_rd(vr_rtm[ip_rtm-3], reg1);
      reg0 = __dmul_rd(vr_rtm[ip_rtm-2], reg2);
      reg4 =  -1 * __dmul_rd(vr_rtm[in_rtm-1], reg3);
      reg3 *= vr_rtm[in_rtm-2];
      reg2 *= vr_rtm[ip_rtm-1];
      sp2 += __dadd_rd(reg0, reg4); 
      sp3 -= __dadd_rd(reg3, reg2); 
    }
    idx_sp += 3; 

    sp_rlm[idx_sp-3] += __dmul_rd(__dmul_rd(r_1d_rlm_r, r_1d_rlm_r), sp1);
    sp_rlm[idx_sp-2] += __dmul_rd(r_1d_rlm_r, sp2);
    sp_rlm[idx_sp-1] += __dmul_rd(r_1d_rlm_r, sp3);

  }
}

__global__
void transF_vec(int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants) {
  //dim3 grid(constants.nidx_rlm[1],1,1);
  //dim3 block(nVector, constants.nidx_rtm[0],1,1);

  int k_rtm = threadIdx.y;

// 3 for m-1, m, m+1
  unsigned int ip_rtm, in_rtm;

  double reg0, reg1, reg2, reg3, reg4;
  double sp1, sp2, sp3; 

  int order = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + blockIdx.x];
//  int degree = idx_gl_1d_rlm_j[constants.nidx_rlm[1] + blockIdx.x];
  double gauss_norm = g_sph_rlm_7[blockIdx.x];
  int nTheta = constants.nidx_rtm[1];
  int nVector = constants.nvector;
  int nComp = constants.ncomp;
  int istep_rtm_t = constants.istep_rtm[1];
  int istep_rtm_m = constants.istep_rtm[2];
  int istep_rlm_j = constants.istep_rlm[1];
  int istep_rlm_r = constants.istep_rlm[0];

  int mdx_p = mdx_p_rlm_rtm[blockIdx.x] - 1;
  ip_rtm = k_rtm * constants.istep_rtm[0];
  int mdx_n = mdx_n_rlm_rtm[blockIdx.x] - 1;
  mdx_p *= constants.istep_rtm[2];
  mdx_n *= constants.istep_rtm[2];
  mdx_p += ip_rtm;
  mdx_n += ip_rtm;

  int idx;
  int idx_p_rtm = blockIdx.x*nTheta; 
 
  double r_1d_rlm_r = radius_1d_rlm_r[k_rtm]; 
  int idx_sp = nComp * ( blockIdx.x*istep_rlm_j + k_rtm*istep_rlm_r); 

  sp1=sp2=sp3=0;
  for(int l_rtm=0; l_rtm<nTheta; l_rtm++) {
    ip_rtm = 3*(threadIdx.x+1) + nComp * (l_rtm * istep_rtm_t + mdx_p); 
    in_rtm = 3*(threadIdx.x+1) + nComp * (l_rtm * istep_rtm_t + mdx_n); 

    idx = idx_p_rtm + l_rtm; 
    reg0 = __dmul_rd(gauss_norm, weight_rtm[l_rtm]);
    reg1 = __dmul_rd(reg0, P_rtm[idx]);
    reg2 = __dmul_rd(reg0, dP_rtm[idx]);
    reg4 = __dmul_rd(P_rtm[idx], (double) order);
    reg1 = __dmul_rd(asin_theta_1d_rtm[l_rtm], reg0);
    reg3 = __dmul_rd(reg4, reg1);         

    sp1 += __dmul_rd(vr_rtm[ip_rtm-3], reg1);
    reg0 = __dmul_rd(vr_rtm[ip_rtm-2], reg2);
    reg4 =  -1 * __dmul_rd(vr_rtm[in_rtm-1], reg3);
    reg3 *= vr_rtm[in_rtm-2];
    reg2 *= vr_rtm[ip_rtm-1];
    sp2 += __dadd_rd(reg0, reg4); 
    sp3 -= __dadd_rd(reg3, reg2); 
  }
  idx_sp += 3; 

  sp_rlm[idx_sp-3] += __dmul_rd(__dmul_rd(r_1d_rlm_r, r_1d_rlm_r), sp1);
  sp_rlm[idx_sp-2] += __dmul_rd(r_1d_rlm_r, sp2);
  sp_rlm[idx_sp-1] += __dmul_rd(r_1d_rlm_r, sp3);
}

//Reduction using an open source library CUB supported by nvidia
template <
    int     THREADS_PER_BLOCK,
    int			ITEMS_PER_THREAD,
    cub::BlockReduceAlgorithm ALGORITHM,
    typename T>
__global__ void transF_vec_reduction(int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants) {
  //dim3 grid(constants.nidx_rlm[1],constants.nidx_rtm[0],1);
  //dim3 block(nThreads,1,1);
  // nThreads * ITEMS_PER_THREAD = nTheta

  typedef cub::BlockReduce<T, THREADS_PER_BLOCK, ALGORITHM> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  int k_rtm = blockIdx.y;
  int j_rlm = blockIdx.x;

// 3 for m-1, m, m+1
  unsigned int ip_rtm, in_rtm;

  double reg0, reg1, reg2, reg3, reg4;
  double sp1, sp2, sp3; 

  int order = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + j_rlm];
//  int degree = idx_gl_1d_rlm_j[constants.nidx_rlm[1] + j_rlm];
  double gauss_norm = g_sph_rlm_7[j_rlm];
  int nTheta = constants.nidx_rtm[1];
  int nVector = constants.nvector;
  int nComp = constants.ncomp;

  int mdx_p = mdx_p_rlm_rtm[blockIdx.x] - 1;
  ip_rtm = k_rtm * constants.istep_rtm[0];
  int mdx_n = mdx_n_rlm_rtm[blockIdx.x] - 1;
  mdx_p *= constants.istep_rtm[2];
  mdx_n *= constants.istep_rtm[2];
  mdx_p += ip_rtm;
  mdx_n += ip_rtm;

  int idx;
  int idx_p_rtm = blockIdx.x*nTheta; 
 
  double r_1d_rlm_r = radius_1d_rlm_r[k_rtm]; 
  int idx_sp = nComp * ( blockIdx.x*constants.istep_rlm[1] + k_rtm*constants.istep_rlm[0]); 

  double poloidal[ITEMS_PER_THREAD];
  double radial_diff_poloidal[ITEMS_PER_THREAD]; 
  double toroidal[ITEMS_PER_THREAD];

  unsigned int l_rtm=0;

  for(int t=1; t<=nVector; t++) {
    sp1=sp2=sp3=0;
    for(int counter=0; counter < ITEMS_PER_THREAD; counter++) {  
      l_rtm = blockDim.x*counter + threadIdx.x; 
      ip_rtm = 3*t + nComp * (l_rtm * constants.istep_rtm[1] + mdx_p); 
      in_rtm = 3*t + nComp * (l_rtm * constants.istep_rtm[1] + mdx_n); 

      idx = idx_p_rtm + l_rtm; 
      reg0 = __dmul_rd(gauss_norm, weight_rtm[l_rtm]);
      reg1 = __dmul_rd(reg0, P_rtm[idx]);
      reg2 = __dmul_rd(reg0, dP_rtm[idx]);
      reg4 = __dmul_rd(P_rtm[idx], (double) order);
      reg1 = __dmul_rd(asin_theta_1d_rtm[l_rtm], reg0);
      reg3 = __dmul_rd(reg4, reg1);         

      poloidal[counter] = __dmul_rd(vr_rtm[ip_rtm-3], reg1);
      reg0 = __dmul_rd(vr_rtm[ip_rtm-2], reg2);
      reg4 =  -1 * __dmul_rd(vr_rtm[in_rtm-1], reg3);
      reg3 *= vr_rtm[in_rtm-2];
      reg2 *= vr_rtm[ip_rtm-1];
      radial_diff_poloidal[counter] = __dadd_rd(reg0, reg4); 
      // After the reduction, toroidal[...] * -1
      toroidal[counter] = __dadd_rd(reg3, reg2); 
    }
    
    idx_sp += 3; 

    __syncthreads();
    sp1 = BlockReduceT(temp_storage).Sum(poloidal);
    __syncthreads();
    sp2 = BlockReduceT(temp_storage).Sum(radial_diff_poloidal);
    __syncthreads();
    sp3 = -1 * BlockReduceT(temp_storage).Sum(toroidal);

    sp_rlm[idx_sp-3] += __dmul_rd(__dmul_rd(r_1d_rlm_r, r_1d_rlm_r), sp1);
    sp_rlm[idx_sp-2] += __dmul_rd(r_1d_rlm_r, sp2);
    sp_rlm[idx_sp-1] += __dmul_rd(r_1d_rlm_r, sp3);
  }
}

__global__
void transF_scalar(int kst, double *vr_rtm, double *sp_rlm, double *weight_rtm, int *mdx_p_rlm_rtm, double *P_rtm, double *g_sph_rlm_7, const Geometry_c constants) {
  int k_rtm = threadIdx.x+kst-1;

// 3 for m-1, m, m+1
  unsigned int ip_rtm;

  double gauss_norm = g_sph_rlm_7[blockIdx.x];
  int nTheta = constants.nidx_rtm[1];
  int nVector = constants.nvector;
  int nScalar= constants.nscalar;
  int nComp = constants.ncomp;
  int istep_rtm_r = constants.istep_rtm[0];
  int istep_rtm_t = constants.istep_rtm[1];
  int istep_rtm_m = constants.istep_rtm[2];
  int istep_rlm_r = constants.istep_rlm[0];
  int istep_rlm_j = constants.istep_rlm[1];

  double sp1;
  int mdx_p = mdx_p_rlm_rtm[blockIdx.x];
  int idx_p_rtm = blockIdx.x*nTheta; 
  int idx;
 
  for(int t=1; t<=nScalar; t++) {
    sp1 = 0;
    for(int l_rtm=1; l_rtm<=nTheta; l_rtm++) {
      ip_rtm = t + 3*nVector + nComp * ((l_rtm-1) * istep_rtm_t + k_rtm * istep_rtm_r + (mdx_p-1)*istep_rtm_m); 
      idx = idx_p_rtm + l_rtm - 1; 
      sp1 += __dmul_rd(vr_rtm[ip_rtm-1],__dmul_rd(__dmul_rd(gauss_norm, weight_rtm[l_rtm-1]), P_rtm[idx]));
    } 
     
    idx = t + 3*nVector + nComp*((blockIdx.x) * istep_rlm_j + k_rtm*istep_rlm_r); 
    sp_rlm[idx-1] += sp1;
  } 
}

//Reduction using an open source library CUB supported by nvidia
template <
    int     THREADS_PER_BLOCK,
    int			ITEMS_PER_THREAD,
    cub::BlockReduceAlgorithm ALGORITHM,
    typename T>
__global__
void transF_scalar_reduction(double *vr_rtm, double *sp_rlm, double *weight_rtm, int *mdx_p_rlm_rtm, double *P_rtm, double *g_sph_rlm_7, const Geometry_c constants) {
//grid(nidx_rlm[1], nidx_rlm[0])

  typedef cub::BlockReduce<T, THREADS_PER_BLOCK, ALGORITHM> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;  

  int k_rtm = blockIdx.y;
  int l_rtm; 

// 3 for m-1, m, m+1
  unsigned int ip_rtm;

  double gauss_norm = g_sph_rlm_7[blockIdx.x];
  int nTheta = constants.nidx_rtm[1];
  int nVector = constants.nvector;
  int nScalar= constants.nscalar;
  int nComp = constants.ncomp;

  int mdx_p = mdx_p_rlm_rtm[blockIdx.x];
  int idx_p_rtm = blockIdx.x*nTheta; 
  int idx;

  double spectral[ITEMS_PER_THREAD]; 

  for(int t=1; t<=nScalar; t++) {
    for(int counter = 0; counter < ITEMS_PER_THREAD; counter ++) {
      l_rtm = blockDim.x*counter + threadIdx.x; 
      ip_rtm = t + 3*nVector + nComp * (l_rtm * constants.istep_rtm[1] + k_rtm * constants.istep_rtm[0] + (mdx_p-1)*constants.istep_rtm[2]); 
      idx = idx_p_rtm + l_rtm; 
	  spectral[counter] = __dmul_rd(vr_rtm[ip_rtm-1],__dmul_rd(__dmul_rd(gauss_norm, weight_rtm[l_rtm]), P_rtm[idx]));
    }
    idx = t + 3*nVector + nComp*((blockIdx.x) * constants.istep_rlm[1] + k_rtm*constants.istep_rlm[0]); 
    __syncthreads();
    sp_rlm[idx-1] = BlockReduceT(temp_storage).Sum(spectral);
  } 
}

void legendre_f_trans_cuda_(int *ncomp, int *nvector, int *nscalar) {
  static int nShells = constants.nidx_rtm[0];
  static int nTheta = constants.nidx_rtm[1];

  dim3 grid(constants.nidx_rlm[1],nShells,1);
  dim3 block(constants.nvector, constants.nidx_rtm[0],1);

  constants.ncomp = *ncomp;
  constants.nscalar= *nscalar;
  constants.nvector = *nvector;

  static Timer transF_s("Fwd scalar reduction algorithm 9 threads 1 Items");
  cudaPerformance.registerTimer(&transF_s);
  transF_s.startTimer();
  transF_scalar_reduction <9, 1, 
                     cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                     double>
               <<<grid, 9>>> (deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.p_rtm, deviceInput.g_sph_rlm_7, constants);
  
  transF_scalar<<<grid, nShells, 0>>> (1, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.p_rtm, deviceInput.g_sph_rlm_7, constants);
  cudaDevSync();
  transF_s.endTimer();
  
  //ToDo: Ponder this: if not exact, what are the consequences?
  //Extremeley important! *****
  //int itemsPerThread = constants.nidx_rtm[1]/blockSize; 
  //std::assert(itemsPerThread*blockSize == constants.nidx_rtm[1]);
  //std::assert(minGridSize <= constants.nidx_rlm[1]);
  static Timer transf_reduce_32_3("fwd vector reduction algorithm 64 threads/block");
  cudaPerformance.registerTimer(&transf_reduce_32_3);
  transf_reduce_32_3.startTimer();
  transF_vec_reduction< 9, 1,
                  cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                      double>
            <<<grid, 9>>> (deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.radius_1d_rlm_r, 
                        deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.mdx_n_rlm_rtm, deviceInput.a_r_1d_rlm_r, 
                        deviceInput.g_colat_rtm, deviceInput.p_rtm, deviceInput.dP_rtm, deviceInput.g_sph_rlm_7, deviceInput.asin_theta_1d_rtm, 
                        constants);

  transF_vec<<<grid, block, 0>>> (deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.radius_1d_rlm_r, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.mdx_n_rlm_rtm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, deviceInput.p_rtm, deviceInput.dP_rtm, deviceInput.g_sph_rlm_7, deviceInput.asin_theta_1d_rtm, constants);

  cudaDevSync();
  transf_reduce_32_3.endTimer();
}

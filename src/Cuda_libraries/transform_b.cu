#include <cuda_runtime.h>
#include "legendre_poly.h"
#include "math_functions.h"
#include "math_constants.h"
#include <sstream>


void initDevConstVariables() {
//  cudaError_t error;
//  error = cudaMemcpyToSymbol(lstack_rlm, &constants, sizeof(Geometry_c), 0, cudaMemcpyHostToDevice);
//  cudaErrorCheck(error);
}

template <
    int     THREADS_PER_BLOCK,
    cub::BlockReduceAlgorithm ALGORITHM,
    typename T>
__global__
void transB_scalar_reduction(int *lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *P_jl, const Geometry_c constants) {
  //dim3  grid(nidx_rtm[1], nidx_rtm[0])
  //dim3 block(t_lvl+1)

  typedef cub::BlockReduce<T, THREADS_PER_BLOCK, ALGORITHM> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  int l_rtm = blockIdx.x;
  int jst, jed, j_rlm;

  unsigned int idx_p_jl=0, idx=0, idx_rtm=0;

  double vrs1;
  double P_smdt;
 
  int reg1 = 3*constants.nvector + constants.ncomp*blockIdx.y*constants.istep_rlm[0];
  int reg2 = 3*constants.nvector + constants.ncomp * ((blockIdx.x) * constants.istep_rtm[1] + blockIdx.y*constants.istep_rtm[0]); 

  for(int mp_rlm=0; mp_rlm < constants.nidx_rtm[2]; mp_rlm++) { 
	jst = lstack_rlm[mp_rlm] + 1;
	jed = lstack_rlm[mp_rlm+1];
    int totalWorkLoad = jed-jst+1;
    int threadWorkLoad = totalWorkLoad/THREADS_PER_BLOCK; 
    if( totalWorkLoad % THREADS_PER_BLOCK < threadIdx.x )
      threadWorkLoad++;
    // threadWorkLoad is a negative number... jed is 1 and jst > 1
    // Block (0,5,0) && mp_rlm = 5
    int workingThreads = min(totalWorkLoad, blockDim.x);

    if (threadIdx.x < workingThreads) {
	  j_rlm = jst-1 + threadIdx.x;
      idx_p_jl = constants.nidx_rlm[1]*l_rtm;
      for(int t=1; t<=constants.nscalar; t++) {
        j_rlm = jst-1 + threadIdx.x;
        vrs1=0;
        for( int counter = 0; counter < threadWorkLoad; j_rlm += blockDim.x, counter++) {
          idx = reg1 + t + constants.ncomp*j_rlm*constants.istep_rlm[1]; 
          P_smdt = P_jl[idx_p_jl + j_rlm]; 
          vrs1 += sp_rlm[idx - 1] * P_smdt;
        }
        idx_rtm = reg2 + t + mp_rlm * constants.istep_rtm[2]; 
        __syncthreads();
        vr_rtm[idx_rtm - 1] = BlockReduceT(temp_storage).Sum(vrs1, workingThreads); 
      }
    }
  }
}

__global__
void transB_scalar(int *lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *P_jl, Geometry_c constants) {
  //dim3 grid3(nTheta, constants.nidx_rtm[2]);
  //dim3 block3(nShells,1,1);
 // mp_rlm is the blockIdx.y 
  double vrs1;

  int jst = lstack_rlm[blockIdx.y] + 1;
  int jed = lstack_rlm[blockIdx.y+1];
  int idx_p_jl=0, idx=0, idx_rtm=0; 
  int reg1 = 3*constants.nvector + constants.ncomp*threadIdx.x*constants.istep_rlm[0];

  for(int t=1; t<=constants.nscalar; t++) {
    vrs1 = 0;
    idx_p_jl = constants.nidx_rlm[1]*blockIdx.x+jst-1;
    for(int j_rlm=jst; j_rlm<=jed; j_rlm++) {
      idx = reg1 + t + constants.ncomp*(j_rlm-1)*constants.istep_rlm[1]; 
      vrs1 += sp_rlm[idx - 1] * P_jl[idx_p_jl];
      idx_p_jl++;
    } 
      
    idx_rtm = t + 3*constants.nvector + constants.ncomp*((blockIdx.x) * constants.istep_rtm[1] + threadIdx.x*constants.istep_rtm[0] + (blockIdx.y)*constants.istep_rtm[2]); 
    vr_rtm[idx_rtm - 1] = vrs1;
  } 
}

//Reduction using an open source library CUB supported by nvidia
template <
    int     THREADS_PER_BLOCK,
    cub::BlockReduceAlgorithm ALGORITHM,
    typename T>
__global__
void transB_dydt_reduction(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *g_sph_rlm, double *a_r_1d_rlm_r, double *P_jl, double *dP_jl, const Geometry_c constants) {
  //dim3  grid(nTheta, nidx_rtm[0])
  //dim3 block(nThreads)

  typedef cub::BlockReduce<T, THREADS_PER_BLOCK, ALGORITHM> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  int l_rtm = blockIdx.x;
  int k_rtm = blockIdx.y;
  unsigned int idx_p_jl=0, idx=0, idx_rtm=0;

  int j_rlm = 0;

  double a_r_1d_rlm_r_ = a_r_1d_rlm_r[blockIdx.y];
  double vr1, vr2, vr3;

  for(int mp_rlm = 0; mp_rlm < constants.nidx_rtm[2]; mp_rlm++) { 
    int jst = lstack_rlm[mp_rlm] + 1;
	int jed = lstack_rlm[mp_rlm+1];
    int totalWorkLoad = jed-jst+1;
    int threadWorkLoad = totalWorkLoad/THREADS_PER_BLOCK; 
    if( totalWorkLoad % THREADS_PER_BLOCK < threadIdx.x )
      threadWorkLoad++;
    int workingThreads = min(totalWorkLoad, blockDim.x);

    if (threadIdx.x < workingThreads) {
      for(int t=1; t<=constants.nvector; t++) {
        j_rlm = jst - 1 + threadIdx.x;
        vr1=vr2=vr3=0;
		for(int counter = 0; counter < totalWorkLoad; j_rlm += blockDim.x, counter++) {
		  idx = 3*t + constants.ncomp * (j_rlm * constants.istep_rlm[1] + k_rtm * constants.istep_rlm[0]); 
		  idx_p_jl = constants.nidx_rlm[1]*l_rtm+j_rlm;
		  vr3 += sp_rlm[idx - 3] * __dmul_rd(a_r_1d_rlm_r_, a_r_1d_rlm_r_) * P_jl[idx_p_jl] * g_sph_rlm[j_rlm];    
		  vr2 += sp_rlm[idx - 2]  * a_r_1d_rlm_r_ * dP_jl[idx_p_jl];    
		  vr1 -= sp_rlm[idx - 1] * a_r_1d_rlm_r_ * dP_jl[idx_p_jl];    
        }

        idx_rtm = 3*t + constants.ncomp * (l_rtm * constants.istep_rtm[1] + k_rtm*constants.istep_rtm[0] + mp_rlm * constants.istep_rtm[2]); 
      
	   __syncthreads();
	   vr_rtm[idx_rtm - 2 - 1]  += BlockReduceT(temp_storage).Sum(vr3, workingThreads); 
	   __syncthreads();
	    vr_rtm[idx_rtm - 1 - 1]  += BlockReduceT(temp_storage).Sum(vr2, workingThreads); 
	   __syncthreads();
	   vr_rtm[idx_rtm - 1]  += BlockReduceT(temp_storage).Sum(vr1, workingThreads); 
     }
    }
  }
}

__global__
void transB_dydt_old(double *g_sph_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl, double *dP_jl, const Geometry_c constants) {
  //dim3 grid3(nTheta, constants.nidx_rtm[2]);
  //dim3 block3(nShells,1,1);

  int mp_rlm = blockIdx.y;
  double a_r_1d = a_r_1d_rlm_r[threadIdx.x];
  int jst = lstack_rlm_cmem[mp_rlm]+1;

  double vr1, vr2, vr3;
  int idx_p_jl=0, idx=0, idx_rtm=0; 
  
  int jed = lstack_rlm_cmem[mp_rlm+1];
  double a_r_1d_rlm_r_sq =  __dmul_rd(a_r_1d, a_r_1d);

  int reg1, reg2;
  double dreg1, dreg2;
  for(int t=1; t<=constants.nvector; t++) {
    reg1 = constants.nidx_rlm[1]*blockIdx.x;
    reg2 = jst-1;
    vr1=vr2=vr3=0;
    idx_p_jl = reg1 + reg2;
    for(int j_rlm=jst; j_rlm<=jed; j_rlm++) {
      idx = 3*t;
      reg1 = constants.ncomp * (j_rlm-1) * constants.istep_rlm[1];
      reg2 = constants.ncomp * threadIdx.x * constants.istep_rlm[0]; 
      idx += reg1 + reg2;
      dreg1 = __dmul_rn(P_jl[idx_p_jl], g_sph_rlm[j_rlm]);
      dreg2 = __dmul_rn(a_r_1d, dP_jl[idx_p_jl]);
      vr2 += __dmul_rn(sp_rlm[idx - 2], dreg2);    
      dreg1 *= a_r_1d_rlm_r_sq; 
      vr1 -= __dmul_rn(sp_rlm[idx - 1] , dreg2);
      idx_p_jl++;
      vr3 += __dmul_rn(sp_rlm[idx - 3] , dreg1);
    }
    idx_rtm = 3*t + constants.ncomp * ((blockIdx.x) * constants.istep_rtm[1] + threadIdx.x*constants.istep_rtm[0] + (mp_rlm) * constants.istep_rtm[2]); 

    vr_rtm[idx_rtm - 2 - 1]  += vr3; 
    vr_rtm[idx_rtm - 1 - 1]  += vr2; 
    vr_rtm[idx_rtm - 1]  += vr1; 
  }
}

__global__
void transB_dydt(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl, double *dP_jl, const Geometry_c constants) {
  //dim3 grid3(nTheta, constants.nidx_rtm[2]);
  //dim3 block3(nShells,1,1);

  int mp_rlm = blockIdx.y;
  int jst = lstack_rlm[mp_rlm] + 1;
  int jed = lstack_rlm[mp_rlm+1];

  double vr1, vr2, vr3;
  unsigned int idx_p_jl=0, idx=0, idx_rtm=0; 
  int deg = idx_gl_1d_rlm_j[constants.nidx_rlm[1] + jst -1];
  int ord = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + jst -1];

  //turn this into a terinary operator
  float g_sph_rlm=deg*(deg+1);
  if (ord==0 && deg==0)
    g_sph_rlm=0.5;

  for(int t=1; t<=constants.nvector; t++) {
    vr1=vr2=vr3=0;
    idx_p_jl = constants.nidx_rlm[1]*blockIdx.x+jst-1;
    for(int j_rlm=jst; j_rlm<=jed; j_rlm++) {
      idx = 3*t + constants.ncomp * ((j_rlm-1) * constants.istep_rlm[1] + threadIdx.x * constants.istep_rlm[0]); 
      vr3 += sp_rlm[idx - 3] * __dmul_rd(a_r_1d_rlm_r[threadIdx.x], a_r_1d_rlm_r[threadIdx.x]) * P_jl[idx_p_jl] * g_sph_rlm;    
      vr2 += sp_rlm[idx - 2]  * a_r_1d_rlm_r[threadIdx.x] * dP_jl[idx_p_jl];    
      vr1 -= sp_rlm[idx - 1] * a_r_1d_rlm_r[threadIdx.x] * dP_jl[idx_p_jl];    
      idx_p_jl++;
    }
    idx_rtm = 3*t + constants.ncomp * ((blockIdx.x) * constants.istep_rtm[1] + threadIdx.x*constants.istep_rtm[0] + (mp_rlm) * constants.istep_rtm[2]); 

    vr_rtm[idx_rtm - 2 - 1]  += vr3; 
    vr_rtm[idx_rtm - 1 - 1]  += vr2; 
    vr_rtm[idx_rtm - 1]  += vr1; 
  }
}

template <
    int     THREADS_PER_BLOCK,
    cub::BlockReduceAlgorithm ALGORITHM,
    typename T>
__global__
void transB_dydp_reduction(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl, double *asin_theta_1d_rtm, const Geometry_c constants) {
  //dim3  grid(nTheta, nidx_rtm[0])
  //dim3 block(nThreads)

  typedef cub::BlockReduce<T, THREADS_PER_BLOCK, ALGORITHM> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  int j_rlm = 0;
  int k_rtm = blockIdx.y;
  int l_rtm = blockIdx.x;

  unsigned int idx_p_jl=0, idx=0, idx_rtm=0;
  int ord;

  double a_r_1d_rlm_r_ = a_r_1d_rlm_r[k_rtm];

  double vr4, vr5;
  double reg2;

  double reg1 = __dmul_rd(a_r_1d_rlm_r_, asin_theta_1d_rtm[l_rtm]);

  for(int mp_rlm = 0; mp_rlm < constants.nidx_rtm[2]; mp_rlm++) {
    int mn_rlm = constants.nidx_rtm[2] - mp_rlm;
	int jst = lstack_rlm[mp_rlm] + 1;
	int jed = lstack_rlm[mp_rlm+1];
    int totalWorkLoad = jed-jst+1;
    int threadWorkLoad = totalWorkLoad/THREADS_PER_BLOCK; 
    if( totalWorkLoad % THREADS_PER_BLOCK < threadIdx.x )
      threadWorkLoad++;
    int workingThreads = min(totalWorkLoad, blockDim.x);

    ord = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + jst -1];

    if (threadIdx.x < workingThreads) {
      for(int t=1; t<=constants.nvector; t++) {
        vr4=vr5=0;
        j_rlm = jst - 1 + threadIdx.x; 
        for(int counter = 0; counter < workingThreads; counter++, j_rlm += blockDim.x) {
          idx = 3*t + constants.ncomp * (j_rlm * constants.istep_rlm[1] + blockIdx.y * constants.istep_rlm[0]); 
		  idx_p_jl = constants.nidx_rlm[1]*blockIdx.x+j_rlm;
		  reg2 = -1 * __dmul_rd( P_jl[idx_p_jl], __dmul_rd(reg1,(double) ord));         
          vr5 += sp_rlm[idx - 1] * reg2;
          vr4 += sp_rlm[idx - 2] * reg2;        
        }

        idx_rtm = 3*t + constants.ncomp * (l_rtm * constants.istep_rtm[1] + k_rtm*constants.istep_rtm[0] + (mn_rlm - 1) * constants.istep_rtm[2]); 
      
        __syncthreads();
        vr_rtm[idx_rtm - 1 - 1]  += BlockReduceT(temp_storage).Sum(vr5, workingThreads ); 
        __syncthreads();
        vr_rtm[idx_rtm - 1]  += BlockReduceT(temp_storage).Sum(vr4, workingThreads ); 
      }
    }
  }
}

//When looking at the transformed field data, the first component is off by a sign, oddly. 
__global__
void transB_dydp(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl,  double *asin_theta_1d_rtm, const Geometry_c constants) {
  //dim3 grid3(nTheta, constants.nidx_rtm[2]);
  //dim3 block3(nShells,1,1);
  extern __shared__ double cache[];
  unsigned int idx=0, idx_rtm=0;
  double reg2;
  double vr4, vr5;

  int mn_rlm = constants.nidx_rtm[2] - blockIdx.y;
  int jst = lstack_rlm[blockIdx.y] + 1;
  int jed = lstack_rlm[blockIdx.y+1];
  int order = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + jst -1]; 
  double asin = asin_theta_1d_rtm[blockIdx.x];
  int idx_p_jl=0; 

  cache[threadIdx.x] = a_r_1d_rlm_r[threadIdx.x] * order * asin;
  __syncthreads();

  for(int t=1; t<=constants.nvector; t++) {
    vr4=vr5=0;
    idx_p_jl = constants.nidx_rlm[1]*blockIdx.x+jst-1;
    for(int j_rlm=jst; j_rlm<=jed; j_rlm++) {
      idx = 3*t + constants.ncomp * ((j_rlm-1) * constants.istep_rlm[1] + threadIdx.x * constants.istep_rlm[0]); 
      reg2 = -1 * __dmul_rd( P_jl[idx_p_jl], cache[threadIdx.x]);         
      vr5 += sp_rlm[idx - 1] * reg2;
      vr4 += sp_rlm[idx - 2] * reg2;
      idx_p_jl++;
    }
    // mn_rlm
    idx_rtm = 3*t + constants.ncomp * ((blockIdx.x) * constants.istep_rtm[1] + threadIdx.x*constants.istep_rtm[0] + (mn_rlm-1) * constants.istep_rtm[2]); 

    vr_rtm[idx_rtm - 1 - 1] += vr5; 
    vr_rtm[idx_rtm - 1] += vr4; 
  }
}

void legendre_b_trans_cuda_(int *ncomp, int *nvector, int *nscalar) {
  
//  static int nShells = *ked - *kst + 1;
  static int nShells = constants.nidx_rtm[0];
  static int nTheta = constants.nidx_rtm[1];
 
  constants.ncomp = *ncomp;
  constants.nvector = *nvector;
  constants.nscalar = *nscalar;

  dim3 grid(nTheta, constants.nidx_rtm[2]);

  //The number of threads is an arbitrary value that will vary the amount of thread divergence, the amount of work per thread, and in turn the time efficiency. 

  static Timer transBwdVec_dy_dt("Bwd Vector dydt Transform");
  cudaPerformance.registerTimer(&transBwdVec_dy_dt);
  transBwdVec_dy_dt.startTimer();
  /*transB_dydt_reduction<32, 
                      cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                      double>
                <<<grid, 32>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.g_sph_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.dP_jl, constants);*/
  
  transB_dydt_old<<<grid, nShells>>> (deviceInput.g_sph_rlm, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.dP_jl, constants);
  cudaDevSync();
  transBwdVec_dy_dt.endTimer();

  static Timer transBwdVec_dy_dp("Bwd Vector dydp Transform");
  cudaPerformance.registerTimer(&transBwdVec_dy_dp);
  transBwdVec_dy_dp.startTimer();
  /*transB_dydp_reduction<32, 
                      cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                      double>
                <<<grid, 32>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.asin_theta_1d_rtm, constants);*/

  transB_dydp<<<grid, nShells, sizeof(double)*nShells>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.asin_theta_1d_rtm, constants);
  cudaDevSync();
  transBwdVec_dy_dp.endTimer(); 

  static Timer transBwdScalar("bwd scalar transform");
  cudaPerformance.registerTimer(&transBwdScalar);
  transBwdScalar.startTimer(); 
  /*transB_scalar_reduction<32,
                        cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                      double>
                <<<grid, 32>>> (deviceInput.lstack_rlm, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.p_jl, constants);*/
  transB_scalar<<<grid, nShells>>> (deviceInput.lstack_rlm, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.p_jl, constants);
  cudaDevSync();
  transBwdScalar.endTimer(); 
}

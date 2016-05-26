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


__global__
void transB_scalar(int mp_rlm_st, int *lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *P_jl, Geometry_c constants) {
  //dim3 grid3(nTheta, constants.nidx_rtm[2]);
  //dim3 block3(nShells,1,1);
 // mp_rlm is the blockIdx.y 
  double vrs1;
  int mp_rlm = blockIdx.y + mp_rlm_st;
  int jst = lstack_rlm[mp_rlm] + 1;
  int jed = lstack_rlm[mp_rlm+1];
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
      
    idx_rtm = t + 3*constants.nvector + constants.ncomp*((blockIdx.x) * constants.istep_rtm[1] + threadIdx.x*constants.istep_rtm[0] + mp_rlm*constants.istep_rtm[2]); 
    vr_rtm[idx_rtm - 1] = vrs1;
  } 
}


__global__
void transB_dydt(int mp_rlm_st, double *g_sph_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl, double *dP_jl, const Geometry_c constants) {
  //dim3 grid3(nTheta, constants.nidx_rtm[2]);
  //dim3 block3(nVector,nShells,1);

  double vr1=0, vr2=0, vr3=0;
  int idx_p_jl, idx, idx_rtm; 

  int mp_rlm = blockIdx.y + mp_rlm_st;
  int jst = lstack_rlm_cmem[mp_rlm];
  int jed = lstack_rlm_cmem[mp_rlm+1];

  double a_r_1d = a_r_1d_rlm_r[threadIdx.y];

  idx_p_jl = constants.nidx_rlm[1]*blockIdx.x + jst;
  idx = 3*(threadIdx.x+1) + constants.ncomp * (threadIdx.y * constants.istep_rlm[0] + jst * constants.istep_rlm[1]);
  int idxStride = constants.ncomp * constants.istep_rlm[1];

  double a_r_1d_rlm_r_sq =  __dmul_rd(a_r_1d, a_r_1d);
  double g_sph = g_sph_rlm[jst];
  double p = P_jl[idx_p_jl]; 
  double dpdt = dP_jl[idx_p_jl];  
  double sp1 = sp_rlm[idx-3];
  double sp2 = sp_rlm[idx-2];
  double sp3 = sp_rlm[idx-1];
  double dreg1, dreg2;


  //Common sum of physical data
  idx_rtm = 3*(threadIdx.x+1) + constants.ncomp * ((blockIdx.x) * constants.istep_rtm[1] + threadIdx.y*constants.istep_rtm[0] + (mp_rlm) * constants.istep_rtm[2]);


  for(int j_rlm=jst; j_rlm<jed-1;) {
    idx_p_jl++;
    j_rlm++;
    idx += idxStride;
    dreg1 = __dmul_rn(p, g_sph);
    dreg2 = __dmul_rn(a_r_1d, dpdt);
    p = P_jl[idx_p_jl];
    dpdt = dP_jl[idx_p_jl];  
    g_sph = g_sph_rlm[j_rlm];
    vr2 += __dmul_rn(sp2, dreg2);    
    dreg1 *= a_r_1d_rlm_r_sq; 
    vr1 -= __dmul_rn(sp3, dreg2);
    vr3 += __dmul_rn(sp1, dreg1);
    sp2 = sp_rlm[idx-2];
    sp3 = sp_rlm[idx-1];
    sp1 = sp_rlm[idx-3]; 
  }
 
  dreg1 = __dmul_rn(p, g_sph);
  dreg2 = __dmul_rn(a_r_1d, dpdt);
  vr2 += __dmul_rn(sp2, dreg2);    
  dreg1 *= a_r_1d_rlm_r_sq; 
  vr1 -= __dmul_rn(sp3, dreg2);
  vr3 += __dmul_rn(sp1, dreg1);
  
  vr_rtm[idx_rtm-2]  += vr2; 
  vr_rtm[idx_rtm-1]  += vr1; 
  vr_rtm[idx_rtm-3]  += vr3; 
}


//When looking at the transformed field data, the first component is off by a sign, oddly. 
__global__
void transB_dydp(int mp_rlm_st, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl,  double *asin_theta_1d_rtm, const Geometry_c constants) {
  //dim3 grid3(nTheta, constants.nidx_rtm[2]);
  //dim3 block3(nVector, nShells,1);

  unsigned int idx=0, idx_rtm=0;
  double reg2;
  double vr4=0, vr5=0;

  //int mn_rlm = constants.nidx_rtm[2] - blockIdx.y;
  int mn_rlm = mp_rlm_st + blockIdx.y;
  int jst = lstack_rlm_cmem[blockIdx.y];
  int jed = lstack_rlm_cmem[blockIdx.y+1];
  int order = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + jst]; 
  double asin = asin_theta_1d_rtm[blockIdx.x];
  int idx_p_jl=0; 
 
  int idxStride = constants.ncomp * constants.istep_rlm[1];
  // mn_rlm
  //idx_rtm = 3*(threadIdx.x+1) + constants.ncomp * ((blockIdx.x) * constants.istep_rtm[1] + threadIdx.y*constants.istep_rtm[0] + (mn_rlm-1) * constants.istep_rtm[2]); 
  idx_rtm = 3*(threadIdx.x+1) + constants.ncomp * ((blockIdx.x) * constants.istep_rtm[1] + threadIdx.y*constants.istep_rtm[0] + mn_rlm * constants.istep_rtm[2]); 

  double scalar  = a_r_1d_rlm_r[threadIdx.y] * order * asin;
  
  idx_p_jl = constants.nidx_rlm[1]*blockIdx.x+jst;
  idx = 3*(threadIdx.x+1) + constants.ncomp*(threadIdx.y * constants.istep_rlm[0] + jst * constants.istep_rlm[1]); 

  for(int j_rlm=jst; j_rlm<jed; j_rlm++) {
    reg2 = -1 * __dmul_rd( P_jl[idx_p_jl], scalar);
    vr5 += __dmul_rn(sp_rlm[idx - 1],reg2);
    vr4 += __dmul_rn(sp_rlm[idx - 2],reg2);
    idx_p_jl++;
    idx += idxStride; 
  }

  vr_rtm[idx_rtm - 1 - 1] += vr5; 
  vr_rtm[idx_rtm - 1] += vr4; 
}

void legendre_b_trans_cuda_(int *ncomp, int *nvector, int *nscalar, int *mp_rlm_st, int *mp_rlm_ed) {
  
//  static int nShells = *ked - *kst + 1;
  static int nShells = constants.nidx_rtm[0];
  static int nTheta = constants.nidx_rtm[1];
 
  constants.ncomp = *ncomp;
  constants.nvector = *nvector;
  constants.nscalar = *nscalar;

  //dim3 grid(nTheta, constants.nidx_rtm[2]);
  dim3 grid(nTheta, (*mp_rlm_ed - (*mp_rlm_st) + 1));
  dim3 block(*nvector, constants.nidx_rtm[0]);

  //The number of threads is an arbitrary value that will vary the amount of thread divergence, the amount of work per thread, and in turn the time efficiency. 

  /*transB_dydt_reduction<32, 
                      cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                      double>
                <<<grid, 32>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.g_sph_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.dP_jl, constants);*/
    
  transB_dydt<<<grid, block, 0, streams[0]>>> (*mp_rlm_st -1, deviceInput.g_sph_rlm, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.dP_jl, constants);

  /*transB_scalar_reduction<32,
                        cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                      double>
                <<<grid, 32>>> (deviceInput.lstack_rlm, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.p_jl, constants);*/
  transB_scalar<<<grid, nShells, 0, streams[1]>>> (*mp_rlm_st - 1, deviceInput.lstack_rlm, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.p_jl, constants);

  /*transB_dydp_reduction<32, 
                      cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                      double>
                <<<grid, 32>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.asin_theta_1d_rtm, constants);*/

  transB_dydp<<<grid, block, 0, streams[0]>>> (*mp_rlm_st-1, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.asin_theta_1d_rtm, constants);

}

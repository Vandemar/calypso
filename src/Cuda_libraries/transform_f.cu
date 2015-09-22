#include <cuda_runtime.h>
#include "legendre_poly.h"
#include "math_functions.h"
#include "math_constants.h"
//#include <sstream>

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
  int istep_rtm_r = constants.istep_rtm[0];
  int istep_rtm_t = constants.istep_rtm[1];
  int istep_rtm_m = constants.istep_rtm[2];
  int istep_rlm_r = constants.istep_rlm[0];
  int istep_rlm_j = constants.istep_rlm[1];

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
void transF_vec_smem_schmidt(int kst, int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants) {
  extern __shared__ double cache[];
  //dim3 grid(constants.nidx_rlm[1],1,1);
  //dim3 block(constants.nidx_rtm[0],nvec,1);
  int k_rtm = threadIdx.x+kst-1;
  //int j_rlm = blockIdx.x;

// 3 for m-1, m, m+1
  unsigned int ip_rtm, in_rtm;

  double reg0, reg2, reg3, reg4;
  double sp1, sp2, sp3; 
 

  int order = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + blockIdx.x];
//  int degree = idx_gl_1d_rlm_j[constants.nidx_rlm[1] + blockIdx.x];
  double gauss_norm = g_sph_rlm_7[blockIdx.x];
  int nTheta = constants.nidx_rtm[1];
  int nVector = constants.nvector;
  int nComp = constants.ncomp;

  int me = threadIdx.x * blockDim.y + threadIdx.y;
  int idx_p_rtm = blockIdx.x*nTheta; 

  while(me < nTheta) {
    reg0 = __dmul_rd(gauss_norm, weight_rtm[me]);
    cache[me] = P_rtm[idx_p_rtm+me] * reg0;
    cache[me+nTheta] = dP_rtm[idx_p_rtm+me] * reg0;
    me += blockDim.x*blockDim.y;
  }
 
  int istep_rtm_t = constants.istep_rtm[1];

  int mdx_p = mdx_p_rlm_rtm[blockIdx.x] - 1;
  ip_rtm = k_rtm * constants.istep_rtm[0];
  int mdx_n = mdx_n_rlm_rtm[blockIdx.x] - 1;
  mdx_p *= constants.istep_rtm[2];
  mdx_n *= constants.istep_rtm[2];
  mdx_p += ip_rtm;
  mdx_n += ip_rtm;

  int idx;
 
  double r_1d_rlm_r = radius_1d_rlm_r[k_rtm]; 

  __syncthreads(); 

    sp1=sp2=sp3=0;
    for(int l_rtm=0; l_rtm<nTheta; l_rtm++) {
      ip_rtm = 3*(threadIdx.y+1) + nComp * (l_rtm * istep_rtm_t + mdx_p); 
      in_rtm = 3*(threadIdx.y+1) + nComp * (l_rtm * istep_rtm_t + mdx_n); 

      reg4 = __dmul_rd(cache[l_rtm], (double) order);
      reg3 = __dmul_rd(asin_theta_1d_rtm[l_rtm], reg4);

      sp1 += __dmul_rd(vr_rtm[ip_rtm-3], cache[l_rtm]);
      reg0 = __dmul_rd(vr_rtm[ip_rtm-2], cache[l_rtm+nTheta]);
      reg4 =  -1 * __dmul_rd(vr_rtm[in_rtm-1], reg3);
      reg3 *= vr_rtm[in_rtm-2];
      reg2 = __dmul_rd(vr_rtm[ip_rtm-1], cache[l_rtm+nTheta]);
      sp2 += __dadd_rd(reg0, reg4); 
      sp3 -= __dadd_rd(reg3, reg2); 
    }
    int idx_sp = 3*(threadIdx.y+1) + nComp * ( blockIdx.x*constants.istep_rlm[1] + k_rtm*constants.istep_rlm[0]); 

    sp_rlm[idx_sp-3] += __dmul_rd(__dmul_rd(r_1d_rlm_r, r_1d_rlm_r), sp1);
    sp_rlm[idx_sp-2] += __dmul_rd(r_1d_rlm_r, sp2);
    sp_rlm[idx_sp-1] += __dmul_rd(r_1d_rlm_r, sp3);

}

__global__
void transF_scalar(int kst, double *vr_rtm, double *sp_rlm, double *weight_rtm, int *mdx_p_rlm_rtm, double *P_rtm, double *g_sph_rlm_7, const Geometry_c constants) {
  int k_rtm = threadIdx.x+kst-1;

// 3 for m-1, m, m+1
  unsigned int ip_rtm;

  double sp1=0; 

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

void legendre_f_trans_cuda_(int *ncomp, int *nvector, int *nscalar) {
//  static int nShells = *ked - *kst + 1;
  static int nShells = constants.nidx_rtm[0];
  static int nTheta = constants.nidx_rtm[1];

  dim3 grid(constants.nidx_rlm[1],1,1);
  dim3 block(constants.nidx_rtm[0],1,1);

  constants.ncomp = *ncomp;
  constants.nscalar= *nscalar;
  constants.nvector = *nvector;

//  transF_vec<<<grid, block, 0, streams[0]>>> (1, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.radius_1d_rlm_r, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.mdx_n_rlm_rtm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, deviceInput.p_rtm, deviceInput.dP_rtm, deviceInput.g_sph_rlm_7, deviceInput.asin_theta_1d_rtm, constants);

  dim3 block2(constants.nidx_rtm[0],constants.nvector,1);
  transF_vec_smem_schmidt<<<grid, block2, sizeof(double)*nTheta*2, streams[0]>>> (1, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.radius_1d_rlm_r, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.mdx_n_rlm_rtm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, deviceInput.p_rtm, deviceInput.dP_rtm, deviceInput.g_sph_rlm_7, deviceInput.asin_theta_1d_rtm, constants);
  transF_scalar<<<grid, block, 0, streams[1]>>> (1, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.p_rtm, deviceInput.g_sph_rlm_7, constants);
}

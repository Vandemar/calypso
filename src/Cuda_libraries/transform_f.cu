#include <cuda_runtime.h>
#include "legendre_poly.h"
#include "math_functions.h"
#include "math_constants.h"
//#include <sstream>

#ifndef CUDA_DEBUG
__global__
void transF_m_l_OTF(int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_rtm, double *dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants) {
#else
__global__
void transF_m_l_OTF(int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants) {
#endif
  // sdata is a container for legendre polynomials and the derivative of the legendre polynomials with respect to theta
  // The data is organized by nTheta leg polys and then nTheta dp dtheta leg polys.
  extern __shared__ double sdata;
  //grid(nModes,1,1)
  //block(nShells, nvector)
  int m, l;
  int order, degree;
  l = constants.nidx_rlm[1]; 
  m = constants.nidx_rlm[1] * 2; 
  m += blockIdx.x;
  l += blockIdx.x;
//In fortran, column-major:
//idx_gl_1d_rlm_j(j,1): global ID for spherical harmonics
//idx_gl_1d_rlm_j(j,2): spherical hermonincs degree
//idx_gl_1d_rlm_j(j,3): spherical hermonincs order

  order = abs(idx_gl_1d_rlm_j[m]); 
  degree = idx_gl_1d_rlm_j[l]; 
  
  if(degree == 0) 
    return;
  
  int mdx_p = mdx_p_rlm_rtm[blockIdx.x] - 1;
  int ip_rtm = k_rtm * constants.istep_rtm[0];
  int mdx_n = mdx_n_rlm_rtm[blockIdx.x] - 1;
  mdx_p *= constants.istep_rtm[2];
  mdx_n *= constants.istep_rtm[2];
  mdx_p += ip_rtm;
  mdx_n += ip_rtm;

  int idx;
  int idx_p_rtm = blockIdx.x*nTheta; 
 
  double r_1d_rlm_r = radius_1d_rlm_r[k_rtm]; 
  int idx_sp = constants.ncomp * (blockIdx.x*constants.istep_rlm[1] + threadIdx.x*constants.istep_rlm[0]);
  double theta = g_colat_rtm[blockIdx.x]; 
  
  // m,l
  double lgp=1;
  double sin_theta_l=1; 
  for(int k=1; k<=degree; k++) {
    lgp *= __ddiv_ru((double)2*k-1, (double)2*k);
  }
  double cos_theta = cos(theta);
  double sin_theta = sin(theta);
  for(int k=0; k<degree; k++)
    sin_theta_l *= sin_theta;
  double reg1 = __dmul_rd((double)2, lgp);
  double p_m_l_0 = __dsqrt_rd(reg1);
  double reg2 = __dmul_rd((double) (2*order + 1), reg1); 
  // m,l+1
  degree++;
  double p_m_l_1 = __dmul_rd(__dsqrt_rd(reg2), cos_theta);
  // Normalize p_m_l_0 and p_m_l_1
  p_m_l_0 = __dmul_rd(p_m_l_0, sin_theta_l);
  double dp_m_l = __dmul_rd(cos_theta, p_m_l_0);
  p_m_l_1 = __dmul_rd(p_m_l_1, sin_theta_l);
  // Compute normalized dp dtheta
  dp_m_l *= (double) degree;
  reg1 = __dmul_rd((double)order - degree,p_m_l_1);
  reg2 = __dsqrt_rd(__ddiv_rd((double) order+degree, (double) degree-order));
  double reg3 = -sin(theta);
  dp_m_l = __dadd_rd(dp_m_l, __dmul_rd(reg2, reg1));
  dp_m_l = __ddiv_rd(dp_m_l, reg3);
#ifdef CUDA_DEBUG
  if(blockIdx.x ==0 && threadIdx.x ==0 && threadIdx.y==0) {
  int j = (degree-1)*(degree) + order;  
  int idx_debug2 = blockIdx.x*constants.nidx_rlm[1] + j;
  debug_P_smdt[idx_debug2] = p_m_l_0; 
  debug_dP_smdt[idx_debug2] = dp_m_l; 
  idx_debug2 = blockIdx.x*constants.nidx_rlm[1] + (degree-1)*(degree) - order;
  debug_P_smdt[idx_debug2] = p_m_l_0; 
  debug_dP_smdt[idx_debug2] = dp_m_l;} 
#endif     

  degree--;
  //dp_m_l *= 0.5;
  double a_r_1d = a_r_1d_rlm_r[threadIdx.x];
  int idx, idx_rtm_mp, idx_rtm_mn;
  int x,y;
  x = threadIdx.x * constants.istep_rlm[0] * constants.ncomp;
  y = (threadIdx.y+1)*3;
  double a_r_1d_sq = a_r_1d * a_r_1d;
  double asin_theta = __dmul_rd(asin_theta_1d_rtm[blockIdx.x], (double) order);
  double vr1=0, vr2=0, vr3=0, vr4=0, vr5=0;
  double dPdt;
  for(int j_rlm=jst; j_rlm<=jed; j_rlm++, degree++) {
    idx = constants.ncomp * (j_rlm-1) * constants.istep_rlm[1] + x + y; 
    reg2 = -1 * __dmul_rd(asin_theta,p_m_l_0);
    dPdt = __dmul_rd(a_r_1d, dp_m_l);
    reg1 = __dmul_rd(a_r_1d, reg2);
     
    vr5 += sp_rlm[idx - 1] * reg1;
    vr4 += sp_rlm[idx - 2] * reg1; 
    vr3 += sp_rlm[idx - 3] * a_r_1d_sq * p_m_l_0 * g_sph_rlm[j_rlm-1];    
    vr2 += sp_rlm[idx - 2] * dPdt;
    vr1 -= sp_rlm[idx - 1] * dPdt;    
    // m, l+2
    reg1 = __ddiv_rd((double) degree+2-order, (double) degree+2+order);
    reg2 = __dmul_rd(cos_theta, p_m_l_1);
    reg1 = __dsqrt_rd(reg1);
    reg2 = __dmul_rd((double) 2*degree + 3, reg2);
    reg3 = (double) (degree+2-order) * (degree+1-order);
    //One coefficient
    reg1 = __dmul_rd(reg1, reg2);
    reg2 = __dmul_rd((double) order+degree+1, p_m_l_0);
    // dp_m_l is a misnomer here
    dp_m_l = (double) (degree+2+order) * (degree+order+1); 
    p_m_l_0 = p_m_l_1;
    p_m_l_1 = __dsqrt_rd(__ddiv_rd(reg3, dp_m_l));
    p_m_l_1 = __dmul_rd(p_m_l_1, reg2);
    reg3 = (double) degree-order+2;
    // p_m_l_0, m, l+2
    p_m_l_1 = __ddiv_rd(__dadd_rd(reg1,-1*p_m_l_1), reg3);
    //dp_m_l 
    dp_m_l = __dmul_rd(cos_theta, p_m_l_0);
    dp_m_l *= (double) degree+2;
    reg1 = __dmul_rd((double)order - degree - 2,p_m_l_1);
    reg2 = __dsqrt_rd(__ddiv_rd((double) degree+order+2, (double) degree-order+2));
    reg3 = -sin(theta);
    dp_m_l = __dadd_rd(dp_m_l, __dmul_rd(reg2, reg1));
    dp_m_l = __ddiv_rd(dp_m_l, reg3);
#ifdef CUDA_DEBUG
  if(blockIdx.x ==0 && threadIdx.x ==0 && threadIdx.y==0) {
  int idx_debug = blockIdx.x*constants.nidx_rlm[1] + (degree+1)*(degree+2) + order;
  debug_P_smdt[idx_debug] = p_m_l_0; 
  debug_dP_smdt[idx_debug] = dp_m_l; 
  idx_debug = blockIdx.x*constants.nidx_rlm[1] + (degree+1)*(degree+2) - order;
  debug_P_smdt[idx_debug] = p_m_l_0; 
  debug_dP_smdt[idx_debug] = dp_m_l; }
#endif
  }
  // mp_rlm 
  reg1 = (blockIdx.x) * constants.istep_rtm[1] + threadIdx.x*constants.istep_rtm[0];
  idx_rtm_mp = constants.ncomp * (reg1 + (mp_rlm-1) * constants.istep_rtm[2]) + y; 
  // mn_rlm
  idx_rtm_mn = constants.ncomp * (reg1 + (mn_rlm-1) * constants.istep_rtm[2]) + y; 
  vr_rtm[idx_rtm_mp - 2 - 1]  += vr3; 
  vr_rtm[idx_rtm_mp - 1 - 1]  += vr2; 
  vr_rtm[idx_rtm_mp - 1]  += vr1; 
  vr_rtm[idx_rtm_mn - 1 - 1] += vr5; 
  vr_rtm[idx_rtm_mn - 1] += vr4; 
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

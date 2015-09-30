#include <cuda_runtime.h>
#include "legendre_poly.h"
#include "math_functions.h"
#include "math_constants.h"
#include <sstream>

/*
 *   Set of variables that take advantage of constant memory.
 *     Access to constant memory is faster than access to global memory.
 *       */
__constant__ Geometry_c devConstants;

void initDevConstVariables() {
  cudaError_t error;
  error = cudaMemcpyToSymbol(devConstants, &constants, sizeof(Geometry_c), 0, cudaMemcpyHostToDevice);
  cudaErrorCheck(error);
}

__device__ __inline__
double nextLGP_m_eq0(int l, double x, double p_0, double p_1) {
// x = cos(theta)
  return __dadd_rd(__dmul_rd(__dmul_rd(p_1, __ddiv_rd(2*l-1, l)),x), __dmul_rd(p_0, __ddiv_rd(l-1, l))*-1); 
}

__device__ __inline__  
double nextDp_m_eq_0(int l, double lgp_mp) {
   return __dmul_rd(-1*lgp_mp, __dsqrt_rd(l*(l+1)/2)); 
}

__device__ __inline__  
double nextDp_m_eq_1(int l, double lgp_mn, double lgp_mp) {
  return __dmul_rd(0.5, __dadd_rd(__dmul_rd(__dsqrt_rd(2*l*(l+1)), lgp_mn),-1 * __dmul_rd(__dsqrt_rd((l-1)*(l+2)), lgp_mp) ));
}

__device__ __inline__  
double nextDp_m_l(int m, int l, double lgp_mn, double lgp_mp) {
  return __dmul_rd(0.5, __dadd_rd(__dmul_rd(__dsqrt_rd((l+m)*(l-m+1)), lgp_mn), -1*__dmul_rd(__dsqrt_rd((l-m)*(l+m+1)), lgp_mp)));
}

__device__ __inline__  
double calculateLGP_m_eq_l(int mode) {
  double lgp=1;
  for(int k=1; k<=abs(mode); k++) {
    lgp *= __ddiv_ru((double)2*k-1, (double)2*k);
  }
  
  return __dsqrt_rd(__dmul_rd(2, lgp));
}

__device__ __inline__  
double calculateLGP_mp1_eq_l(int m, double x, double lgp_m_eq_l) {
  int mode = abs(m);
  return __dmul_rd(__dmul_rd(lgp_m_eq_l, __dsqrt_rd(2*mode+1)),x); 
}

__device__ __inline__  
double calculateLGP_m_l(int mode, int degree, double theta, double lgp_0, double lgp_1) {
  int m = abs(mode);
  return  __ddiv_rd(__dadd_rd(__dmul_rd(2*degree-1, __dmul_rd(cos(theta), lgp_1)), __dmul_rd(-1 * lgp_0, __dsqrt_rd((degree-1)*(degree-1) - m*m))), __dsqrt_rd((degree*degree) - (m*m)));
}

__device__ __inline__  
double calculateLGP_m_l_mod(int mode, int degree, double cos_theta, double lgp_0, double lgp_1) {
  return  __ddiv_rd(__dadd_rd(__dmul_rd((double)2*degree-1, __dmul_rd(cos_theta, lgp_1)), -1 * __dmul_rd(lgp_0, __dsqrt_rd((double)(degree-1)*(degree-1) - mode*mode))), __dsqrt_rd((double) degree*degree - mode*mode));
}

__device__ __inline__  
double scaleBySine(int m, double lgp, double theta) {
  double reg1 = sin(theta);
  m = abs(m);
  for(int r=0; r<m; r++)
    lgp = __dmul_rd(lgp,reg1); 
  return lgp;
}

__global__ void set_leg_poly_m_ep_l(double *leg_poly_m_eq_l) { 
  //dim3 block(threads)
  // threads loop over order, nidx_rtm[2] 
  double lgp=1;
  double upload_val=0;  
  int remaining_polys = devConstants.t_lvl % blockDim.x;
  int cycle_counter=0;
  for(int k=1; k<=abs(devConstants.t_lvl); k++) {
    lgp *= __ddiv_ru((double)2*k-1, (double)2*k);
    if((k-1)%blockDim.x == threadIdx.x) {
      upload_val = lgp;
      
    }
    if(k % blockDim.x == 0) {
      leg_poly_m_eq_l[threadIdx.x + cycle_counter*blockDim.x] = __dsqrt_rd(__dmul_rd(2,upload_val)); 
      cycle_counter++;
    }
    else if(k==devConstants.t_lvl && threadIdx.x < remaining_polys) 
      leg_poly_m_eq_l[threadIdx.x + cycle_counter*blockDim.x] = __dsqrt_rd(__dmul_rd(2,upload_val)); 
  }
}

//
//excluding transformation of harmonics 0,0 and 1,1
#ifdef CUDA_DEBUG
__global__
void transB_m_l_neo(int const* __restrict__ lstack_rlm, int const* __restrict__ idx_gl_1d_rlm_j, double const* __restrict__ g_colat_rtm, double const* __restrict__ g_sph_rlm, double const* __restrict__ asin_theta_1d_rtm, double const* __restrict__ a_r_1d_rlm_r, double const* __restrict__ sp_rlm, double *vr_rtm, double *debug_P_smdt, double *debug_dP_smdt) {
#else
__global__
void transB_m_l_neo(int const* __restrict__ lstack_rlm, int const* __restrict__ idx_gl_1d_rlm_j, double const* __restrict__ g_colat_rtm, double const* __restrict__ g_sph_rlm, double const* __restrict__ asin_theta_1d_rtm, double const* __restrict__ a_r_1d_rlm_r, double const* __restrict__ sp_rlm, double *vr_rtm) {
#endif 
  //dim3 grid(nTheta, nModes)
  //dim3 block(x, nshells, nvectors) 
  int m, l;
  int order, degree;
  int jst = 1;
  int jed;
  int mp_rlm = blockIdx.y+1;
  int mn_rlm = devConstants.nidx_rtm[2] - mp_rlm + 1;
  jst += lstack_rlm[blockIdx.y];
  l = devConstants.nidx_rlm[1] * 2 - 1; 
  m = devConstants.nidx_rlm[1] - 1; 
  jed = lstack_rlm[blockIdx.y+1];
  m += jst;
  l += jst;
//In fortran, column-major:
//idx_gl_1d_rlm_j(j,1): global ID for spherical harmonics
//idx_gl_1d_rlm_j(j,2): spherical hermonincs degree
//idx_gl_1d_rlm_j(j,3): spherical hermonincs order

  order = abs(idx_gl_1d_rlm_j[m]); 
  degree = idx_gl_1d_rlm_j[l]; 
  
  if(degree == 0 || degree == 1 || degree == -1) 
    return;
  
  double theta = g_colat_rtm[blockIdx.x]; 
  
  // m,l
  double lgp=1;
  double sin_theta_l=1; 
  for(int k=1; k<=order; k++) {
    lgp *= __ddiv_ru((double)2*k-1, (double)2*k);
  }
  double cos_theta = cos(theta);
  double sin_theta = sin(theta);
  for(int k=0; k<degree; k++)
    sin_theta_l *= sin_theta;
  double reg1 = __dmul_rd((double)2, lgp)
  double p_m_l_0 = __dsqrt_rd(reg1) * sin_theta_l;
  double reg2 = __dmul_rd((double) (2*order + 1), cos_theta); 
  // m,l+1
  degree++;
  double p_m_l_1 = __dsqrt_rd(__dmul_rd(reg1, reg2));
//  double dp_m_l = __dmul_rd(sin_theta_l, p_mn_l_1);
//  dp_m_l *= __dsqrt_rd(2*order);
#ifdef CUDA_DEBUG
  int j = (degree-1)*(degree) + order;  
  int idx_debug = blockIdx.x*devConstants.nidx_rlm[1] + j;
  debug_P_smdt[idx_debug] = p_m_l_0; 
#endif     

  //dp_m_l *= 0.5;
  double a_r_1d = a_r_1d_rlm_r[threadIdx.x];
  double reg2, reg3; 
  int idx, idx_rtm_mp, idx_rtm_mn;
  int x,y;
  x = threadIdx.x * devConstants.istep_rlm[0] * devConstants.ncomp;
  y = (threadIdx.y+1)*3;
  double a_r_1d_sq = a_r_1d * a_r_1d;
  double asin_theta = __dmul_rd(asin_theta_1d_rtm[blockIdx.x], (double) order);
  double vr1=0, vr2=0, vr3=0, vr4=0, vr5=0;
  double dPdt;
  for(int j_rlm=jst; j_rlm<=jed; j_rlm++, degree++) {
    idx = devConstants.ncomp * (j_rlm-1) * devConstants.istep_rlm[1] + x + y; 
    reg2 = -1 * __dmul_rd(asin_theta,p_m_l_0);
    //dPdt = __dmul_rd(a_r_1d, dp_m_l);
    reg1 = __dmul_rd(a_r_1d, reg2);
     
    vr5 += sp_rlm[idx - 1] * reg1;
    vr4 += sp_rlm[idx - 2] * reg1; 
    vr3 += sp_rlm[idx - 3] * a_r_1d_sq * p_m_l_0 * g_sph_rlm[j_rlm-1];    
   // vr2 += sp_rlm[idx - 2]  * dPdt;
   // vr1 -= sp_rlm[idx - 1] * dPdt;    
      
    // m-1, l+1 
    //reg1 = calculateLGP_m_l_mod(order-1, degree+1, cos_theta, p_mn_l_0, p_mn_l_1); 
    //p_mn_l_0 = p_mn_l_1;
    //p_mn_l_1 = reg1;

    // m, l+2
    reg2 = calculateLGP_m_l_mod(order, degree+2, cos_theta, p_m_l_0, p_m_l_1);
    p_m_l_0 = p_m_l_1;
    sin_theta_l *= sin_theta;
    p_m_l_0 *= sin_theta_l;
    // p_m_l_0, m, l+1
    p_m_l_1 = reg2;
 
#ifdef CUDA_DEBUG
  idx_debug = blockIdx.x*devConstants.nidx_rlm[1] + (degree+1)*(degree+2) + order;
  debug_P_smdt[idx_debug] = p_m_l_0; 
#endif
    //m, l+1
//    sin_theta_l *= sin_theta; 
//    reg1 = sin_theta_l * p_mn_l_1;
//    reg2 = sin_theta_l * p_mp_l_0; 
//    dp_m_l = nextDp_m_l(order, degree+1, reg1, reg2);
     
    //m+1, l+3
//    reg3 = calculateLGP_m_l_mod(order+1, degree+3, cos_theta, p_mp_l_0, p_mp_l_1);  
//    p_mp_l_0 = p_mp_l_1;
    // p_mp_1_0, m+1, l+2
//    p_mp_l_1 = reg3;
        
  }
  // mp_rlm 
  reg1 = (blockIdx.x) * devConstants.istep_rtm[1] + threadIdx.x*devConstants.istep_rtm[0];
  idx_rtm_mp = devConstants.ncomp * (reg1 + (mp_rlm-1) * devConstants.istep_rtm[2]) + y; 
  // mn_rlm
  idx_rtm_mn = devConstants.ncomp * (reg1 + (mn_rlm-1) * devConstants.istep_rtm[2]) + y; 
  vr_rtm[idx_rtm_mp - 2 - 1]  += vr3; 
  vr_rtm[idx_rtm_mp - 1 - 1]  += vr2; 
  vr_rtm[idx_rtm_mp - 1]  += vr1; 
  vr_rtm[idx_rtm_mn - 1 - 1] += vr5; 
  vr_rtm[idx_rtm_mn - 1] += vr4; 
}

//excluding transformation of harmonics 0,0 and 1,1
__global__
void transB_m_l_ver3D_block_of_vectors_smem(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_smdt, double *dP_smdt, double *g_sph_rlm, double *asin_theta_1d_rtm) {
  extern __shared__ double cache[];
  //dim3 grid(nTheta, constants.nidx_rtm[2]);
  //dim3 block(nShells,constants.nvector,1);

  int deg, j;

  // P(m,m)[cos theta]
  double p_mn_l_0=0, p_mn_l_1=0;
  double p_m_l_0=0, p_m_l_1=0;
  double p_mp_l_0=0, p_mp_l_1=0;
  double dp_m_l=0;
 
  double x=1, theta=0;
// 3 for m-1, m, m+1
  unsigned int idx[3] = {0,0,0}, idx_rtm[3] = {0,0,0};
  double reg1, reg2, reg3;
 
  double norm=0;
 
  double vr1=0, vr2=0, vr3=0, vr4=0, vr5=0; 

  int mp_rlm = blockIdx.y+1;
  double mn_rlm = devConstants.nidx_rtm[2] - mp_rlm + 1;
  int jst = lstack_rlm[mp_rlm-1] + 1;
  int jed = lstack_rlm[mp_rlm];
  int order = abs(idx_gl_1d_rlm_j[devConstants.nidx_rlm[1]*2 + jst -1]); 
  int degree = idx_gl_1d_rlm_j[devConstants.nidx_rlm[1]*1 + jst - 1]; 

  if(degree == 0 || degree == 1)
    return;

  theta = g_colat_rtm[blockIdx.x];
  x = cos(theta);
  deg = degree;

  // m-1,l-1
  p_mn_l_0 = calculateLGP_m_eq_l(order-1);
  // m-1,l
  p_mn_l_1 = calculateLGP_mp1_eq_l(order-1, x, p_mn_l_0);

  // m,l
  p_m_l_0 = calculateLGP_m_eq_l(order); 
  // m,l+1
  p_m_l_1 = calculateLGP_mp1_eq_l(order, x, p_m_l_0);

  // m+1,l+1 
  p_mp_l_0 = calculateLGP_m_eq_l(order+1);
  // m+1,l+2
  p_mp_l_1 = calculateLGP_mp1_eq_l(order+1, x, p_mp_l_0);

  // m,l
  dp_m_l = __dmul_rd(0.5, __dmul_rd(__dsqrt_rd(2*order), scaleBySine(order-1, p_mn_l_1, theta)));

  cache[threadIdx.x] = a_r_1d_rlm_r[threadIdx.x];

  __syncthreads();

    for(int j_rlm=jst, deg=degree; j_rlm<=jed; j_rlm++, deg++) {
        idx[1] = devConstants.ncomp * ((j_rlm-1) * devConstants.istep_rlm[1] + threadIdx.x * devConstants.istep_rlm[0]) + (threadIdx.y+1)*3; 
        norm = scaleBySine(order, p_m_l_0, theta);
        reg2 = __dmul_rd(__dmul_rd(-1 * norm, (double) order), asin_theta_1d_rtm[blockIdx.x]);         
        vr5 += sp_rlm[idx[1] - 1] * cache[threadIdx.x] * reg2;
        vr4 += sp_rlm[idx[1] - 2] * cache[threadIdx.x] * reg2;
       //vr_reg[t*3 - 3] += sp_rlm[idx[1] - 3] * __dmul_rd(a_r_1d_rlm_r[blockIdx.x], a_r_1d_rlm_r[blockIdx.x]) * scaleBySine(order, p_m_l_0, theta) * g_sph_rlm[j_rlm-1];    
        vr3 += sp_rlm[idx[1] - 3] * __dmul_rd(cache[threadIdx.x], cache[threadIdx.x]) * norm * g_sph_rlm[j_rlm-1];    
        vr2 += sp_rlm[idx[1] - 2]  * cache[threadIdx.x] * dp_m_l;    
        vr1 -= sp_rlm[idx[1] - 1] * cache[threadIdx.x] * dp_m_l;    
      
      // m-1, l+1 
      reg1 = calculateLGP_m_l(abs(order)-1, deg+1, theta, p_mn_l_0, p_mn_l_1); 
      p_mn_l_0 = p_mn_l_1;
      p_mn_l_1 = reg1;

      // m, l+2
      reg2 = calculateLGP_m_l(order, deg+2, theta, p_m_l_0, p_m_l_1);
      p_m_l_0 = p_m_l_1;
      // p_m_l_0, m, l+1
/*    Illegal address exception thrown 
      #ifdef CUDA_DEBUG
        if(deg<=devConstants.t_lvl) 
          j = (deg+1)*(deg+2) + order;
          P_smdt[blockIdx.x*devConstants.nidx_rlm[1] + j] = scaleBySine(order, p_m_l_0, theta);
      #endif
*/
      p_m_l_1 = reg2;
 
      //m, l+1
      dp_m_l = nextDp_m_l(order, deg+1, scaleBySine(abs(order)-1, p_mn_l_1, theta), scaleBySine(abs(order)+1, p_mp_l_0, theta));
     
/*    Illegal address exception thrown 
      #ifdef CUDA_DEBUG
        if(deg<=devConstants.t_lvl)
          j = (deg+1)*(deg+2) + order;
          dP_smdt[blockIdx.x*devConstants.nidx_rlm[1] + j] = dp_m_l; 
      #endif
*/
      //m+1, l+3
      reg3 = calculateLGP_m_l(abs(order)+1, deg+3, theta, p_mp_l_0, p_mp_l_1);  
      p_mp_l_0 = p_mp_l_1;
      // p_mp_1_0, m+1, l+2
      p_mp_l_1 = reg3;
        
    }
    // mp_rlm 
    reg1 = (blockIdx.x) * devConstants.istep_rtm[1] + threadIdx.x*devConstants.istep_rtm[0];
    idx_rtm[0] = devConstants.ncomp * (reg1 + (mp_rlm-1) * devConstants.istep_rtm[2]) + (threadIdx.y + 1)*3; 
    // mn_rlm
    idx_rtm[1] = devConstants.ncomp * (reg1 + (mn_rlm-1) * devConstants.istep_rtm[2]) + (threadIdx.y + 1)*3; 
    vr_rtm[idx_rtm[0] - 2 - 1]  += vr3; 
    vr_rtm[idx_rtm[0] - 1 - 1]  += vr2; 
    vr_rtm[idx_rtm[0] - 1]  += vr1; 
    vr_rtm[idx_rtm[1] - 1 - 1] += vr5; 
    vr_rtm[idx_rtm[1] - 1] += vr4; 
}
//excluding transformation of harmonics 0,0 and 1,1
__global__
void transB_m_l_ver3D_block_of_vectors(int *lstack_rlm, int m0, int m1, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_smdt, double *dP_smdt, double *g_sph_rlm, double *asin_theta_1d_rtm) {
  int deg, j;

  // P(m,m)[cos theta]
  double p_mn_l_0=0, p_mn_l_1=0;
  double p_m_l_0=0, p_m_l_1=0;
  double p_mp_l_0=0, p_mp_l_1=0;
  double dp_m_l=0;
 
  double x=1, theta=0;
// 3 for m-1, m, m+1
  unsigned int idx[3] = {0,0,0}, idx_rtm[3] = {0,0,0};
  double reg1, reg2, reg3;
 
  double norm=0;
 
  double vr1=0, vr2=0, vr3=0, vr4=0, vr5=0; 

  int mp_rlm = blockIdx.y+m0;
  double mn_rlm = devConstants.nidx_rtm[2] - mp_rlm + 1;
  int jst = lstack_rlm[mp_rlm-1] + 1;
  int jed = lstack_rlm[mp_rlm];
  int order = abs(idx_gl_1d_rlm_j[devConstants.nidx_rlm[1]*2 + jst -1]); 
  int degree = idx_gl_1d_rlm_j[devConstants.nidx_rlm[1]*1 + jst - 1]; 

  theta = g_colat_rtm[blockIdx.x];
  x = cos(theta);
  deg = degree;

  // m-1,l-1
  p_mn_l_0 = calculateLGP_m_eq_l(order-1);
  // m-1,l
  p_mn_l_1 = calculateLGP_mp1_eq_l(order-1, x, p_mn_l_0);

  // m,l
  p_m_l_0 = calculateLGP_m_eq_l(order); 
  // m,l+1
  p_m_l_1 = calculateLGP_mp1_eq_l(order, x, p_m_l_0);

  // m+1,l+1 
  p_mp_l_0 = calculateLGP_m_eq_l(order+1);
  // m+1,l+2
  p_mp_l_1 = calculateLGP_mp1_eq_l(order+1, x, p_mp_l_0);

  // m,l
  dp_m_l = __dmul_rd(0.5, __dmul_rd(__dsqrt_rd(2*order), scaleBySine(order-1, p_mn_l_1, theta)));

    for(int j_rlm=jst, deg=degree; j_rlm<=jed; j_rlm++, deg++) {
        idx[1] = devConstants.ncomp * ((j_rlm-1) * devConstants.istep_rlm[1] + threadIdx.x * devConstants.istep_rlm[0]) + (threadIdx.y+1)*3; 
        norm = scaleBySine(order, p_m_l_0, theta);
        reg2 = __dmul_rd(__dmul_rd(-1 * norm, (double) order), asin_theta_1d_rtm[blockIdx.x]);         
        vr5 += sp_rlm[idx[1] - 1] * a_r_1d_rlm_r[threadIdx.x] * reg2;
        vr4 += sp_rlm[idx[1] - 2] * a_r_1d_rlm_r[threadIdx.x] * reg2;
       //vr_reg[t*3 - 3] += sp_rlm[idx[1] - 3] * __dmul_rd(a_r_1d_rlm_r[blockIdx.x], a_r_1d_rlm_r[blockIdx.x]) * scaleBySine(order, p_m_l_0, theta) * g_sph_rlm[j_rlm-1];    
        vr3 += sp_rlm[idx[1] - 3] * __dmul_rd(a_r_1d_rlm_r[threadIdx.x], a_r_1d_rlm_r[threadIdx.x]) * norm * g_sph_rlm[j_rlm-1];    
        vr2 += sp_rlm[idx[1] - 2]  * a_r_1d_rlm_r[threadIdx.x] * dp_m_l;    
        vr1 -= sp_rlm[idx[1] - 1] * a_r_1d_rlm_r[threadIdx.x] * dp_m_l;    
      
      // m-1, l+1 
      reg1 = calculateLGP_m_l(abs(order)-1, deg+1, theta, p_mn_l_0, p_mn_l_1); 
      p_mn_l_0 = p_mn_l_1;
      p_mn_l_1 = reg1;

      // m, l+2
      reg2 = calculateLGP_m_l(order, deg+2, theta, p_m_l_0, p_m_l_1);
      p_m_l_0 = p_m_l_1;
      // p_m_l_0, m, l+1
/*    Illegal address exception thrown 
      #ifdef CUDA_DEBUG
        if(deg<=devConstants.t_lvl) 
          j = (deg+1)*(deg+2) + order;
          P_smdt[blockIdx.x*devConstants.nidx_rlm[1] + j] = scaleBySine(order, p_m_l_0, theta);
      #endif
*/
      p_m_l_1 = reg2;
 
      //m, l+1
      dp_m_l = nextDp_m_l(order, deg+1, scaleBySine(abs(order)-1, p_mn_l_1, theta), scaleBySine(abs(order)+1, p_mp_l_0, theta));
     
/*    Illegal address exception thrown 
      #ifdef CUDA_DEBUG
        if(deg<=devConstants.t_lvl)
          j = (deg+1)*(deg+2) + order;
          dP_smdt[blockIdx.x*devConstants.nidx_rlm[1] + j] = dp_m_l; 
      #endif
*/
      //m+1, l+3
      reg3 = calculateLGP_m_l(abs(order)+1, deg+3, theta, p_mp_l_0, p_mp_l_1);  
      p_mp_l_0 = p_mp_l_1;
      // p_mp_1_0, m+1, l+2
      p_mp_l_1 = reg3;
        
    }
    // mp_rlm 
    reg1 = (blockIdx.x) * devConstants.istep_rtm[1] + threadIdx.x*devConstants.istep_rtm[0];
    idx_rtm[0] = devConstants.ncomp * (reg1 + (mp_rlm-1) * devConstants.istep_rtm[2]) + (threadIdx.y + 1)*3; 
    // mn_rlm
    idx_rtm[1] = devConstants.ncomp * (reg1 + (mn_rlm-1) * devConstants.istep_rtm[2]) + (threadIdx.y + 1)*3; 
    vr_rtm[idx_rtm[0] - 2 - 1]  += vr3; 
    vr_rtm[idx_rtm[0] - 1 - 1]  += vr2; 
    vr_rtm[idx_rtm[0] - 1]  += vr1; 
    vr_rtm[idx_rtm[1] - 1 - 1] += vr5; 
    vr_rtm[idx_rtm[1] - 1] += vr4; 
}
//Excluding transform of harmonics (0,0) and (1,1) 
__global__
void transB_scalars_OTF(int *lstack_rlm, int m0, int m1, int *idx_gl_1d_rlm_j, double *vr_rtm, double *sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *g_sph_rlm, double *asin_theta_1d_rtm) {

//        dim3 grid(nTheta, m1-m0+1);
//        dim3 block(nShells,devConstants.nscalar,1);
  int deg;

  // P(m,m)[cos theta]
  double p_m_l_0, p_m_l_1;
 
  double x=1, theta=0;
// 3 for m-1, m, m+1
  unsigned int idx, idx_rtm;
  double reg1, reg2;
 
  double norm=0;
  
  double vr=0;

  int mp_rlm = blockIdx.y+m0;
  int jst = lstack_rlm[mp_rlm-1] + 1;
  int jed = lstack_rlm[mp_rlm];
  int order = abs(idx_gl_1d_rlm_j[devConstants.nidx_rlm[1]*2 + jst -1]); 
  int degree = idx_gl_1d_rlm_j[devConstants.nidx_rlm[1]*1 + jst - 1]; 

  theta = g_colat_rtm[blockIdx.x];
  x = cos(theta);
  deg = degree;

  // m,l
  p_m_l_0 = calculateLGP_m_eq_l(order); 
  // m,l+1
  p_m_l_1 = calculateLGP_mp1_eq_l(order, x, p_m_l_0);

  for(int j_rlm=jst; j_rlm<=jed; j_rlm++, deg++) {
    idx = devConstants.ncomp * ((j_rlm-1) * devConstants.istep_rlm[1] + threadIdx.x * devConstants.istep_rlm[0]) + 3*devConstants.nvector + (threadIdx.y+1); 
    norm = scaleBySine(order, p_m_l_0, theta);
    vr += sp_rlm[idx - 1] * norm;

    // m, l+2
    reg2 = calculateLGP_m_l(order, deg+2, theta, p_m_l_0, p_m_l_1);
    p_m_l_0 = p_m_l_1;
    // p_m_l_0, m, l+1
    p_m_l_1 = reg2;
  }
    // mp_rlm 
    reg1 = (blockIdx.x) * devConstants.istep_rtm[1] + threadIdx.x*devConstants.istep_rtm[0];
    idx_rtm = devConstants.ncomp * (reg1 + (mp_rlm-1) * devConstants.istep_rtm[2]) + 3*devConstants.nvector + threadIdx.x+1; 
    vr_rtm[idx_rtm - 1] = vr;
}

__global__
void transB_scalars_OTF_smem(int *lstack_rlm, int m0, int m1, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *g_sph_rlm, double *asin_theta_1d_rtm) {
  extern __shared__ double cacheSchmidt[];

  int deg;

  double x=1, theta=0;
// 3 for m-1, m, m+1
  unsigned int idx, idx_rtm;
  double reg1;
 
  double norm=0;
  double vr=0;

  int mp_rlm = blockIdx.y+m0;
  int jst = lstack_rlm[mp_rlm-1] + 1;
  int jed = lstack_rlm[mp_rlm];
  int order = abs(idx_gl_1d_rlm_j[devConstants.nidx_rlm[1]*2 + jst -1]); 
  int degree = idx_gl_1d_rlm_j[devConstants.nidx_rlm[1]*1 + jst - 1]; 

  theta = g_colat_rtm[blockIdx.x];
  x = cos(theta);
  deg = degree;
  int i=0;
 
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    reg1 = devConstants.t_lvl-degree+1;
    // m,l
    cacheSchmidt[0] = calculateLGP_m_eq_l(order); 
    // m,l+1
    cacheSchmidt[1] = calculateLGP_mp1_eq_l(order, x, cacheSchmidt[0]);
    for(i=2; i<reg1; i++) {
      // m,l+2
      cacheSchmidt[i] = calculateLGP_m_l(order, deg+2, theta, cacheSchmidt[i-2], cacheSchmidt[i-1]);    }
  }
 
  __syncthreads();

  for(int j_rlm=jst, i=0; j_rlm<=jed; j_rlm++, deg++, i++) {
    idx = devConstants.ncomp * ((j_rlm-1) * devConstants.istep_rlm[1] + threadIdx.x * devConstants.istep_rlm[0]) + 3*devConstants.nvector + (threadIdx.y+1); 
    norm = scaleBySine(order, cacheSchmidt[i], theta);
    vr += sp_rlm[idx - 1] * norm;
  }
    // mp_rlm 
    reg1 = (blockIdx.x) * devConstants.istep_rtm[1] + threadIdx.x*devConstants.istep_rtm[0];
    idx_rtm = devConstants.ncomp * (reg1 + (mp_rlm-1) * devConstants.istep_rtm[2]) + 3*devConstants.nvector + threadIdx.y+1; 
    vr_rtm[idx_rtm - 1] = vr;
}

__global__
void transB_scalar(int *lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *P_jl) {
  //dim3 grid3(nTheta, devConstants.nidx_rtm[2]);
  //dim3 block3(nShells,1,1);
 // mp_rlm is the blockIdx.y 
  double vrs1;

  int jst = lstack_rlm[blockIdx.y] + 1;
  int jed = lstack_rlm[blockIdx.y+1];
  int idx_p_jl=0, idx=0, idx_rtm=0; 
  int reg1 = 3*devConstants.nvector + devConstants.ncomp*threadIdx.x*devConstants.istep_rlm[0];

  for(int t=1; t<=devConstants.nscalar; t++) {
    vrs1 = 0;
    idx_p_jl = devConstants.nidx_rlm[1]*blockIdx.x+jst-1;
    for(int j_rlm=jst; j_rlm<=jed; j_rlm++) {
      idx = reg1 + t + devConstants.ncomp*(j_rlm-1)*devConstants.istep_rlm[1]; 
      vrs1 += sp_rlm[idx - 1] * P_jl[idx_p_jl];
      idx_p_jl++;
    } 
      
    idx_rtm = t + 3*devConstants.nvector + devConstants.ncomp*((blockIdx.x) * devConstants.istep_rtm[1] + threadIdx.x*devConstants.istep_rtm[0] + (blockIdx.y)*devConstants.istep_rtm[2]); 
    vr_rtm[idx_rtm - 1] = vrs1;
  } 
}

__global__
void transB_scalar_smem(int *lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *P_jl, const Geometry_c constants) {
  //dim3 grid3(nTheta, devConstants.nidx_rtm[2]);
  //dim3 block3(nShells,1,1);
 // mp_rlm is the blockIdx.y 
  extern __shared__ double schmidt[];
  double vrs1;

  int jst = __ldg(&lstack_rlm[blockIdx.y]) + 1;
  int jed = __ldg(&lstack_rlm[blockIdx.y+1]);
  int idx_p_jl=0, idx=0, idx_rtm=0; 
  int reg1 = 3*constants.nvector + constants.ncomp*threadIdx.x*constants.istep_rlm[0];
  idx_p_jl = constants.nidx_rlm[1]*blockIdx.x+jst-1 + threadIdx.x;
  int me = threadIdx.x;
  while(me < jed-jst+1) {
    schmidt[me] = P_jl[idx_p_jl];
    me += blockDim.x;
  }
   
  __syncthreads();

  int count=0; 
  for(int t=1; t<=constants.nscalar; t++) {
    vrs1 = 0;
    for(int j_rlm=jst-1,count=0; j_rlm<jed; j_rlm++,count++) {
      idx = reg1 + t + constants.ncomp*j_rlm*constants.istep_rlm[1]; 
      vrs1 += sp_rlm[idx - 1] * schmidt[count];
    } 
      
    idx_rtm = t + 3*constants.nvector + constants.ncomp*((blockIdx.x) * constants.istep_rtm[1] + threadIdx.x*constants.istep_rtm[0] + (blockIdx.y)*constants.istep_rtm[2]); 
    vr_rtm[idx_rtm - 1] = vrs1;
  } 
}

__global__
void transB_scalar_L1_cache(int const* __restrict__ lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double const* __restrict__ P_jl) {
  //dim3 grid3(nTheta, devConstants.nidx_rtm[2]);
  //dim3 block3(nShells,1,1);
 // mp_rlm is the blockIdx.y 
  double vrs1;

  int jst = lstack_rlm[blockIdx.y] + 1;
  int jed = lstack_rlm[blockIdx.y+1];
  int idx_p_jl=0, idx=0, idx_rtm=0; 
  int reg1 = 3*devConstants.nvector + devConstants.ncomp*threadIdx.x*devConstants.istep_rlm[0];
  double leg_pol=0;
  idx_p_jl = devConstants.nidx_rlm[1]*blockIdx.x+jst-1;
  leg_pol = P_jl[idx_p_jl+(threadIdx.x%(jed-jst+1))];
  for(int t=1; t<=devConstants.nscalar; t++) {
    vrs1 = 0;
    idx_p_jl = devConstants.nidx_rlm[1]*blockIdx.x+jst-1;
    for(int j_rlm=jst; j_rlm<=jed; j_rlm++) {
      idx = reg1 + t + devConstants.ncomp*(j_rlm-1)*devConstants.istep_rlm[1]; 
      vrs1 += sp_rlm[idx - 1] * P_jl[idx_p_jl];
      idx_p_jl++;
    } 
      
    idx_rtm = t + 3*devConstants.nvector + devConstants.ncomp*((blockIdx.x) * devConstants.istep_rtm[1] + threadIdx.x*devConstants.istep_rtm[0] + (blockIdx.y)*devConstants.istep_rtm[2]); 
    vr_rtm[idx_rtm - 1] = vrs1;
  } 
}

__global__
void transB_scalar_opt_mem_access(int *lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *P_jl) {
 // mp_rlm is the blockIdx.y 
  //dim3 grid4(nTheta, devConstants.nidx_rtm[2], nShells);
  //dim3 block4(devConstants.nscalar,1,1);
  //looped over shell. More work per thread. Less context switching
  double vrs1;

  int jst = lstack_rlm[blockIdx.y] + 1;
  int jed = lstack_rlm[blockIdx.y+1];
  int idx_p_jl=0, idx=0, idx_rtm=0; 
  int reg1=0;

    reg1 = 3*devConstants.nvector + devConstants.ncomp*blockIdx.z*devConstants.istep_rlm[0];
    vrs1 = 0;
    idx_p_jl = devConstants.nidx_rlm[1]*blockIdx.x+jst-1;
    for(int j_rlm=jst; j_rlm<=jed; j_rlm++) {
      idx = reg1 + threadIdx.x + devConstants.ncomp*(j_rlm-1)*devConstants.istep_rlm[1]; 
      vrs1 += sp_rlm[idx - 1] * P_jl[idx_p_jl];
      idx_p_jl++;
    } 
      
    idx_rtm = threadIdx.x + 3*devConstants.nvector + devConstants.ncomp*((blockIdx.x) * devConstants.istep_rtm[1] + blockIdx.z*devConstants.istep_rtm[0] + (blockIdx.y)*devConstants.istep_rtm[2]); 
    vr_rtm[idx_rtm - 1] = vrs1;
}

__global__
void transB_scalar_block_mp_rlm(int const* __restrict__ lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double const* __restrict__ P_jl) {
  //dim3 grid3(nTheta, nShells);
  //dim3 block(nidx_rtm[2])
  //Thread divergence within a block because jst to jed varies per thread
  double vrs1;
  //each thread is associated to a unique harmonics mode
  int mp_rlm = threadIdx.x;
  int jst = lstack_rlm[mp_rlm] + 1;
  int jed = lstack_rlm[mp_rlm+1];
  int idx_p_jl=0, idx=0, idx_rtm=0; 
  int reg1 = 3*devConstants.nvector + devConstants.ncomp*blockIdx.y*devConstants.istep_rlm[0];

  for(int t=1; t<=devConstants.nscalar; t++) {
    vrs1 = 0;
    idx_p_jl = devConstants.nidx_rlm[1]*blockIdx.x+jst-1;
    for(int j_rlm=jst; j_rlm<=jed; j_rlm++) {
      idx = reg1 + t + devConstants.ncomp*(j_rlm-1)*devConstants.istep_rlm[1]; 
      vrs1 += sp_rlm[idx - 1] * P_jl[idx_p_jl];
      idx_p_jl++;
    } 
      
    idx_rtm = t + 3*devConstants.nvector + devConstants.ncomp*((blockIdx.x) * devConstants.istep_rtm[1] + blockIdx.y*devConstants.istep_rtm[0] + (mp_rlm)*devConstants.istep_rtm[2]); 
    vr_rtm[idx_rtm - 1] = vrs1;
  } 
}

__global__
void transB_scalar_block_mp_rlm_smem(int const* __restrict__ lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *P_jl) {
  extern __shared__ double schmidt[];
  //dim3 grid3(nTheta, nShells);
  //dim3 block(nidx_rtm[2])
  //Thread divergence within a block because jst to jed varies per thread
  double vrs1;
  //each thread is associated to a unique harmonics mode
  int mp_rlm = threadIdx.x;
  int jst = lstack_rlm[mp_rlm] + 1;
  int jed = lstack_rlm[mp_rlm+1];
  int idx_p_jl=0, idx=0, idx_rtm=0; 
  int reg1 = 3*devConstants.nvector + devConstants.ncomp*blockIdx.y*devConstants.istep_rlm[0];
  int count=0;
  int n_modes = devConstants.nidx_rlm[1];
  int me = threadIdx.x + count*blockDim.x;
  idx_p_jl = devConstants.nidx_rlm[1]*blockIdx.x;
  while(me < n_modes) {
    schmidt[me] = P_jl[idx_p_jl + me];
    count++;
    me += blockDim.x;
  }
  
  __syncthreads();

  for(int t=1; t<=devConstants.nscalar; t++) {
    vrs1 = 0;
    for(int j_rlm=jst-1; j_rlm<jed; j_rlm++) {
      idx = reg1 + t + devConstants.ncomp*(j_rlm)*devConstants.istep_rlm[1]; 
      vrs1 += sp_rlm[idx - 1] * schmidt[j_rlm];
    } 
      
    idx_rtm = t + 3*devConstants.nvector + devConstants.ncomp*((blockIdx.x) * devConstants.istep_rtm[1] + blockIdx.y*devConstants.istep_rtm[0] + (mp_rlm)*devConstants.istep_rtm[2]); 
    vr_rtm[idx_rtm - 1] = vrs1;
  } 
}

__global__
void transB_dydt(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl, double *dP_jl, const Geometry_c constants) {
  int mp_rlm = blockIdx.y;
  int jst = lstack_rlm[mp_rlm] + 1;
  int jed = lstack_rlm[mp_rlm+1];

  double vr1, vr2, vr3;
  unsigned int idx_p_jl=0, idx=0, idx_rtm=0; 
  int deg = idx_gl_1d_rlm_j[constants.nidx_rlm[1] + jst -1];
  int ord = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + jst -1];
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

__global__
void transB_dydt_smem_a_r(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double const* __restrict__ P_jl, double *dP_jl, const Geometry_c constants) {

  //dim3 grid3(nTheta, constants.nidx_rtm[2]);
  //dim3 block3(nShells,1,1);

  extern __shared__ double cache[];
  int mp_rlm = blockIdx.y;
  int jst = lstack_rlm[mp_rlm] + 1;
  int jed = lstack_rlm[mp_rlm+1];

  double vr1, vr2, vr3;
  unsigned int idx_p_jl=0, idx=0, idx_rtm=0; 
  int deg = idx_gl_1d_rlm_j[constants.nidx_rlm[1] + jst -1];
  int ord = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + jst -1];
  float g_sph_rlm=deg*(deg+1);
  if (ord==0 && deg==0)
    g_sph_rlm=0.5;
  
  cache[threadIdx.x] = a_r_1d_rlm_r[threadIdx.x];
  double reg1 = __dmul_rd(cache[threadIdx.x], cache[threadIdx.x]) * g_sph_rlm; 

  for(int t=1; t<=constants.nvector; t++) {
    vr1=vr2=vr3=0;
    idx_p_jl = constants.nidx_rlm[1]*blockIdx.x+jst-1;
    for(int j_rlm=jst; j_rlm<=jed; j_rlm++) {
      idx = 3*t + constants.ncomp * ((j_rlm-1) * constants.istep_rlm[1] + threadIdx.x * constants.istep_rlm[0]); 
      vr3 += sp_rlm[idx - 3] * reg1 * P_jl[idx_p_jl];    
      vr2 += sp_rlm[idx - 2]  * cache[threadIdx.x] * dP_jl[idx_p_jl];    
      vr1 -= sp_rlm[idx - 1] * cache[threadIdx.x] * dP_jl[idx_p_jl];    
      idx_p_jl++;
    }
    idx_rtm = 3*t + constants.ncomp * ((blockIdx.x) * constants.istep_rtm[1] + threadIdx.x*constants.istep_rtm[0] + (mp_rlm) * constants.istep_rtm[2]); 

    vr_rtm[idx_rtm - 2 - 1]  += vr3; 
    vr_rtm[idx_rtm - 1 - 1]  += vr2; 
    vr_rtm[idx_rtm - 1]  += vr1; 
  }
}

__global__
void transB_dydt_smem_dpschmidt(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double const* __restrict__ P_jl, double const* __restrict__ dP_jl, const Geometry_c constants) {
  //dim3 grid3(nTheta, constants.nidx_rtm[2]);
  //dim3 block3(nShells,1,1);

  extern __shared__ double dp_schmidt[];
  int mp_rlm = blockIdx.y;
  int jst = lstack_rlm[mp_rlm] + 1;
  int jed = lstack_rlm[mp_rlm+1];

  int count = 0;
  int me = threadIdx.x + count*blockDim.x;
  int n_modes = constants.nidx_rlm[1];
  int idx_p_jl = n_modes*blockIdx.x;
  
  double a_r = a_r_1d_rlm_r[threadIdx.x];

  while(me < n_modes) {
    dp_schmidt[me] = dP_jl[idx_p_jl + me] * a_r;
    count++;
    me += blockDim.x;
  }
  
  double a_r_sq = __dmul_rd(a_r,a_r);

  double vr1, vr2, vr3;
  unsigned int idx=0, idx_rtm=0; 
  int deg = idx_gl_1d_rlm_j[constants.nidx_rlm[1] + jst -1];
  int ord = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + jst -1];
  float g_sph_rlm=deg*(deg+1);
  if (ord==0 && deg==0)
    g_sph_rlm=0.5;

  a_r_sq *= g_sph_rlm;

  __syncthreads();
 
  for(int t=1; t<=constants.nvector; t++) {
    vr1=vr2=vr3=0;
    idx_p_jl = constants.nidx_rlm[1]*blockIdx.x+jst-1;
    for(int j_rlm=jst-1; j_rlm<jed; j_rlm++) {
      idx = 3*t + constants.ncomp * (j_rlm * constants.istep_rlm[1] + threadIdx.x * constants.istep_rlm[0]); 
      vr2 += sp_rlm[idx - 2]  * dp_schmidt[j_rlm];    
      vr3 += sp_rlm[idx - 3] * a_r_sq * P_jl[idx_p_jl];    
      vr1 -= sp_rlm[idx - 1] * dp_schmidt[j_rlm];    
      idx_p_jl++;
    }
    idx_rtm = 3*t + constants.ncomp * ((blockIdx.x) * constants.istep_rtm[1] + threadIdx.x*constants.istep_rtm[0] + (mp_rlm) * constants.istep_rtm[2]); 

    vr_rtm[idx_rtm - 2 - 1]  += vr3; 
    vr_rtm[idx_rtm - 1 - 1]  += vr2; 
    vr_rtm[idx_rtm - 1]  += vr1; 
  }
}

__global__
void transB_dydp_smem_schmidt_more_threads(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl,  double *asin_theta_1d_rtm, const Geometry_c constants) {
  extern __shared__ double schmidt[];
  unsigned int idx=0, idx_rtm=0;
  double reg2;
  double vr4, vr5;

  int mp_rlm = blockIdx.y;
  int mn_rlm = constants.nidx_rtm[2] - mp_rlm;
  int jst = lstack_rlm[mp_rlm] + 1;
  int jed = lstack_rlm[mp_rlm+1];

  int count = 0;
  //int me = threadIdx.x * blockDim.y + threadIdx.y + count*(blockDim.x*blockDim.y);
  int me = threadIdx.x * blockDim.y + threadIdx.y;
  int n_modes = constants.nidx_rlm[1];
  int idx_p_jl = n_modes*blockIdx.x;

  double a_r = a_r_1d_rlm_r[threadIdx.x];
  double asin = asin_theta_1d_rtm[blockIdx.x];

  while(me < n_modes) {
    schmidt[me] = P_jl[idx_p_jl + me] * asin;
    me += blockDim.x*blockDim.y;
  }

  int order = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + jst -1]; 

    vr4=vr5=0;
    for(int j_rlm=jst-1, count=0; j_rlm<jed; j_rlm++, count++) {
      idx = 3*(threadIdx.y+1) + constants.ncomp * (j_rlm * constants.istep_rlm[1] + threadIdx.x * constants.istep_rlm[0]); 
      reg2 = __dmul_rd(-1 * schmidt[count], (double) order);         
      vr5 += sp_rlm[idx - 1] * a_r * reg2;
      vr4 += sp_rlm[idx - 2] * a_r * reg2;
    }
    // mn_rlm
    idx_rtm = 3*(threadIdx.y+1) + constants.ncomp * ((blockIdx.x) * constants.istep_rtm[1] + threadIdx.x*constants.istep_rtm[0] + (mn_rlm-1) * constants.istep_rtm[2]); 

   //vr5 is incorrect
    vr_rtm[idx_rtm - 1 - 1] += vr5; 
    vr_rtm[idx_rtm - 1] += vr4; 
}

__global__
void transB_dydt_smem_schmidt_more_threads(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl, double *dP_jl, const Geometry_c constants) {
  //dim3 grid3(nTheta, constants.nidx_rtm[2]);
  //dim3 block3(nShells,nvector,1);

  extern __shared__ double cache[];
  int mp_rlm = blockIdx.y;
  int jst = lstack_rlm[mp_rlm] + 1;
  int jed = lstack_rlm[mp_rlm+1];

  //int me = threadIdx.x * blockDim.y + threadIdx.y + count*(blockDim.x*blockDim.y);
  int me = threadIdx.x * blockDim.y + threadIdx.y;
  int n_modes = constants.nidx_rlm[1];
  int idx_p_jl = n_modes*blockIdx.x;
  
  double a_r = a_r_1d_rlm_r[threadIdx.x];
  double a_r_sq = __dmul_rd(a_r,a_r);

  while(me < n_modes) {
    cache[me] = P_jl[idx_p_jl + me] * a_r_sq;
    cache[me+n_modes] = dP_jl[idx_p_jl + me] * a_r;
    me += blockDim.x*blockDim.y;
  }
  

  double vr1=0, vr2=0, vr3=0, reg1=0;
  unsigned int idx=0, idx_rtm=0; 
  int deg = idx_gl_1d_rlm_j[constants.nidx_rlm[1] + jst -1];
  int ord = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + jst -1];
  float g_sph_rlm=deg*(deg+1);
  if (ord==0 && deg==0)
    g_sph_rlm=0.5;

  __syncthreads();
 
    vr1=vr2=vr3=0;
    for(int j_rlm=jst-1; j_rlm<jed; j_rlm++) {
      idx = 3*(threadIdx.y+1) + constants.ncomp * (j_rlm * constants.istep_rlm[1] + threadIdx.x * constants.istep_rlm[0]); 
      reg1 = cache[j_rlm+n_modes];
      vr2 += sp_rlm[idx - 2]  * reg1;    
      vr3 += sp_rlm[idx - 3] * g_sph_rlm * cache[j_rlm];    
      vr1 -= sp_rlm[idx - 1] * reg1;    
    }
    idx_rtm = 3*(threadIdx.y+1) + constants.ncomp * ((blockIdx.x) * constants.istep_rtm[1] + threadIdx.x*constants.istep_rtm[0] + (mp_rlm) * constants.istep_rtm[2]); 

    vr_rtm[idx_rtm - 2 - 1]  += vr3; 
    vr_rtm[idx_rtm - 1 - 1]  += vr2; 
    vr_rtm[idx_rtm - 1]  += vr1; 
}

__global__
void transB_dydt_read_only_data(int const* __restrict__ lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double const* __restrict__ P_jl, double const* __restrict__ dP_jl) {
  //dim3 grid5(nTheta, nShells);
  //dim3 block5(constants.nidx_rtm[2],1,1);

  int mp_rlm = threadIdx.x;
  int jst = lstack_rlm[mp_rlm] + 1;
  int jed = lstack_rlm[mp_rlm+1];

  double vr1, vr2, vr3;
  unsigned int idx_p_jl=0, idx=0, idx_rtm=0; 
  int deg = idx_gl_1d_rlm_j[devConstants.nidx_rlm[1] + jst -1];
  int ord = idx_gl_1d_rlm_j[devConstants.nidx_rlm[1]*2 + jst -1];
  float g_sph_rlm=deg*(deg+1);
  if (ord==0 && deg==0)
    g_sph_rlm=0.5;
  
  double a_r_1d_rlm_r_val = a_r_1d_rlm_r[blockIdx.y]; 
  for(int t=1; t<=devConstants.nvector; t++) {
    vr1=vr2=vr3=0;
    idx_p_jl = devConstants.nidx_rlm[1]*blockIdx.x+jst-1;
    for(int j_rlm=jst; j_rlm<=jed; j_rlm++) {
      idx = 3*t + devConstants.ncomp * ((j_rlm-1) * devConstants.istep_rlm[1] + blockIdx.y * devConstants.istep_rlm[0]); 
      vr3 += sp_rlm[idx - 3] * __dmul_rd(a_r_1d_rlm_r_val, a_r_1d_rlm_r_val) * P_jl[idx_p_jl] * g_sph_rlm;    
      vr2 += sp_rlm[idx - 2]  * a_r_1d_rlm_r_val * dP_jl[idx_p_jl];    
      vr1 -= sp_rlm[idx - 1] * a_r_1d_rlm_r_val * dP_jl[idx_p_jl];    
      idx_p_jl++;
    }
    idx_rtm = 3*t + devConstants.ncomp * ((blockIdx.x) * devConstants.istep_rtm[1] + blockIdx.y*devConstants.istep_rtm[0] + (mp_rlm) * devConstants.istep_rtm[2]); 

    vr_rtm[idx_rtm - 2 - 1]  += vr3; 
    vr_rtm[idx_rtm - 1 - 1]  += vr2; 
    vr_rtm[idx_rtm - 1]  += vr1; 
  }
}

//When looking at the transformed field data, the first component is off by a sign, oddly. 
__global__
void transB_dydp(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl,  double *asin_theta_1d_rtm, const Geometry_c constants) {
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

  //initDevConstVariables();

//  dim3 grid5(nTheta, nShells);
//  dim3 block5(constants.nidx_rtm[2],1,1);
  dim3 grid3(nTheta, constants.nidx_rtm[2]);
  dim3 block3(nShells,1,1);
//  transB_dydt<<<grid3, block3, 0, streams[0]>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.dP_jl, constants);
//Fastest
//  transB_dydt_smem_a_r<<<grid3, block3, sizeof(double)*nShells, streams[0]>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.dP_jl, constants);
 // transB_dydt_smem_dpschmidt<<<grid3, block3, sizeof(double)*constants.nidx_rlm[1], streams[0]>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.dP_jl, constants);
  
//  dim3 block8(nShells,constants.nvector,1);
//  transB_dydt_smem_schmidt_more_threads<<<grid3, block8, sizeof(double)*2*constants.nidx_rlm[1], streams[0]>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.dP_jl, constants);
 // transB_dydt_read_only_data<<<grid5, block5, 0, streams[0]>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.dP_jl);
//Fastest
 //transB_dydp<<<grid3, block3, sizeof(double)*nShells, streams[0]>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.asin_theta_1d_rtm, constants);
//  transB_dydp_smem_schmidt_more_threads<<<grid3, block8, sizeof(double)*constants.nidx_rlm[1], streams[0]>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.p_jl, deviceInput.asin_theta_1d_rtm, constants);
// transB_scalar<<<grid3, block3, 0, streams[1]>>> (deviceInput.lstack_rlm, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.p_jl);

//  transB_scalar_L1_cache<<<grid3, block3, 0, streams[1]>>> (deviceInput.lstack_rlm, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.p_jl);

//FASTEST
  transB_scalar_smem<<<grid3, block3, (constants.t_lvl+1)*sizeof(double), streams[1]>>> (deviceInput.lstack_rlm, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.p_jl, constants);

//  dim3 grid4(nTheta, constants.nidx_rtm[2], nShells);
//  dim3 block4(constants.nscalar,1,1);
//  transB_scalar_opt_mem_access<<<grid4, block4, 0, streams[1]>>> (deviceInput.lstack_rlm, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.p_jl);

//  transB_scalar_block_mp_rlm<<<grid5, block5, 0, streams[1]>>> (deviceInput.lstack_rlm, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.p_jl);
//  transB_scalar_block_mp_rlm_smem<<<grid5, block5, constants.nidx_rlm[1]*sizeof(double), streams[1]>>> (deviceInput.lstack_rlm, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.p_jl);

 /* int jst, jed, l;
  int m=0, m0=0, m1=0;

  bool begin_set = false, end_set = false;
*/ 
  dim3 grid(nTheta, constants.nidx_rtm[2]);
  dim3 block(nShells,constants.nvector,1);
//  transB_m_l_ver3D_block_of_vectors_smem<<<grid, block, nShells*sizeof(double), streams[l%2]>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, d_debug.P_smdt, d_debug.dP_smdt, deviceInput.g_sph_rlm, deviceInput.asin_theta_1d_rtm);
#ifndef CUDA_DEBUG
  transB_m_l_neo<<<grid, block, 0, streams[0]>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.g_colat_rtm, deviceInput.g_sph_rlm, deviceInput.asin_theta_1d_rtm, deviceInput.a_r_1d_rlm_r, deviceInput.sp_rlm, deviceInput.vr_rtm);
#else
  transB_m_l_neo<<<grid, block, 0, streams[0]>>> (deviceInput.lstack_rlm, deviceInput.idx_gl_1d_rlm_j, deviceInput.g_colat_rtm, deviceInput.g_sph_rlm, deviceInput.asin_theta_1d_rtm, deviceInput.a_r_1d_rlm_r, deviceInput.sp_rlm, deviceInput.vr_rtm, d_debug.P_smdt, d_debug.dP_smdt);
#endif
  
  /*for(int mp_rlm=1; mp_rlm<=constants.nidx_rtm[2]; mp_rlm++) {
    jst = h_debug.lstack_rlm[mp_rlm-1] + 1;
    jed = h_debug.lstack_rlm[mp_rlm]; 
    m = h_debug.idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + jst - 1]; 
    l = h_debug.idx_gl_1d_rlm_j[constants.nidx_rlm[1]*1 + jst - 1]; 
   
    if(begin_set == false && l != 0 && l != 1) {
      begin_set = true;
      m0=mp_rlm;
    }
    if(begin_set == true && end_set == false && (l == 0 || l == 1 )) {
      end_set = true;
      m1=mp_rlm-1;
    }
    if(begin_set == true && end_set == false && mp_rlm == constants.nidx_rtm[2]) {
      end_set = true;
      m1 = mp_rlm;
    }
    if(begin_set == true && end_set == true) {
        dim3 grid_1(nTheta, m1-m0+1);
        dim3 block_1(nShells,constants.nvector,1);
        //dim3 block(nShells,constants.nscalar,1);
//        transB_m_l_ver3D_block_of_vectors<<<grid_1, block_1, 0, streams[l%2]>>> (deviceInput.lstack_rlm, m0, m1, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, d_debug.P_smdt, d_debug.dP_smdt, deviceInput.g_sph_rlm, deviceInput.asin_theta_1d_rtm);
   //     transB_scalars_OTF_smem<<<grid, block, sizeof(double)*(constants.t_lvl+1), streams[l%2]>>> (deviceInput.lstack_rlm, m0, m1, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, deviceInput.g_sph_rlm, deviceInput.asin_theta_1d_rtm);
    //    transB_scalars_OTF<<<grid, block, 0, streams[l%2]>>> (deviceInput.lstack_rlm, m0, m1, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, deviceInput.g_sph_rlm, deviceInput.asin_theta_1d_rtm);
        
      begin_set = false;
      end_set = false;
    } 
  } */
}

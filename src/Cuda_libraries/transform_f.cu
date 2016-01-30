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

  Timer wallClock;
  double elapsedTime=0;

  cout << "\tCUDA Fwd vector transform Algorithms: \n"; 
  cout << "nVectors: " << constants.nvector << " nShells: " << constants.nidx_rtm[0] << "\n";

  /*for(int i=0; i<2; i++) {
    wallClock.startTimer();
    switch (i) {
    case naive:
      cout << "\t\t Static implementation with a block size of nShells: ";
	  transF_vec<<<constants.nidx_rlm[1], constants.nidx_rtm[0], 0>>> (1, deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.radius_1d_rlm_r, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.mdx_n_rlm_rtm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, deviceInput.p_rtm, deviceInput.dP_rtm, deviceInput.g_sph_rlm_7, deviceInput.asin_theta_1d_rtm, constants);
      break;
    case naive_w_more_threads:
      cout << "\t\t Static implementation with a block size of nVector x nShells: ";
	  transF_vec<<<constants.nidx_rlm[1], block, 0>>> (deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.radius_1d_rlm_r, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.mdx_n_rlm_rtm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, deviceInput.p_rtm, deviceInput.dP_rtm, deviceInput.g_sph_rlm_7, deviceInput.asin_theta_1d_rtm, constants);
      break;
    case reduction:
	  cout << "\t\t Static reduction: ";
	  transF_vec_reduction< 32, 3,
                  cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                      double>
            <<<grid, 32>>> (deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, 
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
  }*/
}

__global__
void transF_vec(int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *asin_theta_1d_rtm, const Geometry_c constants) {
  //dim3 grid(constants.nidx_rlm[1],1,1);
  //dim3 block(constants.nvector, constants.nidx_rtm[0],1,1);
  int k_rtm = threadIdx.y;
  //int j_rlm = blockIdx.x;

// 3 for m-1, m, m+1
  unsigned int ip_rtm, in_rtm;

  double reg0, reg1, reg2, reg3, reg4;
  double sp1, sp2, sp3; 

  int order = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + blockIdx.x];
//  int degree = idx_gl_1d_rlm_j[constants.nidx_rlm[1] + blockIdx.x];
  double r_1d_rlm_r = radius_1d_rlm_r[k_rtm]; 

  int mdx_p = mdx_p_rlm_rtm[blockIdx.x] - 1;
  ip_rtm = k_rtm * constants.istep_rtm[0];
  int mdx_n = mdx_n_rlm_rtm[blockIdx.x] - 1;
  mdx_p *= constants.istep_rtm[2];
  mdx_n *= constants.istep_rtm[2];
  mdx_p += ip_rtm;
  mdx_n += ip_rtm;


  int idx;
  int idx_p_rtm = blockIdx.x*constants.nidx_rtm[0]; 
 
  int stride = constants.ncomp * constants.istep_rtm[1];
  int idx_sp = constants.ncomp * ( blockIdx.x*constants.istep_rlm[1] + k_rtm*constants.istep_rlm[0]); 

  ip_rtm = 3*(threadIdx.x+1) + constants.ncomp * mdx_p;
  in_rtm = 3*(threadIdx.x+1) + constants.ncomp * mdx_n;
  sp1=sp2=sp3=0;
  for(int l_rtm=0; l_rtm<constants.nidx_rtm[0]; l_rtm++) {
    idx = idx_p_rtm + l_rtm; 
    
    reg0 = P_rtm[idx] * asin_theta_1d_rtm[l_rtm] * (double) order;
    reg1 = __dmul_rd(vr_rtm[ip_rtm-2], dP_rtm[idx]);
    reg2 = __dmul_rd(vr_rtm[in_rtm-1], reg0);
    reg3 = __dmul_rd(vr_rtm[in_rtm-2], reg0);
    reg4 = __dmul_rd(vr_rtm[ip_rtm-1], dP_rtm[idx]);
    

     sp1 = fma(vr_rtm[ip_rtm-3], P_rtm[idx], sp1);
     sp2 += __dadd_rd(reg1, reg2); 
     sp3 -= __dadd_rd(reg3, reg4); 
     
     ip_rtm +=  stride; 
     in_rtm +=  stride; 
   }
   idx_sp += 3*(threadIdx.x+1); 

   sp_rlm[idx_sp-3] += __dmul_rd(__dmul_rd(r_1d_rlm_r, r_1d_rlm_r), sp1);
   sp_rlm[idx_sp-2] = fma(r_1d_rlm_r, sp2, sp_rlm[idx_sp-2]);
   sp_rlm[idx_sp-1] = fma(r_1d_rlm_r, sp3, sp_rlm[idx_sp-1]);

}

__device__ __forceinline__ void prefetchL1( const double *data, int offset ){

data += offset;

asm("prefetch.global.L1 [%0];"::"l"(data) );

}

/*"The ld.cs load cached streaming operation allocates global lines with evict-first policy in L1 and L2 to limit cache pollution by temporary streaming data that may be accessed once or twice."

Read more at: http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#ixzz3vprmMjjs 
*/

__device__ __forceinline__ double loadCS( const double *data, int offset ){

double variable;

data += offset;

asm("ld.global.f64 %0, [%1];": "=d"(variable) : "l"(data) );

return variable;
}

// Cache at all levels (L1 & L2)
__device__ __forceinline__ double cacheCA( const double *data, int offset ){

double variable;

data += offset;

asm("ld.global.ca.f64 %0, [%1];": "=d"(variable) : "l"(data) );

return variable;
}

__device__
int findSPHId(int *idx_gl_1d_rlm_j, int nModes, int degree) {
  for(int i=0; i<nModes; i++) {
    if(idx_gl_1d_rlm_j[nModes+i] == degree)
      return idx_gl_1d_rlm_j[i];
  }
  return -1;
}
 
__device__
int idSymmetricMode(int *idx_gl_1d_rlm_j, int nModes, int order, int degree) {
  for(int i=0; i<nModes; i++) {
    if(idx_gl_1d_rlm_j[nModes+i] == degree && idx_gl_1d_rlm_j[nModes*2+i] == -1*order)
      return i; 
  }
  return -1;
}

/*__device__ 
void computeLegPoly(int order, int initialDegree, int degree, double theta, double p1, double p2, double *p, double *dp) {
  double reg1, reg2, reg3;
  double dp_;
  for(int itr=initialDegree; itr<degree; itr++) {
    reg1 = __ddiv_rd((double) itr+2-order, (double) itr+2+order); 
    reg2 = __dmul_rd(cos(theta), p2);
    reg1 = __dsqrt_rd(reg1);
    reg2 = __dmul_rd((double) 2*itr + 3, reg2);
    reg3 = (double) (itr+2-order) * (itr+1-order);
    reg1 = __dmul_rd(reg1, reg2);
    reg2 = __dmul_rd((double) order+itr+1, p1);
    //dp_ is a misnomer here
    dp_ = (double) (itr+2+order) * (itr+order+1);
    p1 = p2;
    p2 = __dsqrt_rd(__ddiv_rd(reg3, dp_));
    p2 = __dmul_rd(p2, reg2);
    reg3 = (double) itr-order+2;
    // p1, m, l+2
    p2 = __ddiv_rd(__dadd_rd(reg1,-1*p2), reg3);
    //dp_
    dp_ = __dmul_rd(cos(theta), p1);
    dp_ *= (double) itr+2;
    reg1 = __dmul_rd((double)order - itr - 2,p2);
    reg2 = __dsqrt_rd(__ddiv_rd((double) itr+order+2, (double) itr-order+2));
    dp_ = __dadd_rd(dp_, __dmul_rd(reg2, reg1));
    reg3 = -sin(theta);
    dp_ = __ddiv_rd(dp_, reg3);
  }

  *p = p1; 
  *dp = dp_; 
}
*/

#define STRIDE 16

//__global__ __launch_bounds__(256, 3)
__global__
void transF_vec(int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants) {
  //dim3 grid(constants.nidx_rlm[1],1,1);
/*TODO:  //dim3 blockNew(nThreads);
  // Mapping from new block pattern to old block pattern should optimize access pattern?? 
  // In the works!!*/
  //dim3 block(nVector, constants.nidx_rtm[0],1,1);

  extern __shared__ double legendre[];

  //For a given order, the sum is over variable theta.
  // What order am I?
  int order = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + blockIdx.x];
  int degree = idx_gl_1d_rlm_j[constants.nidx_rlm[1] + blockIdx.x];
 
  int symModeIdx;
 
  if(order < 0) return;
  if(order==0) symModeIdx=blockIdx.x;
  else
    symModeIdx = idSymmetricMode(idx_gl_1d_rlm_j,constants.nidx_rlm[1],order,degree); 
  //Unique thread id for a 3D block
  int tId = (threadIdx.z * blockDim.x *blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;

  int ip_rtm, in_rtm;

  double reg0, reg1, reg2, reg3;
  //Spectral Data points
  double sp1, sp2, sp3; 

    //Cache modes across thread block.
    //Assuming that the amount of shared memory is equivalent to the number of theta values.
    //TODO: For any given amount of shared memory, the block of threads should simply cache, compute,recache, and so on.
  int chunks = (constants.nidx_rtm[1])/( blockDim.x * blockDim.y * blockDim.z);  
    int rem = (constants.nidx_rtm[1]) % ( blockDim.x * blockDim.y * blockDim.z);

    for(int itr=0; itr < chunks; itr++) {
      legendre[( blockDim.x * blockDim.y * blockDim.z) * itr + tId] = P_rtm[itr*( blockDim.x * blockDim.y * blockDim.z) + tId + blockIdx.x*constants.nidx_rtm[1]]; 
      legendre[( blockDim.x * blockDim.y * blockDim.z) * itr + tId + constants.nidx_rtm[1]] = dP_rtm[itr*( blockDim.x * blockDim.y * blockDim.z) + tId + blockIdx.x*constants.nidx_rtm[1]]; 
    }
    //Cache the leftover theta terms.
    if (tId < rem) {
      legendre[( blockDim.x * blockDim.y * blockDim.z) * chunks + tId] = P_rtm[chunks*( blockDim.z * blockDim.x * blockDim.y ) + tId + blockIdx.x*constants.nidx_rtm[1]]; 
      legendre[( blockDim.x * blockDim.y * blockDim.z) * chunks + tId + constants.nidx_rtm[1]] = dP_rtm[chunks*( blockDim.z * blockDim.x * blockDim.y ) + tId + blockIdx.x*constants.nidx_rtm[1]]; 
    }
    __syncthreads();

/*
  int chunks = (constants.nidx_rtm[1])/( blockDim.x * blockDim.y * blockDim.z);  
  int rem = (constants.nidx_rtm[1]) % ( blockDim.x * blockDim.y * blockDim.z);
 
  int initialDegree=(degree/(STRIDE+2)) * (STRIDE+2);

  //TODO: Implement a smarter search algorithm that takes into account data locality.
  int j1, j2;
  if((initialDegree+1) < degree && degree < (initialDegree+2+STRIDE)) {
    j1 = findSPHId(idx_gl_1d_rlm_j, constants.nidx_rlm[1], initialDegree);
    j2 = findSPHId(idx_gl_1d_rlm_j, constants.nidx_rlm[1], initialDegree+1);
  }

  if ((initialDegree+1) < degree && degree < (initialDegree+2+STRIDE)) {
    for(int itr=0; itr<chunks; itr++) {
      double p1, p2, dp_;
      p1 = P_rtm[itr*( blockDim.x * blockDim.y * blockDim.z) + tId + (j1)*constants.nidx_rtm[1]]; 
      p2 = P_rtm[itr*( blockDim.x * blockDim.y * blockDim.z) + tId + (j2)*constants.nidx_rtm[1]]; 
      double theta = g_colat_rtm[itr*( blockDim.x * blockDim.y * blockDim.z) + tId];;
//      computeLegPoly(order, initialDegree, degree, g_colat_rtm[itr*( blockDim.x * blockDim.y * blockDim.z) + tId], reg1, reg2, &reg3, &reg0);
      for(int deg=initialDegree; deg<degree; deg++) {
        reg1 = __ddiv_rd((double) deg+2-order, (double) deg+2+order); 
        reg2 = __dmul_rd(cos(theta), p2);
        reg1 = __dsqrt_rd(reg1);
        reg2 = __dmul_rd((double) 2*deg + 3, reg2);
        reg3 = (double) (deg+2-order) * (deg+1-order);
        reg1 = __dmul_rd(reg1, reg2);
        reg2 = __dmul_rd((double) order+deg+1, p1);
        //dp_ is a misnomer here
        dp_ = (double) (deg+2+order) * (deg+order+1);
        p1 = p2;
        p2 = __dsqrt_rd(__ddiv_rd(reg3, dp_));
        p2 = __dmul_rd(p2, reg2);
        reg3 = (double) deg-order+2;
        // p1, m, l+2
        p2 = __ddiv_rd(__dadd_rd(reg1,-1*p2), reg3);
      }
      //dp_
      dp_ = __dmul_rd(cos(theta), p1);
      dp_ *= (double) degree;
      reg1 = __dmul_rd((double)order - degree - 3,p2);
      reg2 = __dsqrt_rd(__ddiv_rd((double) degree+order+1, (double) degree-order+1));
      dp_ = __dadd_rd(dp_, __dmul_rd(reg2, reg1));
      reg3 = -sin(theta);
      dp_ = __ddiv_rd(dp_, reg3);
      
      legendre[( blockDim.x * blockDim.y * blockDim.z) * itr + tId] = p1; 
      legendre[( blockDim.x * blockDim.y * blockDim.z) * itr + tId + constants.nidx_rtm[1]] = dp_; 
    } 
  }
  else {
    for(int itr=0; itr<chunks; itr++) {
      legendre[( blockDim.x * blockDim.y * blockDim.z) * itr + tId] = P_rtm[itr*( blockDim.x * blockDim.y * blockDim.z) + tId + blockIdx.x*constants.nidx_rtm[1]]; 
      legendre[( blockDim.x * blockDim.y * blockDim.z) * itr + tId + constants.nidx_rtm[1]] = dP_rtm[itr*( blockDim.x * blockDim.y * blockDim.z) + tId + blockIdx.x*constants.nidx_rtm[1]]; 
    }
  }

    //Cache the leftover theta terms.
  if (tId < rem) {
    if ((initialDegree+1) < degree && degree < (initialDegree+2+STRIDE)) {
      double p1, p2, dp_;
      double theta = g_colat_rtm[chunks*( blockDim.x * blockDim.y * blockDim.z) + tId];;
      p1 = P_rtm[chunks*( blockDim.x * blockDim.y * blockDim.z) + tId + (j1)*constants.nidx_rtm[1]]; 
      p2 = P_rtm[chunks*( blockDim.x * blockDim.y * blockDim.z) + tId + (j2)*constants.nidx_rtm[1]]; 
      for(int itr=initialDegree+1; itr<degree; itr++) {
        reg1 = __ddiv_rd((double) itr+2-order, (double) itr+2+order); 
        reg2 = __dmul_rd(cos(theta), p2);
        reg1 = __dsqrt_rd(reg1);
        reg2 = __dmul_rd((double) 2*itr + 3, reg2);
        reg3 = (double) (itr+2-order) * (itr+1-order);
        reg1 = __dmul_rd(reg1, reg2);
        reg2 = __dmul_rd((double) order+itr+1, p1);
        //dp_ is a misnomer here
        dp_ = (double) (itr+2+order) * (itr+order+1);
        p1 = p2;
        p2 = __dsqrt_rd(__ddiv_rd(reg3, dp_));
        p2 = __dmul_rd(p2, reg2);
        reg3 = (double) itr-order+2;
        // p1, m, l+2
        p2 = __ddiv_rd(__dadd_rd(reg1,-1*p2), reg3);
      }
      //dp_
      dp_ = __dmul_rd(cos(theta), p1);
      dp_ *= (double) degree;
      reg1 = __dmul_rd((double)order - degree - 3,p2);
      reg2 = __dsqrt_rd(__ddiv_rd((double) degree+order+1, (double) degree-order+1));
      dp_ = __dadd_rd(dp_, __dmul_rd(reg2, reg1));
      reg3 = -sin(theta);
      dp_ = __ddiv_rd(dp_, reg3);
      
      legendre[( blockDim.x * blockDim.y * blockDim.z) * chunks + tId] = p1; 
      legendre[( blockDim.x * blockDim.y * blockDim.z) * chunks + tId + constants.nidx_rtm[1]] = dp_; 
    }
    else {
      legendre[( blockDim.x * blockDim.y * blockDim.z) * chunks + tId] = P_rtm[chunks*( blockDim.x * blockDim.y * blockDim.z) + tId + blockIdx.x*constants.nidx_rtm[1]]; 
      legendre[( blockDim.x * blockDim.y * blockDim.z) * chunks + tId + constants.nidx_rtm[1]] = dP_rtm[chunks*( blockDim.x * blockDim.y * blockDim.z) + tId + blockIdx.x*constants.nidx_rtm[1]]; 
    }
  }
  __syncthreads();
  */
 
//** nVectors*nShells has to be less than or equal to nTheta
  //prefetchL1(weight_rtm, tId);
  //prefetchL1(asin_theta_1d_rtm, tId);

  //Case (nVec * nShells < nThreads) unaccounted for! 

//    vecId = tId % constants.nvector;
//    k_rtm = tId/constants.nvector;
       

//  int k_rtm = threadIdx.y;
//  int idx_p_rtm = blockIdx.x*constants.nidx_rtm[1]; 


  double gauss_norm = g_sph_rlm_7[blockIdx.x];
/*  double gauss_norm;
  if( order==0 ) {
    if(degree == 0)
      gauss_norm=1;
    else
      gauss_norm = (2*degree+1)/(2*degree*(degree+1));
  } else 
    gauss_norm = (2*degree+1)/(4*degree*(degree+1)); 
*/
  int idx_p_rtm = blockIdx.x*constants.nidx_rtm[1];
  int idx_p_rtm_sym = symModeIdx*constants.nidx_rtm[1]; 

  //double p_leg = P_rtm[idx_p_rtm];
  //p_leg = legendre[0];
  //dpdt = legendre[constants.nidx_rtm[1]];
//  double dpdt = dP_rtm[idx_p_rtm];
//  double weight =  weight_rtm[0];
//  double asin_t = asin_theta_1d_rtm[0];

  int mdx_p = mdx_p_rlm_rtm[blockIdx.x] - 1;
  int mdx_n = mdx_n_rlm_rtm[blockIdx.x] - 1;
  ip_rtm = threadIdx.y * constants.istep_rtm[0] ;
  mdx_p *= constants.istep_rtm[2];
  mdx_n *= constants.istep_rtm[2];
  mdx_p += ip_rtm;
  mdx_n += ip_rtm;

  ip_rtm = 3*(threadIdx.x+1) + constants.ncomp * mdx_p;
  in_rtm = 3*(threadIdx.x+1) + constants.ncomp * mdx_n;
  int stride = constants.ncomp * constants.istep_rtm[1];
 
  mdx_p = mdx_p_rlm_rtm[symModeIdx] - 1;
  mdx_n = mdx_n_rlm_rtm[symModeIdx] - 1;
  int ip_rtm_sym = threadIdx.y * constants.istep_rtm[0] ;
  mdx_p *= constants.istep_rtm[2];
  mdx_n *= constants.istep_rtm[2];
  mdx_p += ip_rtm_sym;
  mdx_n += ip_rtm_sym;

  ip_rtm_sym = 3*(threadIdx.x+1) + constants.ncomp * mdx_p;
  int in_rtm_sym = 3*(threadIdx.x+1) + constants.ncomp * mdx_n;
 
  double reg4, reg5, reg6, reg7; 
  double sp1_sym, sp2_sym, sp3_sym;

  for(int l_rtm=0; l_rtm<constants.nidx_rtm[1]; l_rtm++) {
    idx_p_rtm++; 
    idx_p_rtm_sym++; 
    reg0 = __dmul_rd(gauss_norm, weight_rtm[l_rtm]);
    reg1 = __dmul_rd(reg0, legendre[l_rtm]);
    reg2 = __dmul_rd(reg0, legendre[l_rtm+constants.nidx_rtm[1]]);
    sp1 = fma(vr_rtm[ip_rtm-3], reg1, sp1);
    sp1_sym = fma(vr_rtm[ip_rtm_sym-3], reg1, sp1_sym);
    reg3 = __dmul_rd(__dmul_rd(asin_theta_1d_rtm[l_rtm], (double) order), reg1);
    
  //  weight = weight_rtm[l_rtm];
  //  asin_t = asin_theta_1d_rtm[l_rtm];
//    p_leg = P_rtm[idx_p_rtm];
  //  p_leg = legendre[l_rtm];
 //   dpdt = dP_rtm[idx_p_rtm];
  //  dpdt = legendre[l_rtm + constants.nidx_rtm[1]];
    
    reg0 = __dmul_rd(vr_rtm[ip_rtm-2], reg2);
    reg4 = __dmul_rd(vr_rtm[ip_rtm_sym-2], reg2);

    reg1 =  -1 * __dmul_rd(vr_rtm[in_rtm-1], reg3);
    reg5 = __dmul_rd(vr_rtm[in_rtm_sym-1], reg3);

    reg2 *= vr_rtm[ip_rtm-1];
    reg3 *= vr_rtm[in_rtm-2];

    reg6 = reg2*vr_rtm[ip_rtm_sym-1];
    reg7 = -1*reg3*vr_rtm[in_rtm_sym-2];

    ip_rtm += stride;
    in_rtm += stride;
    ip_rtm_sym += stride;
    in_rtm_sym += stride;

    sp2 += __dadd_rd(reg0, reg1); 
    sp3 -= __dadd_rd(reg2, reg3); 

    sp2_sym += __dadd_rd(reg4, reg5);
    sp3_sym -+ __dadd_rd(reg6, reg7);
  }

  //reg0 = __dmul_rd(gauss_norm, weight);
  int idx_sp = constants.ncomp * ( blockIdx.x*constants.istep_rlm[1] + threadIdx.y*constants.istep_rlm[0]); 
  int idx_sp_sym = constants.ncomp * ( symModeIdx*constants.istep_rlm[1] + threadIdx.y*constants.istep_rlm[0]); 
  /*reg1 = __dmul_rd(reg0, p_leg);
  reg2 = __dmul_rd(reg0, dpdt);
  sp1 += __dmul_rd(vr_rtm[ip_rtm-3], reg1);
  reg3 = __dmul_rd(__dmul_rd(asin_t, (double) order), reg1);

  idx_sp += 3*(threadIdx.x+1); 
  reg0 = __dmul_rd(vr_rtm[ip_rtm-2], reg2);
  reg1 =  -1 * __dmul_rd(vr_rtm[in_rtm-1], reg3);
  reg2 *=  vr_rtm[ip_rtm-1];
  reg3 *=  vr_rtm[in_rtm-2];
  sp2 += __dadd_rd(reg0, reg1); 
  sp3 -= __dadd_rd(reg2, reg3); 
   */ 
  sp_rlm[idx_sp-3] = fma(__dmul_rd(radius_1d_rlm_r[threadIdx.y],radius_1d_rlm_r[threadIdx.y]), sp1, sp_rlm[idx_sp-3]);
  sp_rlm[idx_sp-2] = fma(radius_1d_rlm_r[threadIdx.y], sp2, sp_rlm[idx_sp-2]);
  sp_rlm[idx_sp-1] = fma(radius_1d_rlm_r[threadIdx.y], sp3, sp_rlm[idx_sp-1]);
  
  sp_rlm[idx_sp_sym-3] = fma(__dmul_rd(radius_1d_rlm_r[threadIdx.y],radius_1d_rlm_r[threadIdx.y]), sp1_sym, sp_rlm[idx_sp_sym-3]);
  sp_rlm[idx_sp_sym-2] = fma(radius_1d_rlm_r[threadIdx.y], sp2_sym, sp_rlm[idx_sp_sym-2]);
  sp_rlm[idx_sp_sym-1] = fma(radius_1d_rlm_r[threadIdx.y], sp3_sym, sp_rlm[idx_sp_sym-1]);
}

__global__ void normalizeLegendre(double *P_rtm, double *dP_rtm, double *g_sph_rlm_7, double *weight_rtm, const Geometry_c constants) {
   // dim3 grid(nidx_rlm[1])
   // dim3 block(nidx_rtm[1],1,1)
    P_rtm[blockIdx.x*constants.nidx_rtm[1] + threadIdx.x] *= g_sph_rlm_7[blockIdx.x] * weight_rtm[threadIdx.x];
    dP_rtm[blockIdx.x*constants.nidx_rtm[1] + threadIdx.x] *= g_sph_rlm_7[blockIdx.x] * weight_rtm[threadIdx.x];
}

__global__ void transF_vec_paired(symmetricModes *pairedList, double *vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants) 
{
 //dim3 grid(constants.nPairs,1,1);
  //dim3 block(constants.nvector, constants.nidx_rtm[0],1);

  extern __shared__ double legendre[];
  int k_rtm = threadIdx.y;
  
 // 3 for m-1, m, m+1
  unsigned int ip_rtm, in_rtm;
  
  double reg0, reg1, reg2, reg3;
  double sp1, sp2, sp3;
 
  int tId = (threadIdx.z * blockDim.x *blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;

  int order, modeIdx;
  order = pairedList[blockIdx.x].order;
  modeIdx = pairedList[blockIdx.x].positiveModeIdx;

  //Cache modes across thread block.
  //Assuming that the amount of shared memory is equivalent to the number of theta values.
  //TODO: For any given amount of shared memory, the block of threads should simply cache, compute,recache, and so on.
  int chunks = (constants.nidx_rtm[1])/( blockDim.x * blockDim.y * blockDim.z);  
  int rem = (constants.nidx_rtm[1]) % ( blockDim.x * blockDim.y * blockDim.z);

  for(int itr=0; itr < chunks; itr++) {
    legendre[( blockDim.x * blockDim.y * blockDim.z) * itr + tId] = P_rtm[itr*( blockDim.x * blockDim.y * blockDim.z) + tId + modeIdx*constants.nidx_rtm[1]]; 
    legendre[( blockDim.x * blockDim.y * blockDim.z) * itr + tId + constants.nidx_rtm[1]] = dP_rtm[itr*( blockDim.x * blockDim.y * blockDim.z) + tId + modeIdx*constants.nidx_rtm[1]]; 
  }
  //Cache the leftover theta terms.
  if (tId < rem) {
    legendre[( blockDim.x * blockDim.y * blockDim.z) * chunks + tId] = P_rtm[chunks*( blockDim.z * blockDim.x * blockDim.y ) + tId + modeIdx*constants.nidx_rtm[1]]; 
    legendre[( blockDim.x * blockDim.y * blockDim.z) * chunks + tId + constants.nidx_rtm[1]] = dP_rtm[chunks*( blockDim.z * blockDim.x * blockDim.y ) + tId + modeIdx*constants.nidx_rtm[1]]; 
  }

  __syncthreads();

  
//  #pragma unroll
  for(int i=0; i<2; i++) {
    if(i==1) {
      order = -1*pairedList[blockIdx.x].order;
      modeIdx = pairedList[blockIdx.x].negativeModeIdx;
    }

    double gauss_norm = g_sph_rlm_7[modeIdx]; 
//    double gauss_norm = (2*degree+1)/(4*degree*(degree+1)); 
    double weight = weight_rtm[0];
    double asin_t = asin_theta_1d_rtm[0];
    double p_leg = legendre[0];
    double dpdt = legendre[constants.nidx_rtm[1]];

    int mdx_p = mdx_p_rlm_rtm[modeIdx] - 1;
    int mdx_n = mdx_n_rlm_rtm[modeIdx] - 1;
    ip_rtm = k_rtm * constants.istep_rtm[0] ;
    mdx_p *= constants.istep_rtm[2];
    mdx_n *= constants.istep_rtm[2];
    mdx_p += ip_rtm;
    mdx_n += ip_rtm;

    ip_rtm = 3*(threadIdx.x+1) + constants.ncomp * mdx_p;
    in_rtm = 3*(threadIdx.x+1) + constants.ncomp * mdx_n;

    int stride = constants.ncomp * constants.istep_rtm[1];

    sp1=sp2=sp3=0;
    for(int l_rtm=1; l_rtm<constants.nidx_rtm[1]; l_rtm++) {
     sp1 += __dmul_rd(vr_rtm[ip_rtm-3], p_leg);
     reg3 = __dmul_rd(__dmul_rd(asin_t, (double) order), p_leg);
     reg0 = __dmul_rd(vr_rtm[ip_rtm-2], dpdt);
     reg1 =  -1 * __dmul_rd(vr_rtm[in_rtm-1], reg3);
     dpdt *= vr_rtm[ip_rtm-1];
     reg3 *= vr_rtm[in_rtm-2];
     ip_rtm += stride;
     in_rtm += stride;
     sp2 += __dadd_rd(reg0, reg1);
     sp3 -= __dadd_rd(dpdt, reg3);
     asin_t = asin_theta_1d_rtm[l_rtm];
     p_leg = legendre[l_rtm];
     dpdt = legendre[l_rtm+constants.nidx_rtm[1]];
    }
   
    reg0 = __dmul_rd(gauss_norm, weight);
    double r_1d_rlm_r = radius_1d_rlm_r[k_rtm];
    int idx_sp = constants.ncomp * ( modeIdx*constants.istep_rlm[1] + k_rtm*constants.istep_rlm[0]);
    sp1 += __dmul_rd(vr_rtm[ip_rtm-3], p_leg);
    reg3 = __dmul_rd(__dmul_rd(asin_t, (double) order), p_leg);

    idx_sp += 3*(threadIdx.x+1);
    reg0 = __dmul_rd(vr_rtm[ip_rtm-2], dpdt);
    reg1 =  -1 * __dmul_rd(vr_rtm[in_rtm-1], reg3);
    dpdt *= vr_rtm[ip_rtm-1];
    reg3 *= vr_rtm[in_rtm-2];
    double r_1d_sq = __dmul_rd(r_1d_rlm_r, r_1d_rlm_r);
    sp2 += __dadd_rd(reg0, reg1);
    sp3 -= __dadd_rd(dpdt, reg3);

    sp_rlm[idx_sp-3] += __dmul_rd(r_1d_sq, sp1);
    sp_rlm[idx_sp-2] += __dmul_rd(r_1d_rlm_r, sp2);
    sp_rlm[idx_sp-1] += __dmul_rd(r_1d_rlm_r, sp3);
  }    
}

__global__ void transF_vec_unpaired(unsymmetricModes *unpairedList, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants) {
 //dim3 grid(constants.nSingletons,1,1);
  //dim3 block(constants.nvector, constants.nidx_rtm[0],1);

  extern __shared__ double legendre[];
  int k_rtm = threadIdx.y;
  
 // 3 for m-1, m, m+1
  unsigned int ip_rtm, in_rtm;
  
  double reg0, reg1, reg2, reg3;
  double sp1, sp2, sp3;
 
  int tId = (threadIdx.z * blockDim.x *blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;

  int order = unpairedList[blockIdx.x].order;
  int modeIdx = unpairedList[blockIdx.x].modeIdx;


//  int order = idx_gl_1d_rlm_j[constants.nidx_rlm[1]*2 + blockIdx.x];
//  int degree = idx_gl_1d_rlm_j[constants.nidx_rlm[1] + blockIdx.x];

    //Cache modes across thread block.
    //Assuming that the amount of shared memory is equivalent to the number of theta values.
    //TODO: For any given amount of shared memory, the block of threads should simply cache, compute,recache, and so on.
  int chunks = (constants.nidx_rtm[1])/( blockDim.x * blockDim.y * blockDim.z);  
  int rem = (constants.nidx_rtm[1]) % ( blockDim.x * blockDim.y * blockDim.z);

  for(int itr=0; itr < chunks; itr++) {
    legendre[( blockDim.x * blockDim.y * blockDim.z) * itr + tId] = P_rtm[itr*( blockDim.x * blockDim.y * blockDim.z) + tId + modeIdx*constants.nidx_rtm[1]]; 
    legendre[( blockDim.x * blockDim.y * blockDim.z) * itr + tId + constants.nidx_rtm[1]] = dP_rtm[itr*( blockDim.x * blockDim.y * blockDim.z) + tId + modeIdx*constants.nidx_rtm[1]]; 
  }
  //Cache the leftover theta terms.
  if (tId < rem) {
    legendre[( blockDim.x * blockDim.y * blockDim.z) * chunks + tId] = P_rtm[chunks*( blockDim.z * blockDim.x * blockDim.y ) + tId + modeIdx*constants.nidx_rtm[1]]; 
    legendre[( blockDim.x * blockDim.y * blockDim.z) * chunks + tId + constants.nidx_rtm[1]] = dP_rtm[chunks*( blockDim.z * blockDim.x * blockDim.y ) + tId + modeIdx*constants.nidx_rtm[1]]; 
  }

    __syncthreads();

  double gauss_norm = g_sph_rlm_7[modeIdx];
  double weight = weight_rtm[0];
  double asin_t = asin_theta_1d_rtm[0];
  double p_leg = legendre[0]; 
  double dpdt = legendre[constants.nidx_rtm[1]];

  int mdx_p = mdx_p_rlm_rtm[modeIdx] - 1;
  int mdx_n = mdx_n_rlm_rtm[modeIdx] - 1;
  ip_rtm = k_rtm * constants.istep_rtm[0] ;
  mdx_p *= constants.istep_rtm[2];
  mdx_n *= constants.istep_rtm[2];
  mdx_p += ip_rtm;
  mdx_n += ip_rtm; 
  
  ip_rtm = 3*(threadIdx.x+1) + constants.ncomp * mdx_p;
  in_rtm = 3*(threadIdx.x+1) + constants.ncomp * mdx_n;
  
  int stride = constants.ncomp * constants.istep_rtm[1];
  
  sp1=sp2=sp3=0;
  
  for(int l_rtm=1; l_rtm<constants.nidx_rtm[1]; l_rtm++) {
    reg0 = __dmul_rd(gauss_norm, weight);
    reg1 = __dmul_rd(reg0, p_leg);
    reg2 = __dmul_rd(reg0, dpdt); 
    sp1 += __dmul_rd(vr_rtm[ip_rtm-3], reg1);
    reg3 = __dmul_rd(__dmul_rd(asin_t, (double) order), reg1);
    
    weight = weight_rtm[l_rtm];
    asin_t = asin_theta_1d_rtm[l_rtm];
    p_leg = legendre[l_rtm];
    dpdt = legendre[l_rtm+constants.nidx_rtm[1]];
    
    reg0 = __dmul_rd(vr_rtm[ip_rtm-2], reg2);
    reg1 =  -1 * __dmul_rd(vr_rtm[in_rtm-1], reg3);
    reg2 *= vr_rtm[ip_rtm-1];
    reg3 *= vr_rtm[in_rtm-2];
    ip_rtm += stride;
    in_rtm += stride;
    sp2 += __dadd_rd(reg0, reg1);
    sp3 -= __dadd_rd(reg2, reg3);
  } 
    
  reg0 = __dmul_rd(gauss_norm, weight);
  double r_1d_rlm_r = radius_1d_rlm_r[k_rtm];
  int idx_sp = constants.ncomp * ( modeIdx*constants.istep_rlm[1] + k_rtm*constants.istep_rlm[0]);
  reg1 = __dmul_rd(reg0, p_leg);
  reg2 = __dmul_rd(reg0, dpdt);
  sp1 += __dmul_rd(vr_rtm[ip_rtm-3], reg1);
  reg3 = __dmul_rd(__dmul_rd(asin_t, (double) order), reg1);

  idx_sp += 3*(threadIdx.x+1); 
  reg0 = __dmul_rd(vr_rtm[ip_rtm-2], reg2);
  reg1 =  -1 * __dmul_rd(vr_rtm[in_rtm-1], reg3);
  reg2 *= vr_rtm[ip_rtm-1];
  reg3 *= vr_rtm[in_rtm-2];
  double r_1d_sq = __dmul_rd(r_1d_rlm_r, r_1d_rlm_r);
  sp2 += __dadd_rd(reg0, reg1); 
  sp3 -= __dadd_rd(reg2, reg3); 
    

  sp_rlm[idx_sp-3] += __dmul_rd(r_1d_sq, sp1);
  sp_rlm[idx_sp-2] += __dmul_rd(r_1d_rlm_r, sp2);
  sp_rlm[idx_sp-1] += __dmul_rd(r_1d_rlm_r, sp3);
}
 
__global__ void transF_vec_paired_tiny(symmetricModes *pairedList, const int kLoad, double const* vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_rtm, double *dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants) 
{
 //dim3 grid(constants.nPairs,constants.nidx_rtm[0],1);
  //dim3 block(constants.nvector,1);

  extern __shared__ double legendre[];

  
 // 3 for m-1, m, m+1
  int ip_rtm, in_rtm;
  
  double reg0, reg1, reg2, reg3;
  double sp1, sp2, sp3;
 
  int tId = threadIdx.x;

  int order, modeIdx;
  order = pairedList[blockIdx.x].order;
  modeIdx = pairedList[blockIdx.x].positiveModeIdx;

  //Cache modes across thread block.
  //Assuming that the amount of shared memory is equivalent to the number of theta values.
  //TODO: For any given amount of shared memory, the block of threads should simply cache, compute,recache, and so on.
  int chunks = (constants.nidx_rtm[1])/blockDim.x;  
  int rem = (constants.nidx_rtm[1]) % blockDim.x;

  int stride = modeIdx*constants.nidx_rtm[1] + tId;
  for(int itr=0; itr < chunks; itr++) {
    legendre[blockDim.x * itr + tId] = P_rtm[itr*blockDim.x + stride]; 
    legendre[blockDim.x * itr + tId] = dP_rtm[itr*blockDim.x + stride]; 
  }
  //Cache the leftover theta terms.
  if (tId < rem) {
    legendre[ blockDim.x * chunks + tId] = P_rtm[chunks*blockDim.x + stride]; 
    legendre[ blockDim.x * chunks + tId + constants.nidx_rtm[1]] = dP_rtm[chunks*blockDim.x + stride]; 
  }

  __syncthreads();

  
  for(int i=0; i<2; i++) {
    if(i==1) {
      order = -1*pairedList[blockIdx.x].order;
      modeIdx = pairedList[blockIdx.x].negativeModeIdx;
    }

    // Constant variables over block of threads: 
    double gauss_norm = g_sph_rlm_7[modeIdx]; 
//    double gauss_norm = (2*degree+1)/(4*degree*(degree+1)); 
    int mdx_p = (mdx_p_rlm_rtm[modeIdx] - 1) * constants.istep_rtm[2];
    int mdx_n = (mdx_n_rlm_rtm[modeIdx] - 1) * constants.istep_rtm[2];

    for(int k_rtm=0; k_rtm<kLoad; k_rtm++) {
      k_rtm = blockIdx.y;
      double weight = weight_rtm[0];
      double asin_t = asin_theta_1d_rtm[0];
      double p_leg = legendre[0];
      double dpdt = legendre[constants.nidx_rtm[1]];

      ip_rtm = 3*(threadIdx.x+1) + constants.ncomp * (mdx_p + k_rtm * constants.istep_rtm[0]);
      in_rtm = 3*(threadIdx.x+1) + constants.ncomp * (mdx_n + k_rtm * constants.istep_rtm[0]);

      int stride = constants.ncomp * constants.istep_rtm[1];

      double sp1=sp2=sp3=0;
      for(int l_rtm=0; l_rtm<constants.nidx_rtm[1]; l_rtm++, ip_rtm+=stride, in_rtm+=stride) {
        reg0 = gauss_norm * weight_rtm[l_rtm] * legendre[l_rtm];
        reg1 = gauss_norm * weight_rtm[l_rtm] * legendre[l_rtm+constants.nidx_rtm[1]];
        sp1 += vr_rtm[ip_rtm-3] * reg0;
        sp2 += vr_rtm[ip_rtm-2]*reg1 - vr_rtm[in_rtm-1]*asin_theta_1d_rtm[l_rtm]*order*reg0;
        sp3 -+ vr_rtm[ip_rtm-1]*reg1 + vr_rtm[in_rtm-2]*asin_theta_1d_rtm[l_rtm]*order*reg0;
      }

      double r_1d_rlm_r = radius_1d_rlm_r[k_rtm];
      double r_1d_sq = __dmul_rd(r_1d_rlm_r, r_1d_rlm_r);

      int idx_sp = 3*(threadIdx.x+1) + constants.ncomp * ( modeIdx*constants.istep_rlm[1] + k_rtm*constants.istep_rlm[0]);

      sp_rlm[idx_sp-3] += r_1d_rlm_r*r_1d_rlm_r*sp1;
      sp_rlm[idx_sp-2] += r_1d_rlm_r*sp2;
      sp_rlm[idx_sp-1] += r_1d_rlm_r*sp3;
    }
  }    
}

//Reduction using an open source library CUB supported by nvidia
/*template <
    int     THREADS_PER_BLOCK,
    int			ITEMS_PER_THREAD,
    cub::BlockReduceAlgorithm ALGORITHM,
    typename T>
__global__ void transF_vec_paired(symmetricModes *pairedList, int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants) {
  //dim3 grid(constants.nidx_rlm[1],constants.nidx_rtm[0],1); 
  //dim3 block(nThreads,1,1);
  // nThreads * ITEMS_PER_THREAD = nTheta

  typedef cub::BlockReduce<T, THREADS_PER_BLOCK, ALGORITHM> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  int k_rtm = blockIdx.y;
  int j_rlm = pairedList[blockIdx.x].positiveModeIdx; 

// 3 for m-1, m, m+1
  unsigned int ip_rtm, in_rtm;

  double reg0, reg1, reg2, reg3, reg4;
  double sp1, sp2, sp3; 

  int order = pairedList[blockIdx.x].order;

  double gauss_norm = g_sph_rlm_7[j_rlm];
  int nTheta = constants.nidx_rtm[1];
  int nVector = constants.nvector;
  int nComp = constants.ncomp;

  int mdx_p = mdx_p_rlm_rtm[j_rlm] - 1;
  ip_rtm = k_rtm * constants.istep_rtm[0];
  int mdx_n = mdx_n_rlm_rtm[j_rlm] - 1;
  mdx_p *= constants.istep_rtm[2];
  mdx_n *= constants.istep_rtm[2];
  mdx_p += ip_rtm;
  mdx_n += ip_rtm;

  int idx;
  int idx_p_rtm = j_rlm*nTheta; 
 
  double r_1d_rlm_r = radius_1d_rlm_r[k_rtm]; 
  int idx_sp = nComp * ( blockIdx.x*constants.istep_rlm[1] + k_rtm*constants.istep_rlm[0]); 

  double poloidal[ITEMS_PER_THREAD];
  double radial_diff_poloidal[ITEMS_PER_THREAD]; 
  double toroidal[ITEMS_PER_THREAD];

  unsigned int l_rtm=0;

  for(int t=1; t<=nVector; t++) {
    sp1=sp2=sp3=0;
    for(int counter=0; counter < ITEMS_PER_THREAD; counter++) {  
      l_rtm = j_rlm*counter + threadIdx.x; 
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
*/
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
/*
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
*/

void legendre_f_trans_cuda_(int *ncomp, int *nvector, int *nscalar) {
  static int nShells = constants.nidx_rtm[0];

  constants.ncomp = *ncomp;
  constants.nscalar= *nscalar;
  constants.nvector = *nvector;

  dim3 grid(constants.nidx_rlm[1],nShells,1);
  dim3 block(constants.nvector, constants.nidx_rtm[0],1);

  //ToDo: Ponder this: if not exact, what are the consequences?
  //Extremeley important! *****
  //int itemsPerThread = constants.nidx_rtm[1]/blockSize; 
  //std::assert(itemsPerThread*blockSize == constants.nidx_rtm[1]);
  //std::assert(minGridSize <= constants.nidx_rlm[1]);

#ifdef CUDA_TIMINGS
  static Timer transF_v("fwd vector algorithm ");
  cudaPerformance.registerTimer(&transF_v);
  transF_v.startTimer();
#endif
/*  transF_vec_reduction< 32, 3,
                  cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                      double>
            <<<grid, 32>>> (deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.radius_1d_rlm_r, 
                        deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.mdx_n_rlm_rtm, deviceInput.a_r_1d_rlm_r, 
                        deviceInput.g_colat_rtm, deviceInput.p_rtm, deviceInput.dP_rtm, deviceInput.g_sph_rlm_7, deviceInput.asin_theta_1d_rtm, 
                        constants);
*/
  transF_vec<<<constants.nidx_rlm[1], block, 0, streams[0]>>> (deviceInput.idx_gl_1d_rlm_j, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.radius_1d_rlm_r, deviceInput.mdx_p_rlm_rtm, deviceInput.mdx_n_rlm_rtm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, deviceInput.p_rtm, deviceInput.dP_rtm, deviceInput.asin_theta_1d_rtm, constants);
//  transF_vec_unpaired<<<constants.nSingletons, block, 2*sizeof(double)*constants.nidx_rtm[1], streams[0]>>> (deviceInput.unpairedList, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.radius_1d_rlm_r, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.mdx_n_rlm_rtm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, deviceInput.p_rtm, deviceInput.dP_rtm, deviceInput.g_sph_rlm_7, deviceInput.asin_theta_1d_rtm, constants);
//  transF_vec_paired<<<constants.nPairs, block, 2*sizeof(double)*constants.nidx_rtm[1], streams[1]>>> (deviceInput.pairedList, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.radius_1d_rlm_r, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.mdx_n_rlm_rtm, deviceInput.a_r_1d_rlm_r, deviceInput.g_colat_rtm, deviceInput.p_rtm, deviceInput.dP_rtm, deviceInput.g_sph_rlm_7, deviceInput.asin_theta_1d_rtm, constants);
 
#ifdef CUDA_TIMINGS
  cudaDevSync();
  transF_v.endTimer();

  static Timer transF_s("Fwd scalar algorithm ");
  cudaPerformance.registerTimer(&transF_s);
  transF_s.startTimer();
#endif
  /*transF_scalar_reduction <32, 3, 
                     cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                     double>
               <<<grid, 32>>> (deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.p_rtm, deviceInput.g_sph_rlm_7, constants);
*/  
  transF_scalar<<<constants.nidx_rlm[1], nShells, 0, streams[1]>>> (1, deviceInput.vr_rtm, deviceInput.sp_rlm, deviceInput.weight_rtm, deviceInput.mdx_p_rlm_rtm, deviceInput.p_rtm, deviceInput.g_sph_rlm_7, constants);
#ifdef CUDA_TIMINGS
  cudaDevSync();
  transF_s.endTimer();
#endif
  
}

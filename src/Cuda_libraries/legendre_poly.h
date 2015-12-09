//Author: Harsha V. Lokavarapu

#include <cstdlib>
#include "helper_cuda.h"
#include <string>
#include <math.h>
#include "cuda.h"
#include <fstream>
#include <iostream>
//#include <mpi.h>
//#include <cub/block/block_reduce.cuh>
//#include "/home/harsha_lv/Downloads/cub-1.4.1/cub/block/block_reduce.cuh"
//#include "/home/harsha_lv/Downloads/cub-1.4.1/cub/util_allocator.cuh"
//#include "/home/harsha_lv/Downloads/cub-1.4.1/cub/device/device_reduce.cuh"

#include "logger.h"

#ifdef CUDA_TIMINGS
  #include "cuda_profiler_api.h"
#endif
#define ARGC 3 

using namespace std;

/*#if __CUDA_ARCH__ < 350 
#error "Incompatable compute capability for sm. Using dynamic parallelism (>= 35)"
#endif
*/
//Function and variable declarations.
extern int nComp;
// For the kernel transF_vec_reduction
extern int minGridSize;
extern int blockSize; 

extern cudaDeviceProp prop;
extern size_t devMemory;

extern int my_rank;

//CUDA Unbound - part of device reduce example
//extern bool g_verbose; // Whether to display input/output to console
//extern cub::CachingDeviceAllocator g_allocator; // Caching allocator for device memory

/*
 *   Set of variables that take advantage of constant memory.
 *     Access to constant memory is faster than access to global memory.
 *       */

// **** lstack_rlm resides in global memory as well as constant memory
// ** Pick one or the other
//extern __constant__ int lstack_rlm_cmem[1000];

// Fortran function calls
extern "C" {
  void legendre_b_trans_cuda_(int*, int*, int*);
  void legendre_f_trans_cuda_(int*, int*, int*);
}

//Fortran Variables
typedef struct {
  int nidx_rtm[3];
  int nidx_rlm[2];
  int istep_rtm[3];
  int istep_rlm[2];
  int nnod_rtp;
  int nnod_rlm;
  int nnod_rtm;
  int ncomp;
  int nscalar;
  int nvector;
  int t_lvl;
  int np_smp;
} Geometry_c;

//Cublas library/Cuda variables
extern cudaError_t error;
extern cudaStream_t *streams;
extern int nStreams;

//Helper functions, declared but not defined. 

extern void cudaErrorCheck(cudaError_t error);

typedef struct 
{
  // OLD: 0 = g_point_med, 1 =  double* g_colat_med, 2 = double* weight_med;
  // Current: 0 = vr_rtm,  = g_sph_rlm
  double *vr_rtm, *g_colat_rtm, *g_sph_rlm, *g_sph_rlm_7;
  double *sp_rlm;
  double *a_r_1d_rlm_r; //Might be pssible to copy straight to constant memory
  double *asin_theta_1d_rtm;
  int *idx_gl_1d_rlm_j;
  int *lstack_rlm;
  double *radius_1d_rlm_r, *weight_rtm;
  int *mdx_p_rlm_rtm, *mdx_n_rlm_rtm;
  double *p_jl;
  double *dP_jl;
  double *p_rtm, *dP_rtm;
  double *leg_poly_m_eq_l;
  //cudaUnbound dev reduction 
  double *reductionSpace;
} Parameters_s;

typedef struct
{
  int *mdx_p_rlm_rtm;
  int *mdx_n_rlm_rtm;
  int *idx_gl_1d_rlm_j;
  double *radius_1d_rlm_r;
  double *g_sph_rlm_7;
} References;

typedef struct 
{
  double *P_smdt; 
  double *dP_smdt;
  double *g_sph_rlm;
  int *lstack_rlm;
  int *idx_gl_1d_rlm_j;
  double *g_colat_rtm;
  double *vr_rtm, *sp_rlm;
// Dim: jx3
} Debug;

//CPU pointers to GPU memory data
extern Parameters_s deviceInput;
//Memory for debugging kernels
extern Debug h_debug, d_debug;
extern Geometry_c constants;
//C Pointers to Fortran allocated memory
extern References hostData;

//FileStream for recording performance 
extern Logger cudaPerformance;

// Counters for forward and backward Transform
extern int countFT, countBT;

////////////////////////////////////////////////////////////////////////////////
//! Function Defines
////////////////////////////////////////////////////////////////////////////////
extern "C" {

void initialize_gpu_(int *my_rank);
//void initgpu_(int *nnod_rtp, int *nnod_rtm, int *nnod_rlm, int nidx_rtm[], int nidx_rtp[], int istep_rtm[], int istep_rlm[], int *ncomp, double *a_r_1d_rlm_r, int lstack_rlm[], double *g_colat_rtm, int *trunc_lvl, double *g_sph_rlm);
void set_constants_(int *nnod_rtp, int *nnod_rtm, int *nnod_rlm, int nidx_rtm[], int nidx_rtp[], int istep_rtm[], int istep_rlm[], int *t_lvl, int *np_smp);
void setptrs_(int *idx_gl_1d_rlm_j);
void initialize_leg_trans_gpu_();
void finalizegpu_(); 
void initDevConstVariables();

void alloc_space_on_gpu_(int *ncomp, int *nvector, int *nscalar);
void memcpy_h2d_(int *lstack_rlm, double *a_r, double *g_colat, double *g_sph_rlm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, int *idx_gl_1d_rlm_j, double *rad_1d_rlm_r, double *weights, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm);
void deAllocMemOnGPU();
void deAllocDebugMem();
void allocHostDebug(Debug*);
void allocDevDebug(Debug*);
void cpy_field_dev2host_4_debug_(int *ncomp);
void cpy_spec_dev2host_4_debug_(int *ncomp);
void cpy_schmidt_2_gpu_(double *P_jl, double *dP_jl, double *P_rtm, double *dP_rtm);
void set_spectrum_data_(double *sp_rlm, int *ncomp);
void set_physical_data_(double *vr_rtm, int *ncomp);
void retrieve_spectrum_data_(double *sp_rlm, int *ncomp);
void retrieve_physical_data_(double *vr_rtm, int *ncomp);
void clear_spectrum_data_(int *ncomp);
void clear_field_data_(int *ncomp);
void check_bwd_trans_cuda_(int*, double*, double*, double*);
void check_fwd_trans_cuda_(int *my_rank, double *sp_rlm);
void output_spectral_data_cuda_(int *my_rank, int *ncomp, int *nvector, int *nscalar);
void cleangpu_();
void cuda_sync_device_();

void find_optimal_algorithm_(int *ncomp, int *nvector, int *nscalar);

  __device__ double nextLGP_m_eq0(int l, double x, double p_0, double p_1);
  __device__ double nextDp_m_eq_0(int l, double lgp_mp);
  __device__ double nextDp_m_eq_1(int l, double p_mn_l, double p_pn_l);
  __device__ double nextDp_m_l(int m, int l, double p_mn, double p_pn);
  __device__ double calculateLGP_m_eq_l(int mode);
  __device__ double calculateLGP_mp1_eq_l(int mode, double x, double lgp_m_eq_l);
  __device__ double calculateLGP_m_l(int m, int degree, double theta, double lgp_0, double lgp_1);
  __device__ double calculateLGP_m_l_mod(int m, int degree, double cos_theta, double lgp_0, double lgp_1);
  __device__ double scaleBySine(int mode, double lgp, double theta);
}

// Shortcuts and Simplifications
  void cudaDevSync();
//

//enum fwd_vec{naive, naive_w_more_threads, reduction}; 
enum fwd_vec{naive, naive_w_more_threads}; 

/*__global__ void transB_m_l_eq0_ver1D(int mp_rlm, int jst, int jed, double *vr_rtm,  double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_smdt, double *dP_smdt, double *g_sph_rlm, double *asin_theta_1d_rtm); 
__global__ void transB_m_l_eq1_ver1D( int mp_rlm,  int jst,  int jed, int order, int degree, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r,     double *g_colat_rtm, double *P_smdt, double *dP_smdt, double *g_sph_rlm, double *asin_theta_1d_rtm);
__global__ void transB_m_l_ver1D( int mp_rlm,  int jst,  int jed, int order, int degree, double *vr_rtm,  double *sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_smdt, double *dP_smdt, double *g_sph_rlm, double *asin_theta_1d_rtm);
__global__ void transB_m_l_ver2D(int *lstack_rlm, int m0, int m1, int *idx_gl_1d_rlm_j, double *vr_rtm, double *sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_smdt, double *dP_smdt, double *g_sph_rlm, double *asin_theta_1d_rtm);
__global__ void transB_m_l_ver3D(int *lstack_rlm, int m0, int m1, int *idx_gl_1d_rlm_j, double *vr_rtm, double *sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_smdt, double *dP_smdt, double *g_sph_rlm, double *asin_theta_1d_rtm);
*/
__global__ void transB_m_l_ver3D_block_of_vectors(int *lstack_rlm, int m0, int m1, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_smdt, double *dP_smdt, double *g_sph_rlm, double *asin_theta_1d_rtm);
__global__ void transB_m_l_ver3D_block_of_vectors_smem(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_smdt, double *dP_smdt, double *g_sph_rlm, double *asin_theta_1d_rtm);
/*
__global__ void transB_m_l_ver3D_block_of_scalars(int *lstack_rlm, int m0, int m1, int *idx_gl_1d_rlm_j, double *vr_rtm, double *sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_smdt, double *dP_smdt, double *g_sph_rlm, double *asin_theta_1d_rtm);
__global__ void transB_m_l_ver4D(int *lstack_rlm, int m0, int m1, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_smdt, double *dP_smdt, double *g_sph_rlm, double *asin_theta_1d_rtm);
__global__ void transB_m_l_ver5D(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_smdt, double *dP_smdt, double *g_sph_rlm, double *asin_theta_1d_rtm);
__global__ void transB_m_l_ver6D(int *lstack_rlm, int m0, int m1, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *P_jl, double *dP_jl, double *g_sph_rlm, double *asin_theta_1d_rtm);
__global__ void transB(double *vr_rtm, const double *sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm); 
*/
#ifndef CUDA_DEBUG
__global__ void transB_m_l_neo(int const* __restrict__ lstack_rlm, int const* __restrict__ idx_gl_1d_rlm_j, double const* __restrict__ g_colat_rtm, double const* __restrict__ g_sph_rlm, double const* __restrict__ asin_theta_1d_rtm, double const* __restrict__ a_r_1d_rlm_r, double const* __restrict__ sp_rlm, double *vr_rtm, double *leg_poly_m_eq_l,const Geometry_c constants);
#else
__global__ void transB_m_l_neo(int const* __restrict__ lstack_rlm, int const* __restrict__ idx_gl_1d_rlm_j, double const* __restrict__ g_colat_rtm, double const* __restrict__ g_sph_rlm, double const* __restrict__ asin_theta_1d_rtm, double const* __restrict__ a_r_1d_rlm_r, double const* __restrict__ sp_rlm, double *vr_rtm, double *P_smdt, double *dP_smdt, double *leg_poly_m_eq_l, const Geometry_c constants);
#endif
__global__ void set_leg_poly_m_ep_l(double *leg_poly_m_eq_l);

__global__ void transB_dydt(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl, double *dP_jl, const Geometry_c constants);
__global__ void transB_dydt_smem_dpschmidt(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double const* __restrict__ P_jl, double const* __restrict__ dP_jl, const Geometry_c constants);
__global__ void transB_dydt_smem_schmidt_more_threads(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl, double *dP_jl, const Geometry_c constants);
__global__ void transB_dydt_smem_a_r(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double const* __restrict__ P_jl, double *dP_jl, const Geometry_c constants);
__global__ void transB_dydt_read_only_data(int const* __restrict__ lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double const* __restrict__ P_jl, double const* __restrict__ dP_jl);
__global__ void transB_dydp(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl, double *asin_theta_1d_rtm, const Geometry_c constants);
__global__ void transB_dydp_smem_schmidt_more_threads(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl, double *asin_theta_1d_rtm, const Geometry_c constants);
__global__ void transB_scalar(int *lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *P_jl, const Geometry_c constants);
__global__ void transB_scalar_opt_mem_access(int *lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *P_jl);
__global__ void transB_scalar_block_mp_rlm(int const* __restrict__ lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double const* __restrict__ P_jl);
__global__ void transB_scalar_block_mp_rlm_smem(int const* __restrict__ lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *P_jl);
__global__ void transB_scalar_L1_cache(int const* __restrict__ lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double const* __restrict__ P_jl);
__global__ void transB_scalar_smem(int *lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *P_jl, const Geometry_c constants);
__global__ void transB_scalars_OTF(int *lstack_rlm, int m0, int m1, int *idx_gl_1d_rlm_j, double *vr_rtm, double *sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *g_sph_rlm, double *asin_theta_1d_rtm); 
__global__ void transB_scalars_OTF_smem(int *lstack_rlm, int m0, int m1, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *g_colat_rtm, double *g_sph_rlm, double *asin_theta_1d_rtm);

__global__ void transF_vec(int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants); 
__global__ void transF_vec(int kst, int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants); 

__global__ void transF_vec_smem_schmidt(int kst, int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants); 
__global__ void transF_scalar(int kst, double *vr_rtm, double *sp_rlm, double *weight_rtm, int *mdx_p_rlm_rtm, double *P_rtm, double *g_sph_rlm_7, const Geometry_c constants);

//Reduction Declerations
void transformMode(int shellId, int modeId);
__global__ void integrateFirstComponent(bool init, int shellId, int modeId, int vectorId, int order, int mdx_n, int mdx_p, double r_1d_rlm_r_sq, double gauss_norm, double *g_colat_rtm, double *weight_rtm, double *asin_theta_1d_rtm, double const* __restrict__ P_rtm, double *sp_rlm, double const* __restrict__ input, double *output, const Geometry_c constants);

//Reduction using CUDA UnBound
/*template< 
      int THREADS_PER_BLOCK,
      int ITEMS_PER_THREAD,
      cub::BlockReduceAlgorithm ALGORITHM,
      typename T> 
__global__ void transF_vec_reduction(int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, const Geometry_c constants);

template <
    int     THREADS_PER_BLOCK,
    int			ITEMS_PER_THREAD,
    cub::BlockReduceAlgorithm ALGORITHM,
    typename T>
__global__
void transF_scalar_reduction(double *vr_rtm, double *sp_rlm, double *weight_rtm, int *mdx_p_rlm_rtm, double *P_rtm, double *g_sph_rlm_7, const Geometry_c constants); 


template <
    int     THREADS_PER_BLOCK,
    cub::BlockReduceAlgorithm ALGORITHM,
    typename T>
__global__
void transB_dydt_reduction(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *g_sph_rlm, double *a_r_1d_rlm_r, double *P_jl, double *dP_jl, const Geometry_c constants);
*/
__global__ void transB_dydt_old(double *g_sph_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl, double *dP_jl, const Geometry_c constants);
/*
template <
    int     THREADS_PER_BLOCK,
    cub::BlockReduceAlgorithm ALGORITHM,
    typename T>
__global__
void transB_dydp_reduction(int *lstack_rlm, int *idx_gl_1d_rlm_j, double *vr_rtm, double const* __restrict__ sp_rlm, double *a_r_1d_rlm_r, double *P_jl, double *asin_theta_1d_rtm, const Geometry_c constants);

template <
    int     THREADS_PER_BLOCK,
    cub::BlockReduceAlgorithm ALGORITHM,
    typename T>
__global__
void transB_scalar_reduction(int *lstack_rlm, double *vr_rtm, double const* __restrict__ sp_rlm, double *P_jl, const Geometry_c constants);
*/
//A unary function whose input is the block size and returns the size in bytes of shared memory needed by a block
size_t computeSharedMemory(int blockSize);
void registerAllTimers();

//Reduction over device
__global__ void prepateInput(int nVec, int k_rlm, int j_rlm, int *idx_gl_1d_rlm_j, double const* __restrict__ vr_rtm, double *sp_rlm, double *radius_1d_rlm_r, double *weight_rtm, 
                                      int *mdx_p_rlm_rtm, int *mdx_n_rlm_rtm, double *a_r_1d_rlm_r, double *g_colat_rtm, double const* __restrict__ P_rtm, 
                                      double const* __restrict__ dP_rtm, double *g_sph_rlm_7, double *asin_theta_1d_rtm, double *input, 
                                      const Geometry_c constants);


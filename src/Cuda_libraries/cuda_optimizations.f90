!Author: Harsha Lokavarapu
!Date: 9/13/14

      module cuda_optimizations 
        use calypso_mpi
        use m_precision
        use m_spheric_parameter
        use m_spheric_param_smp
        use m_machine_parameter
        use m_schmidt_poly_on_rtm
        use schmidt_fix_m
        use m_work_4_sph_trans
        use m_work_4_sph_trans_spin
        use spherical_SRs_N
        use const_coriolis_sph_rlm
        use legendre_bwd_trans_org

        implicit none
 
#ifdef CUDA_DEBUG
!>      field data for Legendre transform  @f$ f(r,\theta,m) @f$ 
!!@n     size:  vr_rtm(ncomp*nidx_rtm(2)*nidx_rtm(1)*nidx_rtm(3))
!      real(kind = kreal), allocatable :: vr_rtm_wk_debug(:)
!
!>      Spectr data for Legendre transform  @f$ f(r,l,m) @f$ 
!>@n      size: sp_rlm(ncomp*nidx_rlm(2)*nidx_rtm(1))
      real(kind = kreal), allocatable :: sp_rlm_wk_debug(:)
#endif

!                                                                                                 
!>     Maximum matrix size for spectr data
      integer(kind = kint) :: nvec_jk
!>     Poloidal component with evem (l-m)
      real(kind = kreal), allocatable :: pol_e(:,:)
!>     radial difference of Poloidal component with evem (l-m)
      real(kind = kreal), allocatable :: dpoldt_e(:,:)
!>     radial difference of Poloidal component with evem (l-m)
      real(kind = kreal), allocatable :: dpoldp_e(:,:)
!>     Toroidal component with evem (l-m)
      real(kind = kreal), allocatable :: dtordt_e(:,:)
!>     Toroidal component with evem (l-m)
      real(kind = kreal), allocatable :: dtordp_e(:,:)

!
!>     Maximum matrix size for field data
      integer(kind = kint) :: nvec_lk
!>     Symmetric radial component
      real(kind = kreal), allocatable :: symp_r(:)
!>     Anti-symmetric theta-component
      real(kind = kreal), allocatable :: asmp_t(:)
!>     Anti-symmetric phi-component
      real(kind = kreal), allocatable :: asmp_p(:)
!>     Symmetric theta-component with condugate order
      real(kind = kreal), allocatable :: symn_t(:)
!>     Symmetric phi-component with condugate order
      real(kind = kreal), allocatable :: symn_p(:)
!
!>     Maximum matrix size for spectr data
      integer(kind = kint) :: nscl_jk
!>     Scalar with evem (l-m)
      real(kind = kreal), allocatable :: scl_e(:,:)
!
!>     Maximum matrix size for field data
      integer(kind = kint) :: nscl_lk
!>     Symmetric scalar component
      real(kind = kreal), allocatable :: symp(:)
   
        contains
      
        subroutine calypso_gpu_init
          call initialize_gpu
        end subroutine calypso_gpu_init

        subroutine alloc_mem_4_reordering(nvector, nscalar)
          integer(kind = kint), intent(in) :: nvector, nscalar
          
!          nvec_jk = maxdegree_rlm * nidx_rlm(1)*nvector
!          allocate(pol_e(nvec_jk,nidx_rlm(1)))
!          allocate(dpoldt_e(nvec_jk,nidx_rlm(1)))
!          allocate(dpoldp_e(nvec_jk,nidx_rlm(1)))
!          allocate(dtordt_e(nvec_jk,nidx_rlm(1)))
!          allocate(dtordp_e(nvec_jk,nidx_rlm(1)))
!     
          nvec_lk = nidx_rtm(2) * nidx_rtm(1)*nvector* min(nidx_rtm(3),32)
          allocate(symp_r(nvec_lk))
          allocate(symn_t(nvec_lk))
          allocate(symn_p(nvec_lk))
          allocate(asmp_t(nvec_lk))
          allocate(asmp_p(nvec_lk))

!          nscl_jk = maxdegree_rlm * nidx_rlm(1)*nscalar
!          allocate(scl_e(nscl_jk,nidx_rlm(1)))
!
          nscl_lk = nidx_rtm(2) * nidx_rtm(1)*nscalar*min(nidx_rtm(3),32)
          allocate(symp(nscl_lk))
        end subroutine alloc_mem_4_reordering

        subroutine clear_memory_fwd_trans_reordering
          
!$omp parallel workshare
            
          symp_r(nvec_lk) = 0.0d0
          symn_t(nvec_lk) = 0.0d0
          symn_p(nvec_lk) = 0.0d0
          asmp_t(nvec_lk) = 0.0d0
          asmp_p(nvec_lk) = 0.0d0

!          nscl_jk = maxdegree_rlm * nidx_rlm(1)*nscalar
!          scl_e(nscl_jk,nidx_rlm(1)) = 0.0d0
!
          symp(nscl_lk) = 0.0d0
!$omp end parallel workshare
        end subroutine clear_memory_fwd_trans_reordering
        subroutine dealloc_mem_4_reordering
!          deallocate(pol_e, dpoldt_e, dpoldp_e, dtordt_e, dtordp_e)
          deallocate(symp_r, symn_t, symn_p, asmp_t, asmp_p)
          deallocate(symp)
        end subroutine dealloc_mem_4_reordering

        subroutine alloc_mem_4_gpu_debug(ncomp)
          integer(kind = kint), intent(in) :: ncomp

#ifdef CUDA_DEBUG       
          allocate(sp_rlm_wk_debug(nnod_rlm*ncomp))
!          allocate(vr_rtm_wk_debug(nnod_rtm*ncomp))    
          
          if(ncomp .le. 0) return
!$omp parallel workshare
            sp_rlm_wk_debug(1:nnod_rlm*ncomp) = 0.0d0
!$omp end parallel workshare
#endif          
        end subroutine alloc_mem_4_gpu_debug
 
        subroutine clear_fwd_leg_work_debug(ncomp)
          integer(kind = kint), intent(in) :: ncomp
        
#ifdef CUDA_DEBUG       
          if(ncomp .le. 0) return
!$omp parallel workshare
            sp_rlm_wk_debug(1:nnod_rlm*ncomp) = 0.0d0
!$omp end parallel workshare
#endif          
        end subroutine clear_fwd_leg_work_debug

        subroutine dealloc_mem_4_gpu_debug(ncomp)

          integer(kind = kint), intent(in) :: ncomp
#ifdef CUDA_DEBUG       
          deallocate(sp_rlm_wk_debug)
!          deallocate(sp_rlm_wk, vr_rtm_wk)
#endif
        end subroutine dealloc_mem_4_gpu_debug

        subroutine set_mem_4_gpu
          call setPtrs(idx_gl_1d_rlm_j(1,1))  
          call cpy_schmidt_2_gpu(P_jl(1,1), dPdt_jl(1,1), P_rtm(1,1),   &
     &                           dPdt_rtm(1,1))

          call memcpy_h2d(lstack_rlm(0), a_r_1d_rlm_r(1),g_colat_rtm(1),&
     &                         g_sph_rlm(1,3), g_sph_rlm(1, 7),       &
     &                                         asin_theta_1d_rtm(1),    &
     &                         idx_gl_1d_rlm_j(1,1), radius_1d_rlm_r(1),&
     &                         weight_rtm(1), mdx_p_rlm_rtm(1),         &
     &                         mdx_n_rlm_rtm(1))
        end subroutine set_mem_4_gpu

        subroutine cpy_spectrum_dat_2_gpu(ncomp, sp_rlm)
          integer(kind = kint), intent(in) :: ncomp
          real(kind = kreal), intent(in) :: sp_rlm(ncomp*nnod_rlm)
          
          call set_spectrum_data(sp_rlm(1), ncomp)
        end subroutine cpy_spectrum_dat_2_gpu

        subroutine cpy_spectrum_dat_from_gpu(ncomp, sp_rlm)
          integer(kind = kint), intent(in) :: ncomp
          real(kind = kreal), intent(in) :: sp_rlm(ncomp*nnod_rlm)
          
          call retrieve_spectrum_data(sp_rlm(1), ncomp)
        end subroutine cpy_spectrum_dat_from_gpu
       
        subroutine cpy_physical_dat_from_gpu(ncomp, vr_rtm)
          integer(kind = kint), intent(in) :: ncomp
          real(kind = kreal), intent(in) :: vr_rtm(ncomp*nnod_rtm)
          
          call retrieve_physical_data(vr_rtm(1), ncomp)
        end subroutine cpy_physical_dat_from_gpu

        subroutine cpy_physical_dat_2_gpu(ncomp, vr_rtm)
          integer(kind = kint), intent(in) :: ncomp
          real(kind = kreal), intent(in) :: vr_rtm(ncomp*nnod_rtm)
          
          call set_physical_data(vr_rtm(1), ncomp)
        end subroutine cpy_physical_dat_2_gpu

        subroutine calypso_gpu_finalize
#ifdef CUDA_DEBUG
          call dealloc_mem_4_gpu_debug(ncomp_sph_trans)
#endif
          call cleangpu
        end subroutine calypso_gpu_finalize
  
        subroutine sync_device 
          call cuda_sync_device
        end subroutine sync_device 

        subroutine init_test_case

        use m_sph_spectr_data
        use m_sph_phys_address

        integer(kind=kint) :: k, j,l,m
        integer(kind=kint) :: inod 
       
! To view values using debugger
        real(kind=kreal) :: reg

!Find mode address 
        l = idx_gl_1d_rj_j(1,2)
        m = idx_gl_1d_rj_j(1,3)
        j = idx_gl_1d_rj_j(1,1)

        if ( j .gt. 0) then
          do k = nlayer_ICB, nlayer_CMB
              inod = local_sph_data_address(k,j)
              d_rj(inod, ipol%i_velo) = cos(radius_1d_rlm_r(k))
       !       write(*,*) d_rj(inod, ipol%i_velo) 
              d_rj(inod, idpdr%i_velo) = -1 * sin(radius_1d_rlm_r(k))
       !       write(*,*) d_rj(inod, idpdr%i_velo) 
              d_rj(inod, itor%i_velo) = 1
       !       write(*,*) d_rj(inod, itor%i_velo) 
              d_rj(inod, ipol%i_temp) = 1 
      !        write(*,*) d_rj(inod, ipol%i_temp) 
         end do
       end if
       end subroutine init_test_case
  
       subroutine cal_heatflux_4_test
       use m_addresses_trans_sph_MHD 
       use const_wz_coriolis_rtp
       use cal_products_smp

      if( (f_trns%i_h_flux*iflag_t_evo_4_temp) .gt. 0) then
        call cal_vec_scalar_prod_w_coef_smp(np_smp, nnod_rtp,           &
     &    inod_rtp_smp_stack, coef_temp, fld_rtp(1,b_trns%i_velo),      &
     &    fld_rtp(1,b_trns%i_temp), frc_rtp(1,f_trns%i_h_flux) )
      end if
       end subroutine cal_heatflux_4_test
       
      end module cuda_optimizations 

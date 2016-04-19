!>@file   legendre_transform_cuda.f90
!!@brief  module legendre_transform_cuda
!!
!!@author H. Matsui
!!@date Programmed in Aug., 2007
!!@n    Modified in Apr. 2013
!
!>@brief  Legendre transforms
!!       (Original version)
!!
!!
!!@verbatim
!!      subroutine leg_backward_trans_cuda                               &
!!     &         (ncomp, nvector, nscalar, n_WR, n_WS, WR, WS)
!!        Input:  sp_rlm   (Order: poloidal,diff_poloidal,toroidal)
!!        Output: vr_rtm   (Order: radius,theta,phi)
!!
!!    Forward transforms
!!      subroutine leg_forwawd_trans_cuda                                &
!!     &         (ncomp, nvector, nscalar, n_WR, n_WS, WR, WS)
!!        Input:  vr_rtm   (Order: radius,theta,phi)
!!        Output: sp_rlm   (Order: poloidal,diff_poloidal,toroidal)
!!
!!@endverbatim
!!
!!@param   ncomp    Total number of components for spherical transform
!!@param   nvector  Number of vector for spherical transform
!!@param   nscalar  Number of scalar (including tensor components)
!!                  for spherical transform
!
      module legendre_transform_cuda
!
      use m_precision
!
      implicit none
!
! -----------------------------------------------------------------------
!
      contains
!
! -----------------------------------------------------------------------
!
      subroutine leg_backward_trans_cuda                                 &
     &         (ncomp, nvector, nscalar, n_WR, n_WS, WR, WS)
!
      use m_work_4_sph_trans_spin
      use spherical_SRs_N
      use cuda_optimizations
      use legendre_bwd_trans_org 
      use calypso_mpi
!
      integer(kind = kint), intent(in) :: ncomp, nvector, nscalar
      integer(kind = kint), intent(in) :: n_WR, n_WS
      real (kind=kreal), intent(inout):: WR(n_WR)
      real (kind=kreal), intent(inout):: WS(n_WS)
!
!
      call calypso_rlm_from_recv_N(ncomp, n_WR, WR, sp_rlm_wk(1))
#if defined(CUDA_DEBUG)
      call clear_bwd_legendre_work(ncomp)
#endif
      
!
      call clear_field_data(ncomp)
#if defined(CUDA_TIMINGS)
      call start_eleps_time(57) 
#endif
      call cpy_spectrum_dat_2_gpu(ncomp, sp_rlm_wk(1)) 
#if defined(CUDA_TIMINGS)
      call sync_device
      call end_eleps_time(57) 
#endif

      if(nvector .gt. 0 .OR. nscalar .gt. 0) then
#if defined(CUDA_TIMINGS)
        call start_eleps_time(59) 
#endif
        call legendre_b_trans_cuda(ncomp, nvector, nscalar)
#if defined(CUDA_TIMINGS)
        call sync_device
        call end_eleps_time(59) 
#endif
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
          call legendre_b_trans_vector_org(ncomp, nvector, sp_rlm_wk(1) &
     &       , vr_rtm_wk(1))
#endif
      end if
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
      if(nscalar .gt. 0) then
        call legendre_b_trans_scalar_org                                &
     &     (ncomp, nvector, nscalar, sp_rlm_wk(1), vr_rtm_wk(1))
      end if
#endif

#if defined(CUDA_TIMINGS)
      call start_eleps_time(58) 
#endif
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
      call cpy_field_dev2host_4_debug(ncomp)
#else 
      call cpy_physical_dat_from_gpu(ncomp, vr_rtm_wk(1))
#endif
#if defined(CUDA_TIMINGS)
      call sync_device
      call end_eleps_time(58) 
#endif
!

#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
      call check_bwd_trans_cuda(my_rank, vr_rtm_wk(1), P_jl(1,1),     &
     &            dPdt_jl(1,1))
#endif
      call finish_send_recv_rj_2_rlm
      call calypso_rtm_to_send_N(ncomp, n_WS, vr_rtm_wk(1), WS(1))
!
      end subroutine leg_backward_trans_cuda
!
! -----------------------------------------------------------------------
!
      subroutine leg_backward_trans_cuda_org                            &
     &         (ncomp, nvector, nscalar, n_WR, n_WS, WR, WS)
!
      use m_work_4_sph_trans_spin
      use spherical_SRs_N
      use cuda_optimizations
      use legendre_bwd_trans_org 
      use calypso_mpi
!
      integer(kind = kint), intent(in) :: ncomp, nvector, nscalar
      integer(kind = kint), intent(in) :: n_WR, n_WS
      real (kind=kreal), intent(inout):: WR(n_WR)
      real (kind=kreal), intent(inout):: WS(n_WS)
!
!
      call calypso_rlm_from_recv_N(ncomp, n_WR, WR, sp_rlm_wk(1))
#if defined(CUDA_DEBUG)
      call clear_bwd_legendre_work(ncomp)
#endif
      
!
      call clear_field_data(ncomp)
#if defined(CUDA_TIMINGS)
      call start_eleps_time(57) 
#endif
      call cpy_spectrum_dat_2_gpu(ncomp, sp_rlm_wk(1)) 
#if defined(CUDA_TIMINGS)
      call sync_device
      call end_eleps_time(57) 
#endif

      if(nvector .gt. 0 .OR. nscalar .gt. 0) then
#if defined(CUDA_TIMINGS)
        call start_eleps_time(59) 
#endif
        call legendre_b_trans_cuda(ncomp, nvector, nscalar)
#if defined(CUDA_TIMINGS)
        call sync_device
        call end_eleps_time(59) 
#endif
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
          call legendre_b_trans_vector_org(ncomp, nvector, sp_rlm_wk(1) &
     &       , vr_rtm_wk(1))
#endif
      end if
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
      if(nscalar .gt. 0) then
        call legendre_b_trans_scalar_org                                &
     &     (ncomp, nvector, nscalar, sp_rlm_wk(1), vr_rtm_wk(1))
      end if
#endif

#if defined(CUDA_TIMINGS)
      call start_eleps_time(58) 
#endif
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
      call cpy_field_dev2host_4_debug(ncomp)
#else 
      call cpy_physical_dat_from_gpu(ncomp, vr_rtm_wk(1))
#endif
#if defined(CUDA_TIMINGS)
      call sync_device
      call end_eleps_time(58) 
#endif
!

#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
      call check_bwd_trans_cuda(my_rank, vr_rtm_wk(1), P_jl(1,1),     &
     &            dPdt_jl(1,1))
#endif
      call finish_send_recv_rj_2_rlm
      call calypso_rtm_to_send_N(ncomp, n_WS, vr_rtm_wk(1), WS(1))
!
      end subroutine leg_backward_trans_cuda_org
!
! -----------------------------------------------------------------------
!
      subroutine leg_forward_trans_cuda                                  &
     &         (ncomp, nvector, nscalar, n_WR, n_WS, WR, WS)
!
      use m_work_4_sph_trans_spin
      use spherical_SRs_N
      use legendre_fwd_trans_org 
      use cuda_optimizations
      use m_spheric_param_smp
      use set_vr_rtm_for_leg_cuda
!
      integer(kind = kint), intent(in) :: ncomp, nvector, nscalar
      integer(kind = kint), intent(in) :: n_WR, n_WS
      real (kind=kreal), intent(inout):: WR(n_WR)
      real (kind=kreal), intent(inout):: WS(n_WS)
      integer(kind = kint) :: mp_rlm, mn_rlm, mp_rlm_st, mp_rlm_rem
      integer(kind = kint) :: remainder, quotient, n
!
!
      
      call write2file(WR(1), n_WR, 'input', 5)
      call finish_send_recv_rtp_2_rtm

      call write2file_int(irev_sr_rtm(1), nnod_rtm, 'commC', 5)

!$omp parallel workshare
      WS(1:ncomp*ntot_item_sr_rlm) = 0.0d0
!$omp end parallel workshare
 
      call clear_spectrum_data(ncomp)
#ifdef CUDA_DEBUG
      call clear_fwd_legendre_work(ncomp)
      call clear_fwd_leg_work_debug(ncomp)
#endif
      if ( nvector .gt. 0 ) then
        quotient = nidx_rtm(3)/min(32, nidx_rtm(3))
        remainder = mod(nidx_rtm(3), min(32, nidx_rtm(3))) 
        call clear_memory_fwd_trans_reordering
        do n = 1, quotient
          do mp_rlm = 1+(n-1)*min(32, nidx_rtm(3)), min(32,nidx_rtm(3))*n
            mn_rlm = nidx_rtm(3) - mp_rlm + 1
            call set_vr_rtm_vector_cuda(nidx_rtm(1), nvector, n,        &
      &        mp_rlm, mn_rlm, ncomp, irev_sr_rtm, n_WR, WR,            &
      &        nvec_lk, symp_r(1), asmp_t(1), asmp_p(1),        &   
      &        symn_t(1), symn_p(1))
!            call set_vr_rtm_scalar_cuda(1, nkr(ip), mp_rlm,       &   
!      &        ncomp, nvector, irev_sr_rtm, n_WR, WR,                    & 
!      &        nscl_lk, symp(1))
          end do
          call write2file(symp_r(1), nvec_lk, 'symp_r.dat', 12)
          call sync_device
          call set_physical_data(symp_r(1), asmp_t(1),    &
      &      asmp_p(1), symn_t(1), symn_p(1), symp(1))
          call legendre_f_trans_vector_cuda(ncomp, nvector, nscalar,      &
      &             1+(n-1)*min(32, nidx_rtm(3)), n*min(32,nidx_rtm(3)))
!         call legendre_f_trans_scalar_cuda_(ncomp, nvector)
        end do
      end if
         
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
      call calypso_rtm_from_recv_N(ncomp, n_WR, WR, vr_rtm_wk(1))
      n = 16
      call write2file(vr_rtm_wk(1), nnod_rtm, 'rtp', 3)
      if ( nvector .gt. 0 ) then
          call legendre_f_trans_vector_org(ncomp, nvector, vr_rtm_wk(1) &
     &       , sp_rlm_wk(1))
      end if

      if(nscalar .gt. 0) then
!        call legendre_f_trans_scalar_cuda(ncomp, nvector, nscalar,      &
!     &                                                  1, nidx_rtm(1)) 
        call legendre_f_trans_scalar_org                                &
     &     (ncomp, nvector, nscalar, vr_rtm_wk(1), sp_rlm_wk(1))
      end if
#endif

        call retrieve_spectrum_data(sp_rlm_wk_debug(1), ncomp)
        call check_fwd_trans_cuda_and_org(my_rank, sp_rlm_wk_debug(1), &
       &                   sp_rlm_wk(1))
         
!#if defined(CUDA_TIMINGS)
!        call start_eleps_time(65) 
!#endif
!#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
!         call cpy_spec_dev2host_4_debug(ncomp)
!#elif defined(CUDA_OPTIMIZED)
!#endif
!#if defined(CUDA_TIMINGS)
!        call sync_device
!        call end_eleps_time(65) 
!#endif
!

!#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
!        call check_fwd_trans_cuda(my_rank, sp_rlm_wk(1))
!#endif
!
      call finish_send_recv_rtp_2_rtm
      call calypso_rlm_to_send_N(ncomp, n_WS, sp_rlm_wk(1), WS)
!
      end subroutine leg_forward_trans_cuda
!
! -----------------------------------------------------------------------
!
      subroutine leg_forward_trans_cuda_and_org                             &
     &         (ncomp, nvector, nscalar, n_WR, n_WS, WR, WS)
!
      use m_work_4_sph_trans_spin
      use spherical_SRs_N
      use legendre_fwd_trans_org 
      use cuda_optimizations
      use m_spheric_param_smp
!
      integer(kind = kint), intent(in) :: ncomp, nvector, nscalar
      integer(kind = kint), intent(in) :: n_WR, n_WS
      real (kind=kreal), intent(inout):: WR(n_WR)
      real (kind=kreal), intent(inout):: WS(n_WS)
!
!
      call calypso_rtm_from_recv_N(ncomp, n_WR, WR, vr_rtm_wk(1))
      call clear_fwd_legendre_work(ncomp)
#if defined(CUDA_DEBUG)
      call clear_fwd_leg_work_debug(ncomp)
#endif
!
      call clear_spectrum_data(ncomp)
#if defined(CUDA_TIMINGS)
      call start_eleps_time(64) 
#endif
      call cpy_physical_dat_2_gpu(ncomp, vr_rtm_wk(1)) 
#if defined(CUDA_TIMINGS)
      call sync_device
      call end_eleps_time(64)
#endif

      if(nvector .gt. 0) then
#if defined(CUDA_TIMINGS)
        call start_eleps_time(60)
#endif
        call legendre_f_trans_vector_cuda(ncomp, nvector, nscalar,      &
     &                        nidx_rtm(1)/4, nidx_rtm(1)) 
        call legendre_f_trans_vector_org_4_cuda(ncomp, nvector,         &
     &          vr_rtm_wk(1), sp_rlm_wk(1), nidx_rtm(1)/4)
#if defined(CUDA_TIMINGS)
        call sync_device
        call end_eleps_time(60)
#endif
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
          call legendre_f_trans_vector_org(ncomp, nvector, vr_rtm_wk(1) &
     &       , sp_rlm_wk_debug(1))
#endif
      end if

      if(nscalar .gt. 0) then
#if defined(CUDA_TIMINGS)
        call start_eleps_time(68) 
#endif
        call legendre_f_trans_scalar_cuda(ncomp, nvector, nscalar,      &
                                  max(2,nidx_rtm(1)/4), nidx_rtm(1)) 
        call legendre_f_trans_scalar_org_4_cuda                         &
     &     (ncomp, nvector, nscalar, vr_rtm_wk(1), sp_rlm_wk(1),        &
     &      max(2,nidx_rtm(1)/4))
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
        call legendre_f_trans_scalar_org                                &
     &     (ncomp, nvector, nscalar, vr_rtm_wk(1), sp_rlm_wk_debug(1))
#endif
#if defined(CUDA_TIMINGS)
        call sync_device
        call end_eleps_time(68) 
#endif
      end if

#if defined(CUDA_TIMINGS)
        call start_eleps_time(65) 
#endif
        call retrieve_spectrum_data_cuda_and_org(sp_rlm_wk(1),    &
     &              ncomp, max(2,nidx_rtm(1)/4), nidx_rtm(1))
#if defined(CUDA_TIMINGS)
        call sync_device
        call end_eleps_time(65) 
#endif
!

#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
        call check_fwd_trans_cuda_and_org(my_rank, sp_rlm_wk_debug(1), &
       &                   sp_rlm_wk(1))
#endif
!
      call finish_send_recv_rtp_2_rtm
      call calypso_rlm_to_send_N(ncomp, n_WS, sp_rlm_wk(1), WS)
!
      end subroutine leg_forward_trans_cuda_and_org

! -----------------------------------------------------------------------
      subroutine leg_forward_trans_cublas                               &
     &         (ncomp, nvector, nscalar, n_WR, n_WS, WR, WS)
      use m_work_4_sph_trans_spin
      use spherical_SRs_N
      use cuda_optimizations
      use m_spheric_param_smp
      use legendre_fwd_trans_org 
!
      integer(kind = kint), intent(in) :: ncomp, nvector, nscalar
      integer(kind = kint), intent(in) :: n_WR, n_WS
      real (kind=kreal), intent(inout):: WR(n_WR)
      real (kind=kreal), intent(inout):: WS(n_WS)
!
!
#ifdef CUBLAS
      call calypso_rtm_from_recv_N(ncomp, n_WR, WR, vr_rtm_wk(1))
#if defined(CUDA_DEBUG)
      call clear_fwd_legendre_work(ncomp)
#endif
!
      call clear_spectrum_data(ncomp)
#if defined(CUDA_TIMINGS)
      call start_eleps_time(64) 
#endif
      call cpy_physical_dat_2_gpu(ncomp, vr_rtm_wk(1)) 
#if defined(CUDA_TIMINGS)
      call sync_device
      call end_eleps_time(64)
#endif

      if(nvector .gt. 0) then
#if defined(CUDA_TIMINGS)
        call start_eleps_time(60)
#endif
        call legendre_f_trans_vector_cublas(ncomp, nvector, nscalar) 
#if defined(CUDA_TIMINGS)
        call sync_device
        call end_eleps_time(60)
#endif
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
          call legendre_f_trans_vector_org(ncomp, nvector, vr_rtm_wk(1) &
     &       , sp_rlm_wk(1))
#endif
      end if

      if(nscalar .gt. 0) then
#if defined(CUDA_TIMINGS)
        call start_eleps_time(68) 
#endif
        call legendre_f_trans_scalar_cuda(ncomp, nvector, nscalar,      &
     &                                                  1, nidx_rtm(1)) 
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
        call legendre_f_trans_scalar_org                                &
     &     (ncomp, nvector, nscalar, vr_rtm_wk(1), sp_rlm_wk(1))
#endif
#if defined(CUDA_TIMINGS)
        call sync_device
        call end_eleps_time(68) 
#endif
      end if

#if defined(CUDA_TIMINGS)
        call start_eleps_time(65) 
#endif
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
         call cpy_spec_dev2host_4_debug(ncomp)
#elif defined(CUDA_OPTIMIZED)
        call cpy_spectrum_dat_from_gpu(ncomp, sp_rlm_wk(1))
#endif
#if defined(CUDA_TIMINGS)
        call sync_device
        call end_eleps_time(65) 
#endif
!

#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
        call check_fwd_trans_cuda(my_rank, sp_rlm_wk(1))
#endif
!
      call finish_send_recv_rtp_2_rtm
      call calypso_rlm_to_send_N(ncomp, n_WS, sp_rlm_wk(1), WS)
#endif
!
      end subroutine leg_forward_trans_cublas

      subroutine leg_forward_trans_cub                               &
     &         (ncomp, nvector, nscalar, n_WR, n_WS, WR, WS)
      use m_work_4_sph_trans_spin
      use spherical_SRs_N
      use cuda_optimizations
      use m_spheric_param_smp
      use legendre_fwd_trans_org 
!
      integer(kind = kint), intent(in) :: ncomp, nvector, nscalar
      integer(kind = kint), intent(in) :: n_WR, n_WS
      real (kind=kreal), intent(inout):: WR(n_WR)
      real (kind=kreal), intent(inout):: WS(n_WS)
!
!
#ifdef CUB
      call calypso_rtm_from_recv_N(ncomp, n_WR, WR, vr_rtm_wk(1))
#if defined(CUDA_DEBUG)
      call clear_fwd_legendre_work(ncomp)
#endif
!
      call clear_spectrum_data(ncomp)
#if defined(CUDA_TIMINGS)
      call start_eleps_time(64) 
#endif
      call cpy_physical_dat_2_gpu(ncomp, vr_rtm_wk(1)) 
#if defined(CUDA_TIMINGS)
      call sync_device
      call end_eleps_time(64)
#endif

      if(nvector .gt. 0) then
#if defined(CUDA_TIMINGS)
        call start_eleps_time(60)
#endif
        call legendre_f_trans_vector_cub(ncomp, nvector, nscalar) 
#if defined(CUDA_TIMINGS)
        call sync_device
        call end_eleps_time(60)
#endif
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
          call legendre_f_trans_vector_org(ncomp, nvector, vr_rtm_wk(1) &
     &       , sp_rlm_wk(1))
#endif
      end if

      if(nscalar .gt. 0) then
#if defined(CUDA_TIMINGS)
        call start_eleps_time(68) 
#endif
        call legendre_f_trans_scalar_cuda(ncomp, nvector, nscalar,      &
     &                                             1, nidx_rtm(1))
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
        call legendre_f_trans_scalar_org                                &
     &     (ncomp, nvector, nscalar, vr_rtm_wk(1), sp_rlm_wk(1))
#endif
#if defined(CUDA_TIMINGS)
        call sync_device
        call end_eleps_time(68) 
#endif
      end if

#if defined(CUDA_TIMINGS)
        call start_eleps_time(65) 
#endif
#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
         call cpy_spec_dev2host_4_debug(ncomp)
#elif defined(CUDA_OPTIMIZED)
        call cpy_spectrum_dat_from_gpu(ncomp, sp_rlm_wk(1))
#endif
#if defined(CUDA_TIMINGS)
        call sync_device
        call end_eleps_time(65) 
#endif
!

#if defined(CUDA_DEBUG) || defined(CHECK_SCHMIDT_OTF)
        call check_fwd_trans_cuda(my_rank, sp_rlm_wk(1))
#endif
!
      call finish_send_recv_rtp_2_rtm
      call calypso_rlm_to_send_N(ncomp, n_WS, sp_rlm_wk(1), WS)
#endif
!
      end subroutine leg_forward_trans_cub
      end module legendre_transform_cuda

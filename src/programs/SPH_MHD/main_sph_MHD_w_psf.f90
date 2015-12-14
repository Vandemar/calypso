!>@file   main_sph_MHD_w_psf.f90
!!@brief  program kemorin_sph_MHD
!!
!!@author H. Matsui
!!@date Programmed by H. Okuda in 2000
!!@n    Modified by H. Matsui in May, 2003 (ver 2.0)
!!@n    Connect to vizs  by H. Matsui in July 2006 (ver 2.0)
!
!>@brief  Main program for MHD dynamo simulation
!
      program kemorin_sph_MHD
!
      use m_precision
!
      use calypso_mpi
      use analyzer_sph_MHD_w_psf
#ifdef CUDA
      use cuda_optimizations
#endif
      use m_work_time
!
      implicit none
      integer (kind=kint) :: x
 
      x=0

!
!
      call calypso_MPI_init
!
#ifdef MPI_DEBUG 
      do 
        if (x .ne. my_rank) then
           write(*,*) "DEbug"
           EXIT
        endif
      end do 
#endif
!
#ifdef CUDA
      call calypso_GPU_init
#endif
!
      call initialize_sph_mhd_w_psf
#ifndef CUDA_DEBUG
      call evolution_sph_mhd_w_psf
#endif
!
#ifdef CUDA
      call calypso_GPU_finalize
#endif
!
      call calypso_MPI_finalize
!
      stop
      end program kemorin_sph_MHD

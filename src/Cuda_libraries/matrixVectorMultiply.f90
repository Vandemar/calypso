! Author: Harsha Lokavarapu

        module matrix_vector_multiply

        implicit none
!
!>  kind parameter for integer type variable (  4 byte)
      integer, parameter   ::  kint  =       4
!>  kind parameter for real    type variable (  8 byte )
      integer, parameter   ::  kreal =       8

!
!----------------------------------------------------------------
!
        contains
!
!---------------------------------------------------------------
!
        subroutine double_matrix_vector_multiply(A, x, y, n)       
!  Fortran90 program to multiply matrix and vector using dgemv
!   y = alpha*A*x + beta*y

! Define
        real(kind=kreal), dimension(:,:), intent(in) ::    A
        real(kind=kreal), dimension(:), intent(in) ::    x
        real(kind=kreal), dimension(:), allocatable, intent(out) ::    y
        real(kind=kreal) :: alpha=1, beta=0
        integer, intent(in) :: n

        allocate(y(n))
        y=0
  
        ! Execute cublas function
        call CUBLAS_DGEMV('n',n,n,alpha,A,n,x,1,beta,y,1)   

        end subroutine double_matrix_vector_multiply

! Fortran90 program to multiply matrix and vector using Sgemv
!   y = alpha*A*x + beta*y
! Single precision 

        subroutine single_matrix_vector_multiply(A, x, y, n)

! Define
        real(kind=kint), dimension(:,:), intent(in) ::    A
        real(kind=kint), dimension(:), intent(in) ::    x
        real(kind=kint), dimension(:), allocatable, intent(out) ::    y
        real(kind=kint) :: alpha=1, beta=0
        integer, intent(in) :: n

        allocate(y(n))
        y=0

! Execute cublas function
        call cublas_SGEMV('n',n,n,alpha,A,n,x,1,beta,y,1)   

        end subroutine single_matrix_vector_multiply

end module matrix_vector_multiply

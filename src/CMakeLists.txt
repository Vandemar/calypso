# Recurse into the subdirectories
IF(CUDA_FOUND)
  ADD_SUBDIRECTORY (Cuda_libraries)
#So that it is a global variable
  FILE(GLOB FORTRAN_CUDA_INTERFACE Cuda_libraries/*.f90)
  SET(CUDA_FORTRAN_SOURCE_CODE ${FORTRAN_CUDA_INTERFACE})
ENDIF(CUDA_FOUND)

ADD_SUBDIRECTORY (External_libs)
SET (FFTW_LIBRARIES ${FFTW_LIBRARIES} fftpack.5d)
ADD_SUBDIRECTORY (Fortran_libraries)
ADD_SUBDIRECTORY (programs)


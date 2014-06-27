# Recurse into the subdirectories
ADD_SUBDIRECTORY (External_libs)
ADD_SUBDIRECTORY (Fortran_libraries)
ADD_SUBDIRECTORY (programs)

IF (CUDA_FOUND)
  ADD_SUBDIRECTORY (Cuda_libraries)
ENDIF (CUDA_FOUND)

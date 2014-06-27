# Define source files in MHD_src and its subdirectories
# LA ---> Linear Algebra
FILE(GLOB CUDA_LA_SRC_FILES *.[fF]90)

ADD_LIBRARY(cudaLA ${CUDA_LA_SRC_FILES})
ADD_DEPENDENCIES(cudaLA iFortran)
TARGET_LINK_LIBRARIES(cudaLA ${CMAKE_BINARY_DIR}/include/iFortran.o ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDART_LIBRARY})



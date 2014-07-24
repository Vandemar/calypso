# Define source files in MHD_src and its subdirectories
# LA ---> Linear Algebra
FILE(GLOB CUDA_SRC_FILES *.cu)

set(cudaLib_objDir ${CMAKE_BINARY_DIR}/include/cudaLib.o)

#set(CUDA_NVCC_FLAGS "-c -L${CUDA_LIBRARIES} -I${CUDA_INCLUDE_DIRS} -lcuda
#-lcublas -lcudart")
#CUDA_ADD_LIBRARY(cudaLib ${CUDA_SRC_FILES})

add_custom_target(cudaLib command ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc -c ${CUDA_SRC_FILES}
	-L${CUDA_LIBRARIES}
	-I${CUDA_INCLUDE_DIRS}
        -o ${cudaLib_objDir})

SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${cudaLib_objDir}") 

#TARGET_LINK_LIBRARIES(cudaLib ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDART_LIBRARY})
#SET(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} cudaLib)


#include <stdlib.h>
#include <cublas_v2.h>
#include <string.h>

static const char *_cublasGetErrorEnum(cublasStatus_t error) 
{
  switch (error)
  {
    case CUBLAS_STATUS_SUCCESS:
      return "cublasSuccess";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "cublas status not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "cublas alloc failed";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "check the parameters to the function call";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "access to gpu memory failed";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "The gpu kernel launch failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "Cublas status internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "The functionality is not supported";
   }
    
   return "<unknown>";
}

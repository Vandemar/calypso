#include "cuda.h"

__global__ void prefetchASMREG( double *data, int offset ){

data += offset;

asm("prefetch.global.L2 [%0];"::"l"(data) );

data[0] = 1.0f;

}

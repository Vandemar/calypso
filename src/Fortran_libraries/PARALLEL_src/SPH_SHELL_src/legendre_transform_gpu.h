#include "cuda.h"

class LegendreTransform {
public:
  __host__ void initialize_gpu();
  __host__ void finalize_gpu();
};

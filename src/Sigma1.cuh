
#include "ROTR.cuh"

#ifndef SHAONGPU_SIGMA1_CUH
#define SHAONGPU_SIGMA1_CUH

__host__ __device__ __inline__ u_int32_t Sigma1(const u_int32_t x) {
    return ROTR<6>(x) ^ ROTR<11>(x) ^ ROTR<25>(x);
}

#endif //SHAONGPU_SIGMA1_CUH

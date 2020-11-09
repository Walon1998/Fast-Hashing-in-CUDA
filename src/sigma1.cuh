
#include "ROTR.cuh"
#include "SHR.cuh"

#ifndef SHAONGPU_sIGMA1_CUH
#define SHAONGPU_sIGMA1_CUH

__host__ __device__ __inline__ u_int32_t sigma1(const u_int32_t x) {
    return ROTR<17>(x) ^ ROTR<19>(x) ^ SHR<10>(x);
}

#endif //SHAONGPU_sIGMA1_CUH

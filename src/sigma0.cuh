
#include <assert.h>
#include "ROTR.cuh"
#include "SHR.cuh"

#ifndef SHAONGPU_sIGMA0_CUH
#define SHAONGPU_sIGMA0_CUH

__host__ __device__ __inline__ u_int32_t sigma0(const u_int32_t x) {
    return ROTR<7>(x) ^ ROTR<18>(x) ^ SHR<3>(x);
}

#endif //SHAONGPU_sIGMA0_CUH

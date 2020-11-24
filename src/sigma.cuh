//
// Created by nevil on 24.11.2020.
//

#include <assert.h>
#include "ROTR.cuh"
#include "SHR.cuh"


#ifndef SHA_ON_GPU_SIGMA_CUH
#define SHA_ON_GPU_SIGMA_CUH

__host__ __device__ __inline__ uint32_t sigma0(const uint32_t x) {
    return ROTR<7>(x) ^ ROTR<18>(x) ^ SHR<3>(x);
}

__host__ __device__ __inline__ uint32_t sigma1(const uint32_t x) {
    return ROTR<17>(x) ^ ROTR<19>(x) ^ SHR<10>(x);
}

__host__ __device__ __inline__ uint32_t Sigma1(const uint32_t x) {
    return ROTR<6>(x) ^ ROTR<11>(x) ^ ROTR<25>(x);
}

__host__ __device__ __inline__ uint32_t Sigma0(const uint32_t x) {
    return ROTR<2>(x) ^ ROTR<13>(x) ^ ROTR<22>(x);
}


#endif //SHA_ON_GPU_SIGMA_CUH

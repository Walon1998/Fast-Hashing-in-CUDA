//
// Created by neville on 03.11.20.
//
#include "ROTR.cuh"

#ifndef SHAONGPU_SIGMA0_CUH
#define SHAONGPU_SIGMA0_CUH

__host__ __device__ __inline__ u_int32_t Sigma0(const u_int32_t x) {
    return ROTR<2>(x) ^ ROTR<13>(x) ^ ROTR<22>(x);
}

#endif //SHAONGPU_SIGMA0_CUH

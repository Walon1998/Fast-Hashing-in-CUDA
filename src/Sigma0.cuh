//
// Created by neville on 03.11.20.
//
#include <assert.h>

#ifndef SHAONGPU_SIGMA0_CUH
#define SHAONGPU_SIGMA0_CUH

__host__ __device__ __inline__ u_int32_t Sigma0(const u_int32_t x) {
    //return ROTR<2>(x) ^ ROTR<13>(x) ^ ROTR<22>(x);
    return 0;
}

__host__ __device__ void Sigma0_test() {

    u_int32_t x = 0x00000000;
    u_int32_t res = Sigma0(x);
    assert(res == 0x00000000);

    x = 0xfffffff;
    res = Sigma0(x);
    assert(res == 0xffffffff);

    x = 0x00000001;
    res = Sigma0(x);
    assert(res == 0x40080400);

}


#endif //SHAONGPU_SIGMA0_CUH

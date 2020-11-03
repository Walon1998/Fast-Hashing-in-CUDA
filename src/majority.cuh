//
// Created by neville on 03.11.20.
//

#include <assert.h>

#ifndef SHAONGPU_MAJORITY_CUH
#define SHAONGPU_MAJORITY_CUH

__host__ __device__ __inline__ u_int32_t maj(const u_int32_t x, const u_int32_t y, const u_int32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__host__ __device__ void maj_test() {
    u_int32_t x = 0;
    u_int32_t y = 0;
    u_int32_t z = 0;

    u_int32_t res = maj(x, y, z);
    assert(res == 0);

    x = 0x00000001;
    y = 0x00000001;
    z = 0x00000001;

    res = maj(x, y, z);
    assert(res == 0x00000001);

    x = 0x00000000;
    y = 0x00000001;
    z = 0x00000001;

    res = maj(x, y, z);
    assert(res == 0x00000001);


    x = 0x00000000;
    y = 0x00000000;
    z = 0x00000001;

    res = maj(x, y, z);
    assert(res == 0x00000000);


    x = 0x10000000;
    y = 0x00000000;
    z = 0x10000000;

    res = maj(x, y, z);
    assert(res == 0x10000000);


    x = 0x000f0000;
    y = 0x000f0000;
    z = 0x00000000;

    res = maj(x, y, z);
    assert(res == 0x000f0000);

}

#endif //SHAONGPU_MAJORITY_CUH

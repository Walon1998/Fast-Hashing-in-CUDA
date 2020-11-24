//
// Created by neville on 03.11.20.
//
#include <assert.h>

#ifndef SHAONGPU_CHOOSE_CUH
#define SHAONGPU_CHOOSE_CUH

__host__ __device__ __inline__ uint32_t ch(const uint32_t x, const uint32_t y, const uint32_t z) {

    return (x & y) ^ (~x & z);
}

__host__ __device__ void ch_test() {

    uint32_t x = 0x00000001;
    uint32_t y = 0x00000001;
    uint32_t z = 0x00000001;

    uint32_t res = ch(x, y, z);
    assert(res == 0x00000001);

    x = 0x00000001;
    y = 0x00000001;
    z = 0x00000000;

    res = ch(x, y, z);
    assert(res == 0x00000001);


    x = 0x00000001;
    y = 0x00000000;
    z = 0x00000001;

    res = ch(x, y, z);
    assert(res == 0x00000000);

    x = 0xffffffff;
    y = 0x12345678;
    z = 0x87654321;

    res = ch(x, y, z);
    assert(res == y);

    x = 0x00000000;
    y = 0x12345678;
    z = 0x87654321;

    res = ch(x, y, z);
    assert(res == z);

}


#endif //SHAONGPU_CHOOSE_CUH

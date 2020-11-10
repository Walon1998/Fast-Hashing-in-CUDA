
#include <assert.h>

#ifndef SHAONGPU_ROTR_CUH
#define SHAONGPU_ROTR_CUH

template<unsigned int N>
__host__ __device__ __inline__ u_int32_t ROTR(const u_int32_t x) {
    return (x >> N) | (x << (32 - N));
};

__host__ __device__ void ROTR_test() {
    int res;
    res = ROTR<0>(0);
    assert(res == 0);

    res = ROTR<0>(34);
    assert(res == 34);

    res = ROTR<1>(1);
    assert(res == 0x80000000);

    res = ROTR<1>(0x1010c0c0);
    assert(res == 0x08086060);

    res = ROTR<0>(0xffffffff);
    assert(res == 0xffffffff);

    res = ROTR<4>(0xffffffff);
    assert(res == 0xffffffff);

    res = ROTR<28>(0xffffffff);
    assert(res == 0xffffffff);

    res = ROTR<31>(0xffffffff);
    assert(res == 0xffffffff);
}

#endif //SHAONGPU_ROTR_CUH

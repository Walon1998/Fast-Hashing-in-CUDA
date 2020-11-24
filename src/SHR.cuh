
#include <assert.h>

#ifndef SHAONGPU_SHR_CUH
#define SHAONGPU_SHR_CUH

template<unsigned int N>
__host__ __device__ __inline__ uint32_t SHR(const uint32_t x) {
    return x >> N;
};

__host__ __device__ void SHR_test() {
    int res = SHR<0>(0);
    assert(res == 0);
    res = SHR<23>(0);
    assert(res == 0);
    res = SHR<32>(0);
    assert(res == 0);

    res = SHR<0>(0xffffffff);
    assert(res == 0xffffffff);

    res = SHR<4>(0xffffffff);
    assert(res == 0x0fffffff);

    res = SHR<28>(0xffffffff);
    assert(res == 0x0000000f);

    res = SHR<31>(0xffffffff);
    assert(res == 1);

    res = SHR<1>(3);
    assert(res == 1);
}

#endif //SHAONGPU_ROTL_CUH

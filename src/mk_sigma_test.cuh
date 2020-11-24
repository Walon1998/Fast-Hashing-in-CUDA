
#include <assert.h>
#include "sigma.cuh"

#ifndef SHAONGPU_MK_SIGMA_TEST_CUH
#define SHAONGPU_MK_SIGMA_TEST_CUH

template<uint32_t(*s)(const uint32_t)>
__host__ __device__ void mk_sigma_test(uint32_t i, uint32_t one) {

    uint32_t x = 0;
    uint32_t res = s(x);
    assert(res == 0);

    x = 0xffffffff;
    res = s(x);
    assert(res == 0xffffffff >> i);

    x = 1;
    res = s(x);
    assert(res == one);
}

template<uint32_t(*S)(const uint32_t)>
__host__ __device__ void mk_Sigma_test(uint32_t one) {
    mk_sigma_test<S>(0, one);
}

#endif // SHAONGPU_MK_SIGMA_TEST_CUH

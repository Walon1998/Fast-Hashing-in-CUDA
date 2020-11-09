
#include <assert.h>
#include "Sigma0.cuh"

#ifndef SHAONGPU_MK_SIGMA_TEST_CUH
#define SHAONGPU_MK_SIGMA_TEST_CUH

template <u_int32_t(*s)(const u_int32_t)>
__host__ __device__ void mk_sigma_test(u_int32_t i, u_int32_t one) {

    u_int32_t x = 0;
    u_int32_t res = s(x);
    assert(res == 0);

    x = 0xffffffff;
    res = s(x);
    assert(res == 0xffffffff >> i);

    x = 1;
    res = s(x);
    assert(res == one);
}

template <u_int32_t(*S)(const u_int32_t)>
__host__ __device__ void mk_Sigma_test(u_int32_t one) {
  mk_sigma_test<S>(0, one);
}

#endif // SHAONGPU_MK_SIGMA_TEST_CUH

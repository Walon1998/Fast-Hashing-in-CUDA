//
// Created by neville on 14.11.20.
//

#ifndef SHA_ON_GPU_LAMBDA_H
#define SHA_ON_GPU_LAMBDA_H

u_int64_t lambda(const u_int32_t i, const u_int32_t m, const u_int32_t n, const u_int32_t l) {
    return std::pow(2, i - 1)*(2 * n - 2 * m - l);
}

#endif //SHA_ON_GPU_LAMBDA_H

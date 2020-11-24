//
// Created by neville on 14.11.20.
//

#ifndef SHA_ON_GPU_DELTA_H
#define SHA_ON_GPU_DELTA_H

uint64_t delta(const uint32_t i, const uint32_t m, const uint32_t n, const uint32_t l) {
    return std::pow(2, i) * (2 * n - 2 * m - l);
}

#endif //SHA_ON_GPU_DELTA_H

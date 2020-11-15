//
// Created by neville on 04.11.20.
//

#ifndef SHA_ON_GPU_PARSHA256_SHA256_H
#define SHA_ON_GPU_PARSHA256_SHA256_H

#include "choose.cuh"
#include "majority.cuh"
#include "sigma0.cuh"
#include "Sigma0.cuh"
#include "sigma1.cuh"
#include "Sigma1.cuh"
#include <vector>

__device__ __host__ void parsha256_sha256(const int *__restrict__ in8, const int *__restrict__ in16, const int *__restrict__ in24, int *__restrict__ out) {

    constexpr u_int32_t K[64] =
            {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
             0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
             0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
             0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
             0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
             0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
             0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
             0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

    // Initial Hash values are first 8 ints of input
    u_int32_t H0 = in8[0];
    u_int32_t H1 = in8[1];
    u_int32_t H2 = in8[2];
    u_int32_t H3 = in8[3];
    u_int32_t H4 = in8[4];
    u_int32_t H5 = in8[5];
    u_int32_t H6 = in8[6];
    u_int32_t H7 = in8[7];


    register int W[64];


    // File Message Schedule
#pragma unroll
    for (int j = 0; j < 8; j++) {
        W[j] = in16[j];
    }

#pragma unroll
    for (int j = 8; j < 16; j++) {
        W[j] = in24[j];
    }

    // Only 16 values of the message schedule are used at the same time
    // it would be possible to integrate this loop in the next one to save registers.
    // A smart compiler would be able to do this on his own.
    // If our does uses a lot of registers (> 80), we have to do this on our own.
    // Otherwise this kernel should use only about 30 registers
#pragma unroll
    for (int j = 16; j < 64; j++) {
        W[j] = sigma1(W[j - 2])
               + W[j - 7]
               + sigma0(W[j - 15])
               + W[j - 16];
    }

    // Initial Hash values
    u_int32_t a = H0;
    u_int32_t b = H1;
    u_int32_t c = H2;
    u_int32_t d = H3;
    u_int32_t e = H4;
    u_int32_t f = H5;
    u_int32_t g = H6;
    u_int32_t h = H7;

#pragma unroll
    for (int j = 0; j < 64; j++) {

        const int T1 = h + Sigma1(e) + ch(e, f, g) + K[j] + W[j];
        const int T2 = Sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    H0 += a;
    H1 += b;
    H2 += c;
    H3 += d;
    H4 += e;
    H5 += f;
    H6 += g;
    H7 += h;


    out[0] = H0;
    out[1] = H1;
    out[2] = H2;
    out[3] = H3;
    out[4] = H4;
    out[5] = H5;
    out[6] = H6;
    out[7] = H7;
}

#endif //SHA_ON_GPU_PARSHA256_SHA256_H

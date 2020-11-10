//
// Created by neville on 04.11.20.
//

#ifndef SHA_ON_GPU_MAIN_LOOP_CPU_H
#define SHA_ON_GPU_MAIN_LOOP_CPU_H

#include "choose.cuh"
#include "majority.cuh"
#include "sigma0.cuh"
#include "Sigma0.cuh"
#include "sigma1.cuh"
#include "Sigma1.cuh"
#include <vector>

// return vector is currently passed by value, could be optimized
std::vector<int> main_loop_cpu(const std::vector<int> in) {
    std::vector<int> W(64);

    constexpr u_int32_t K[64] =
            {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
             0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
             0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
             0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
             0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
             0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
             0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
             0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

    // Initial Hash values
    u_int32_t H0 = 0x6a09e667;
    u_int32_t H1 = 0xbb67ae85;
    u_int32_t H2 = 0x3c6ef372;
    u_int32_t H3 = 0xa54ff53a;
    u_int32_t H4 = 0x510e527f;
    u_int32_t H5 = 0x9b05688c;
    u_int32_t H6 = 0x1f83d9ab;
    u_int32_t H7 = 0x5be0cd19;


    for (int i = 0; i < in.size(); i += 16) {

        // File Message Schedule
        for (int j = 0; j < 16; j++) {
            W[j] = in[i + j];
        }
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


        for (int j = 0; j < 64; j++) {
            const u_int32_t T1 = h + Sigma1(e) + ch(e, f, g) + K[j] + W[j];
            const u_int32_t T2 = Sigma0(a) + maj(a, b, c);
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


    }

    std::vector<int> result(8);
    result[0] = H0;
    result[1] = H1;
    result[2] = H2;
    result[3] = H3;
    result[4] = H4;
    result[5] = H5;
    result[6] = H6;
    result[7] = H7;
    return result;
}

#endif //SHA_ON_GPU_MAIN_LOOP_CPU_H

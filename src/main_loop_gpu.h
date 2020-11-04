//
// Created by neville on 04.11.20.
//

#ifndef SHA_ON_GPU_MAIN_LOOP_GPU_H
#define SHA_ON_GPU_MAIN_LOOP_GPU_H

#include <vector>
#include "choose.cuh"
#include "Sigma0.cuh"
#include "majority.cuh"

__global__ void main_loop_gpu(const int *__restrict__ in, const int N, int *__restrict__ out) {

    // Initial Hash values
    int H0 = 0x6a09e667;
    int H1 = 0xbb67ae85;
    int H2 = 0x3c6ef372;
    int H3 = 0xa54ff53a;
    int H4 = 0x510e527f;
    int H5 = 0x9b05688c;
    int H6 = 0x1f83d9ab;
    int H7 = 0x5be0cd19;

    register int message_schedule[64];

    for (int i = 0; i < N; i += 16) {

        // File Message Schedule
#pragma unroll
        for (int j = 0; j < 16; j++) {
            message_schedule[j] = in[i + j];
        }

        // Only 16 values of the message schedule are used at the same time
        // it would be possible to integrate this loop in the next one to save registers.
        // A smart compiler would be able to do this on his own.
        // If our does uses a lot of registers (> 80), we have to do this on our own.
        // Otherwise this kernel should use only about 30 registers
#pragma unroll
        for (int j = 16; j < 64; j++) {
            message_schedule[j] = sigma_0(message_schedule[j - 2])
                                  + message_schedule[j - 7]
                                  + sigma_0(message_schedule[j - 15])
                                  + message_schedule[j - 16];
        }

        // Initial Hash values
        int a = H0;
        int b = H1;
        int c = H2;
        int d = H3;
        int e = H4;
        int f = H5;
        int g = H6;
        int h = H7;

#pragma unroll
        for (int j = 0; j < 64; j++) {

            const int T1 = h + Sigma1(e) + ch(e, f, g) + K[j] + message_schedule[j];
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


    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = H0;
        out[1] = H1;
        out[2] = H2;
        out[3] = H3;
        out[4] = H4;
        out[5] = H5;
        out[6] = H6;
        out[7] = H7;
    }
}

#endif //SHA_ON_GPU_MAIN_LOOP_GPU_H

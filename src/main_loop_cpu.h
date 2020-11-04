//
// Created by neville on 04.11.20.
//

#ifndef SHA_ON_GPU_MAIN_LOOP_CPU_H
#define SHA_ON_GPU_MAIN_LOOP_CPU_H

#include <vector>
#include "choose.cuh"
#include "Sigma0.cuh"
#include "majority.cuh"

// return vector is currently passed by value, could be optimized
std::vector<int> main_loop_cpu(const std::vector<int> in) {
    std::vector<int> message_schedule(64);

    // Initial Hash values
    int H0 = 0x6a09e667;
    int H1 = 0xbb67ae85;
    int H2 = 0x3c6ef372;
    int H3 = 0xa54ff53a;
    int H4 = 0x510e527f;
    int H5 = 0x9b05688c;
    int H6 = 0x1f83d9ab;
    int H7 = 0x5be0cd19;

    for (int i = 0; i < in.size(); i += 16) {

        // File Message Schedule
        for (int j = 0; j < 16; j++) {
            message_schedule[j] = in[i + j];
        }
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

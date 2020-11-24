//
// Created by neville on 03.11.20.
//

#ifndef SHAONGPU_PARSHA256_ON_GPU_H
#define SHAONGPU_PARSHA256_ON_GPU_H

#include <assert.h>
#include <string>
#include "parsha256_padding.h"
#include "parsha256_sha256.h"
#include "lambda.h"
#include "delta.h"
#include "parsha256_kernel_firstRound.cuh"
#include "parsha256_kernel_middleRounds.cuh"
#include "parsha256_kernel_decreasingRounds.cuh"
#include "parsha256_kernel_lastRound.cuh"
#include <sstream>


std::string parsha256_on_gpu(const std::string in) {


    int added_zeros_bits = 0; // How many 0 bits to addd

    const int n = 768; // Input per Node in bits
    const int m = 256; // Output per Node in bits


    int L = in.size() * sizeof(char) * 8;
    int T = 5; // Height of available processor tree
    int t; // Effective height processor tree
    const int l = 0; // IV length



    if (L < delta(0, m, n, l)) {


        std::vector<int> padded = parsha256_padding(in, n - l - L); // IV padding





        parsha256_sha256(padded.data(), padded.data() + 8, padded.data() + 16, padded.data());



        // Write intermediate result to input buffer
        const int diff = (n - m) / 32; // How many numbers to pad

        for (int i = 0; i < diff - 1; i++) {
            padded[8 + i] = 0;
        }
        padded[m / 32 - 1] = _byteswap_ulong(L);

//        for (int &i : padded) {
//            i = _byteswap_uint32(i);
//        }


        parsha256_sha256(padded.data(), padded.data() + 8, padded.data() + 16, padded.data()); // Write intermediate result to input buffer

        std::string res_string = "";
        char buffer[50];
        for (int i = 0; i < 8; i++) {
            int curr = padded[i];
            sprintf(buffer, "%x", curr);
            res_string += buffer;

        }
        return res_string;

    } else if (L < delta(1, m, n, l)) {
        added_zeros_bits += delta(1, m, n, l) - L;
        L = delta(1, m, n, l);
    }


    if (L >= delta(T, m, n, l)) {
        t = T;
    } else {
        for (int i = 1; i < T; i++) {
            if (delta(i, m, n, l) <= L && L < delta(i + 1, m, n, l)) {
                t = i;
                break;
            }
        }
    }

    int q;
    int r;


    if (L > delta(t, m, n, l)) {
        q = (L - delta(t, m, n, l)) / lambda(t, m, n, l);
        r = (L - delta(t, m, n, l)) % lambda(t, m, n, l);
        if (r == 0) {
            q--;
            r = lambda(t, m, n, l);
        }
    } else if (L == delta(t, m, n, l)) {
        q = 0;
        r = 0;
    }

    const int b = std::ceil(r / (double) (2 * n - 2 * m - l));

    const int R = q + t + 2; // Number of Parallel Rounds



    // 1. Padding
    added_zeros_bits += b * (2 * n - 2 * m - l) - r; // How many zeros to padd in bits

    std::vector<int> padded = parsha256_padding(in, added_zeros_bits);


    // 2. Change byte ordering since SHA-256 uses big endian byte ordering
    // This is only necessary to get the same result as other implementations
//    for (int &i : padded) {
//        i = _byteswap_uint32(i);
//    }
//

    int threads = std::pow(2, t);
    const int threads_per_threadsblock = std::min(128, threads);
    int thread_blocks = (threads_per_threadsblock + threads - 1) / threads_per_threadsblock;

    // Copy data to gpu memory
    int *dev_In;
    int *dev_buf1;
    int *dev_buf2;
    int *out;

    cudaMalloc(&dev_In, padded.size() * sizeof(int));
    cudaMalloc(&dev_buf1, 8 * sizeof(int) * threads_per_threadsblock * thread_blocks);
    cudaMalloc(&dev_buf2, 8 * sizeof(int) * threads_per_threadsblock * thread_blocks);
    cudaMalloc(&out, 8 * sizeof(int));

    cudaMemcpy(dev_In, padded.data(), padded.size() * sizeof(int), cudaMemcpyHostToDevice);

//    // Cal kernel

// First Round
    parsha256_kernel_gpu_firstRound<<<thread_blocks, threads_per_threadsblock>>>(dev_In, dev_buf1);
    dev_In += threads * 24; // Consumed Message so far, every threads consumes 24 integers

    // Rounds 2 to p + 1
    const int p = R - t - 1;
    for (int i = 0; i < p; i++) {
        parsha256_kernel_gpu_middleRound<<<thread_blocks, threads_per_threadsblock>>>(dev_In, dev_buf1, dev_buf2);
        dev_In += 8 * (threads / 2) + 24 * (threads / 2); // Consumed Message so far, half of the threads consume 8 ints (non leafs), other halfs consumes again 24 ints
        std::swap(dev_buf1, dev_buf2);
    }

    // Decreasing Rounds
    for (int i = 0; i < t; i++) {
        parsha256_kernel_gpu_decreasingRound<<<thread_blocks, threads_per_threadsblock>>>(dev_In, dev_buf1, dev_buf2);
        std::swap(dev_buf1, dev_buf2);
        threads /= 2;
        dev_In += 8 * threads; // Half of the thread consume 8 ints the other half copy their stuff around
        thread_blocks = (threads_per_threadsblock + threads - 1) / threads_per_threadsblock;

    }

    std::swap(dev_buf1, dev_buf2);
//    assert(threads == 1)
    dev_In += 8;
    parsha256_kernel_gpu_lastRound<<<1, 1>>>(dev_In, dev_buf1, dev_buf2, out, b, L);



//    // Copy result back
    std::vector<int> res_int(8);
    cudaMemcpy(res_int.data(), out, 8 * sizeof(int), cudaMemcpyDeviceToHost);


    // Convert Result to String
    std::string res_string = "";
    char buffer[50];
    for (int i = 0; i < res_int.size(); i++) {
        int curr = res_int[i];
        sprintf(buffer, "%x", curr);
        res_string += buffer;

    }
    return res_string;
}

void parsha256_on_gpu_test() {

    std::cout << parsha256_on_gpu("abc") << std::endl;
    std::cout << parsha256_on_gpu("abcdefgh") << std::endl;
    std::cout << parsha256_on_gpu(std::string(10000, 'a')) << std::endl;

}


#endif //SHAONGPU_PARSHA256_ON_GPU_H

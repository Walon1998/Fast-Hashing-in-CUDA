//
// Created by neville on 03.11.20.
//

#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Padding is performed on cpu

#ifndef SHAONGPU_PADDING_H
#define SHAONGPU_PADDING_H

// return vector is currently passed by value, could be optimized
std::vector<int> padding(const std::string &in) {

    // No Padding necessary
    if (in.length() % 64 == 0) {
        const int newlength = in.length() / 4;
        std::vector<int> out(newlength);
        memcpy(out.data(), in.data(), in.length() * sizeof(char));
        return out;

    } else {
        // Padding necessary
        int remainder = in.length() % 64;
        int k;

        // Calculate k
        if (remainder < 55) {
            k = 56 - remainder - 1;
        } else {
            k = (56 + 64) - remainder - 1;

        }
        const int newlength = in.length() + k + 1 + 8; // new lenngth in uint8
        std::vector<int> out(newlength / 4);

        memcpy(out.data(), in.data(), in.length() * sizeof(char)); // copy existing data

        // Write leading 1
        auto *start_point = (u_int8_t *) out.data();
        start_point[in.length()] = 0x80;

        // Fill leading zeros
        for (unsigned int i = in.length() + 1; i < newlength - 4; i++) {
            start_point[i] = 0;
        }

//        // Write L at the end
        auto end_point = (u_int64_t *) &start_point[newlength - 8];
        const u_int64_t bitlength = in.length() * sizeof(char) * 8;
        end_point[0] = __builtin_bswap64(bitlength); // Save length and change byte ordering
        return out;
    }


}

void padding_test() {

    // TODO: Does "" need padding?
    // TODO: Padding necessary even if block is message is multiple of 512?

    std::vector<int> out = padding("abc");
    auto testi = (u_int8_t *) out.data();
    for (int i = 0; i < 64; i++) {
        if (i == 0) {
            assert((testi[i] == 'a'));
        } else if (i == 1) {
            assert((testi[i] == 'b'));
        } else if (i == 2) {
            assert((testi[i] == 'c'));
        } else if (i == 3) {
            assert((+testi[i] == 0b10000000));
        } else if (i == 63) {
            assert((+testi[i] == 3 * 8));
        } else {
            assert(+testi[i] == 0);
        }

//        std::cout << +testi[i] << ": " << testi[i] << std::endl;
    }

}

#endif //SHAONGPU_PADDING_H

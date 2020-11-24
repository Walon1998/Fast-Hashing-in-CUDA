//
// Created by neville on 03.11.20.
//

#include "choose.cuh"
#include "majority.cuh"
#include "sha256_padding.h"
#include "SHR.cuh"
#include "ROTR.cuh"
#include "mk_sigma_test.cuh"
#include "sigma.cuh"
#include "sha256_on_cpu.h"
#include "sha256_on_gpu.h"
#include "parsha256_on_gpu.h"

#ifndef SHAONGPU_RUN_TESTS_H
#define SHAONGPU_RUN_TESTS_H

// TODO: Run tests not only on cpu but also on gpu
void sha256_run_tests() {
    maj_test();
    ch_test();
    ROTR_test();
    SHR_test();
    mk_sigma_test<&sigma0>(3, 0x02004000);
    mk_Sigma_test<&Sigma0>(0x40080400);
    mk_sigma_test<&sigma1>(10, 0x0000a000);
    mk_Sigma_test<&Sigma1>(0x04200080);
    sha256_on_gpu_test();
    sha256_on_cpu_test();
    sha256_padding_test();

}

void parsha256_run_tests() {
    parsha256_on_gpu_test();
}


#endif //SHAONGPU_RUN_TESTS_H

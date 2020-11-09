//
// Created by neville on 03.11.20.
//

#include "choose.cuh"
#include "majority.cuh"
#include "padding.h"
#include "SHR.cuh"
#include "ROTR.cuh"
#include "mk_sigma_test.cuh"
#include "sigma0.cuh"
#include "Sigma0.cuh"
#include "sigma1.cuh"
#include "Sigma1.cuh"
#include "sha256_On_Cpu.h"
#include "sha256_On_Gpu.h"

#ifndef SHAONGPU_RUN_TESTS_H
#define SHAONGPU_RUN_TESTS_H

// TODO: Run tests not only on cpu but also on gpu
void run_tests() {
    std::cout << std::endl;

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
    padding_test();

}

#endif //SHAONGPU_RUN_TESTS_H

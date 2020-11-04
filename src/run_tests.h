//
// Created by neville on 03.11.20.
//

#include "majority.cuh"
#include "choose.cuh"
#include "Sigma0.cuh"
#include "sha256_On_Gpu.h"
#include "padding.h"
#include "sha256_On_Cpu.h"

#ifndef SHAONGPU_RUN_TESTS_H
#define SHAONGPU_RUN_TESTS_H

void run_tests() {
    std::cout << "Running Tests" << std::endl;

    maj_test();
    ch_test();
    //Sigma0_test();
    //sha256_on_gpu_test();
    //sha256_on_cpu_test();
    padding_test();

}

#endif //SHAONGPU_RUN_TESTS_H

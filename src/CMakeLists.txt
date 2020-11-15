cmake_minimum_required(VERSION 3.13)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.1/bin/nvcc)
project(SHA_ON_GPU CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(SHA_ON_GPU main.cu lambda.h delta.h parsha256_kernel.cuh)

set_target_properties(
        SHA_ON_GPU
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

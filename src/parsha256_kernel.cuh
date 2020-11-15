//
// Created by neville on 15.11.20.
//

#ifndef SHA_ON_GPU_PARSHA256_KERNEL_CUH
#define SHA_ON_GPU_PARSHA256_KERNEL_CUH

__global__ void parsha256_kernel_gpu(int *__restrict__ in, int *__restrict__ buf1, int *__restrict__ buf2, int *__restrict__ out, const int R, const int t) {

    const int id = threadIdx.x + blockIdx.x * blockDim.x; // My id
    const int parent = id / 2; // Parent id

    const int child1 = 2 * id;
    const int child2 = 2 * id + 1;

    int total_threads = blockDim.x * gridDim.x; // total thread in grid


    int *buf1_me = buf1 + id * 8; // my buffer where to write
    int *buf2_me = buf2 + id * 8; // double buffered


    int *buf1_child1 = buf1 + child1 * 8; // child bufffer, where to read
    int *buf2_child1 = buf2 + child1 * 8; // Everything is double buffered sucht that less synchronization is used

    int *buf1_child2 = buf1 + child2 * 8; // child bufffer, where to read
    int *buf2_child2 = buf2 + child2 * 8; // Everything is double buffered sucht that less synchronization is used


    int *buffer1_read;
    int *buffer2_read;
    int *buffer3_read;
    int *buffer_write;





    // First Round
    const int *first_round_in = in + id * 24;
    parsha256_sha256(first_round_in, first_round_in + 8, first_round_in + 16, buf1_me);
    cudaDeviceSynchronize();

    in += total_threads * 24;

    if (id < total_threads / 2) {
        in += 8 * id;
    } else {
        in += 8 * (total_threads / 2) + id * 24;
    }

    // Rounds 2 to p + 1
    const int p = R - t - 1;

    for (int i = 0; i < p; i++) {

        if (id < total_threads / 2) {
            buffer1_read = buf1_child1;
            buffer2_read = buf1_child2;
            buffer3_read = in;
            buffer_write = buf2_me;


        } else {
            buffer1_read = in;
            buffer2_read = in + 8;
            buffer3_read = in + 16;
            buffer_write = buf2_me;
        }


        parsha256_sha256(buffer1_read, buffer2_read, buffer3_read, buffer_write);

        // Swap Buffers
        auto tmp = buf1_me;
        buf1_me = buf2_me;
        buf2_me = tmp;

        tmp = buf1_child1;
        buf1_child1 = buf2_child1;
        buf2_child1 = tmp;

        tmp = buf1_child2;
        buf1_child2 = buf2_child2;
        buf2_child2 = tmp;


        cudaDeviceSynchronize();
        in += (total_threads / 2) * 24 + (total_threads / 2) * 8;

    }


    for (int i = 0; i < t; i++) {


        if (id < total_threads / 2) { // Compute result

            buffer1_read = buf1_child1;
            buffer2_read = buf1_child2;
            buffer3_read = in;
            buffer_write = buf2_me;


            parsha256_sha256(buffer1_read, buffer2_read, buffer3_read, buffer_write);


        } else { // Copy buffers

#pragma unroll
            for (int j = 0; j < 8; j++) {
                buf2_me[j] = buf1_me[j];
            }

        }

        total_threads = total_threads / 2;

        // Swap Buffers
        auto tmp = buf1_me;
        buf1_me = buf2_me;
        buf2_me = tmp;

        tmp = buf1_child1;
        buf1_child1 = buf2_child1;
        buf2_child1 = tmp;

        tmp = buf1_child2;
        buf1_child2 = buf2_child2;
        buf2_child2 = tmp;


        cudaDeviceSynchronize();
        in += total_threads * 8;

    }

    if (threadIdx.x == 0) {
        out[0] = buf1_me[0];
        out[1] = buf1_me[1];
        out[2] = buf1_me[2];
        out[3] = buf1_me[3];
        out[4] = buf1_me[4];
        out[5] = buf1_me[5];
        out[6] = buf1_me[6];
        out[7] = buf1_me[7];
    }

}

#endif //SHA_ON_GPU_PARSHA256_KERNEL_CUH

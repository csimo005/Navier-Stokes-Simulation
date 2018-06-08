#include <stdio.h>
#include <stdlib.h>
#include "project_support.h"
#include "project.cu"

int main(int argc, char *argv[]) {
    srand(217);

    Timer timer;
    cudaError_t cuda_ret;

    printf("Setting up problem...\n"); fflush(stdout);

    float *initial_h, *final_h;
    float *initial_d, *final_d;

    unsigned int *size_h;

    size_h = (unsigned int*) malloc(sizeof(unsigned int)*3);
    if(argc == 1) {
        size_h[0]=size_h[1]=size_h[2]=100;
    } else if(argc == 2) {
        size_h[0]=size_h[1]=size_h[2]=atoi(argv[1]);
    } else if(argc == 4) {
        size_h[0]=atoi(argv[1]);
        size_h[1]=atoi(argv[2]);
        size_h[2]=atoi(argv[3]);
    } else {
        printf("Error expected usage: force_test <side> or advection_test <x> <y> <z>"); fflush(stdout);
    }

    unsigned int VecSize = size_h[0]*size_h[1]*size_h[2]*3;
    initial_h = (float*) malloc(sizeof(float)*VecSize);
    final_h   = (float*) malloc(sizeof(float)*VecSize);

    for(unsigned int i=0;i<VecSize;i++)
        initial_h[i] = (rand()%100)/100.0;

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u x %u\n  ", size_h[0], size_h[1], size_h[2]);


    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**) &initial_d, sizeof(float)*VecSize);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate initial vector field");
    cuda_ret = cudaMalloc((void**) &final_d,   sizeof(float)*VecSize);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate final vector field");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Copying data from host to device..."); fflush(stdout);

    cudaMemcpy(initial_d, initial_h, sizeof(float)*VecSize, cudaMemcpyHostToDevice);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    
    project(initial_d, final_d, size_h[0], size_h[1], size_h[2]);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cudaMemcpy(final_h, final_d, sizeof(float)*VecSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Not Verfying result..."); fflush(stdout);
 //   verify(initial_h, final_h, force_h, pos_h, r, size_h);

    free(initial_h);
    free(final_h);

    cudaFree(initial_d);
    cudaFree(final_d);

    return 0;
}

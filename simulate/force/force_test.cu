#include <stdio.h>
#include <stdlib.h>
#include "force_support.h"
#include "force.cu"

int main(int argc, char *argv[]) {
    srand(217);

    Timer timer;
    cudaError_t cuda_ret;

    printf("Setting up problem...\n"); fflush(stdout);

    float *initial_h, *final_h;
    float *initial_d, *final_d;

    unsigned int *size_h, *size_d;

    float *force_h, *force_d;
    float *pos_h, *pos_d;
    float r = 0.1;

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
    force_h   = (float*) malloc(sizeof(float)*3);
    pos_h     = (float*) malloc(sizeof(float)*3);

    for(unsigned int i=0;i<VecSize;i++)
        initial_h[i] = (rand()%100)/100.0;

    for(unsigned int i=0;i<3;i++) {
        force_h[i] = (rand()%100)/100.0;
	pos_h[i]   = rand()%size_h[i];
    }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u x %u\n  ", size_h[0], size_h[1], size_h[2]);


    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**) &initial_d, sizeof(float)*VecSize);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate initial vector field");
    cuda_ret = cudaMalloc((void**) &final_d,   sizeof(float)*VecSize);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate final vector field");
    cudaMalloc((void**) &size_d, sizeof(unsigned int)*3);
    cudaMalloc((void**) &force_d, sizeof(float)*3);
    cudaMalloc((void**) &pos_d, sizeof(float)*3);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Copyind data from host to device..."); fflush(stdout);

    cudaMemcpy(initial_d, initial_h, sizeof(float)*VecSize, cudaMemcpyHostToDevice);
    cudaMemcpy(size_d, size_h, sizeof(unsigned int)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(force_d, force_h, sizeof(float)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(pos_d, pos_h, sizeof(float)*3, cudaMemcpyHostToDevice);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    
    force(initial_d, final_d, force_d, pos_d, r, size_h[0], size_h[1], size_h[2]);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cudaMemcpy(final_h, final_d, sizeof(float)*VecSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Verfying result..."); fflush(stdout);
    verify(initial_h, final_h, force_h, pos_h, r, size_h);

    free(initial_h);
    free(final_h);
    free(size_h);
    free(force_h);
    free(pos_h);

    cudaFree(initial_d);
    cudaFree(final_d);
    cudaFree(size_d);
    cudaFree(force_d);
    cudaFree(pos_d);

    return 0;
}

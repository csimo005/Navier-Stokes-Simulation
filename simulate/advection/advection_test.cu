#include <stdio.h>
#include <stdlib.h>
#include "advection_support.h"
#include "advection.cu"

int main(int argc, char *argv[]) {
    Timer timer;
    cudaError_t cuda_ret;

    printf("Setting up problem...\n"); fflush(stdout);

    float *initial_h,*final_h;
    float *initial_d,*final_d;

    unsigned int x, y, z;
    if(argc == 1) {
        x=y=z=100;
    } else if(argc == 2) {
        x=y=z=atoi(argv[1]);
    } else if(argc == 4) {
        x=atoi(argv[1]);
        y=atoi(argv[2]);
        z=atoi(argv[3]);
    } else {
        printf("Error expected usage: advection_test <side> or advection_test <x> <y> <z>"); fflush(stdout);
    }

    unsigned int size = x*y*z*3;
    initial_h = (float*) malloc(sizeof(float)*size);
    final_h = (float*) malloc(sizeof(float)*size);
    for(unsigned int i=0;i<size;i++)
        initial_h[i] = 0;

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u x %u\n  ", x, y, z);


    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cudaMalloc((void**) &initial_d, sizeof(float)*size);
    cudaMalloc((void**) &final_d, sizeof(float)*size);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Copyind data from host to device..."); fflush(stdout);

    cudaMemcpy(initial_d, initial_h, sizeof(float)*size, cudaMemcpyHostToDevice);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    
    advection(initial_d, final_d, x, y, z);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cudaMemcpy(final_h, final_d, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Not Verfying result..."); fflush(stdout);
    //verify(initial_h, final_h, x, y, z, size);

    free(initial_h);
    free(final_h);

    cudaFree(initial_d);
    cudaFree(final_d);    

    return 0;
}

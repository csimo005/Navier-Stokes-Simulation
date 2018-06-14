#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>

#define DELTA_T 0.01667
#define LATTICE_SIZE 0.01

#include "advection/advection.h"
#include "diffusion/diffusion.h"
#include "force/force.h"
#include "projection/project.h"
#include "points/points.h"

#include "util.h"

int main(int argc, char *argv[]) {
    Timer timer;

    unsigned int size_x=200;
    unsigned int size_y=200;
    unsigned int size_z=200;

    unsigned int n=3*size_x*size_y*size_z;
    unsigned int n_p = 70000000; //1 million points defined by 7 flos each

    float *u, *p;
    printf("Begging Initializatoin...\n"); fflush(stdout);
    startTime(&timer);
    cudaMalloc((void **) &u, n*sizeof(float));
    cudaMallocManaged(&p, n_p*sizeof(float));

    float *force_h, *grav_h, *pos_h;
    float *force_d, *grav_d, *pos_d;

    force_h = (float*) malloc(3*sizeof(float));
    pos_h   = (float*) malloc(3*sizeof(float));
    grav_h  = (float*) malloc(3*sizeof(float));

    force_h[0] = 0.5; grav_h[0] =  0;   pos_h[0] = 1.25;
    force_h[1] = 0.5; grav_h[1] = -9.8; pos_h[1] = 1.25;
    force_h[2] = 0.5; grav_h[2] =  0;   pos_h[2] = 1.25;

    cudaMalloc((void**) &force_d, 3*sizeof(float));
    cudaMalloc((void**) &grav_d, 3*sizeof(float));
    cudaMalloc((void**) &pos_d, 3*sizeof(float));
    cudaError cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) {
        printf("Error: failed to allocated device variables\n\tThrew: %s\n",cudaGetErrorString(cuda_ret));fflush(stdout);
	return 0;
    }

    cuda_ret = cudaMemcpy(force_d, force_h, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grav_d, grav_h, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_d, pos_h, 3*sizeof(float), cudaMemcpyHostToDevice);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) {
        printf("Error: Memcpy of initial values failed\n\tThrew: %s\n",cudaGetErrorString(cuda_ret));fflush(stdout);
	return 0;
    }


    dim3 blockDim_1D(1000,1,1);
    dim3 gridDim_1D((n-1)/1000+1,1,1);

    zeroVector<<<gridDim_1D, blockDim_1D>>>(u,n);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) {
        printf("Error: Failed to initialize velocity field\n\tThrew: %s\n",cudaGetErrorString(cuda_ret));fflush(stdout);
        return 0;
    }

    initPoints(p, 1000000, size_x, size_y, size_z); 
    stopTime(&timer); printf("Initialization finished: %f s\n", elapsedTime(timer)); fflush(stdout);
    
    std::ofstream fout;
    fout.open("points.bin",std::ios::out | std::ios::app | std::ios::binary);

    for(unsigned int i=0;i<11/DELTA_T;i++) {
	startTime(&timer);
        globalForce(u,grav_d,size_x,size_y,size_z);
	cudaDeviceSynchronize();
	if(i > 1/DELTA_T && i < 3/DELTA_T) {
            localForce(u,force_d,pos_d,0.25,size_x,size_y,size_z);
	}
	cudaDeviceSynchronize();
	stopTime(&timer); float forceTime = elapsedTime(timer);
	
	startTime(&timer);
        advection(u,size_x,size_y,size_z);
	stopTime(&timer); float advectionTime = elapsedTime(timer);
	startTime(&timer);
        diffusion(u,size_x,size_y,size_z);
        stopTime(&timer); float diffusionTime = elapsedTime(timer);
	startTime(&timer);
	project(u,size_x,size_y,size_z);
        stopTime(&timer); float projectTime = elapsedTime(timer);
        cudaDeviceSynchronize();
	startTime(&timer);
        updatePoints(u, p, 1000000, size_x, size_y, size_z);
	stopTime(&timer); float updateTime = elapsedTime(timer);
        cudaDeviceSynchronize();
	for(unsigned int j=0;j<1000000;j++) {
            fout << p[j*7+0] << p[j*7+1] << p[j*7+2];
	}

	printf("%.2f: %.6f\n", i*DELTA_T,forceTime+advectionTime+diffusionTime+projectTime+updateTime); fflush(stdout);
	printf("\tforce:      %.6f\n", forceTime); fflush(stdout);
	printf("\tadvection:  %.6f\n", advectionTime); fflush(stdout);
	printf("\tdiffusion:  %.6f\n", diffusionTime); fflush(stdout);
	printf("\tprojection: %.6f\n", projectTime); fflush(stdout);
	printf("\tupdate:     %.6f\n", updateTime); fflush(stdout);
    }

    fout.close();

    cudaFree(u);
    cudaFree(force_d);
    cudaFree(pos_d);
    cudaFree(grav_d);
    cudaFree(p);

    free(force_h);
    free(pos_h);
    free(grav_h);

    return 0;
}

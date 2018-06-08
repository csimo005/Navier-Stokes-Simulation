#include <stdio.h>
#include <stdlib.h>
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
    unsigned int size_x=200;
    unsigned int size_y=200;
    unsigned int size_z=200;

    unsigned int n=3*size_x*size_y*size_z;
    unsigned int n_p = 7000000; //1 million points defined by 7 flos each

    float *u, *p;
//    curandState_t *states; 

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

    for(unsigned int i=0;i<11/DELTA_T;i++) {
        globalForce(u,grav_d,size_x,size_y,size_z);
	cudaDeviceSynchronize();
	if(i > 1/DELTA_T && i < 3/DELTA_T) {
            localForce(u,force_d,pos_d,0.25,size_x,size_y,size_z);
	}
	cudaDeviceSynchronize();
	
        advection(u,size_x,size_y,size_z);
        diffusion(u,size_x,size_y,size_z);
	project(u,size_x,size_y,size_z);
        cudaDeviceSynchronize();
        updatePoints(u, p, 1000000, size_x, size_y, size_z);

	printf("%.2f\n", i*DELTA_T); fflush(stdout);
    }

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

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "force_support.h"

#ifndef DELTA_T
#define DELTA_T 0.001
#endif

void verify(float *initial_field, float *final_field, float *f, float *pos, float r, unsigned int *field_size) {
    const float relativeTolerance = 1e-2;

    float Fdt[3] = {f[0]*DELTA_T, f[1]*DELTA_T, f[2]*DELTA_T};
    for(int i=0;i<field_size[2];++i) { //loop over z
        for(int j=0;j<field_size[1];++j) { //loop over y
            for(int k=0;k<field_size[0];++k) { //loop over x
		for(int l=0;l<3;++l) {
                    float sum = initial_field[((i*field_size[1]+j)*field_size[0]+k)*3+l];
		    if(sqrt(pow(pos[0]-k*0.001,2.0)+pow(pos[1]-j*0.001,2.0)+pow(pos[2]-i*0.001,2.0)) < r) {
                        sum += Fdt[l]*exp(-1*(pow(pos[0]-k*0.001,2)+pow(pos[1]-j*0.001,2)+pow(pos[2]-i*0.001,2))/r);
		    }
                    float relativeError = (sum - final_field[((i*field_size[1]+j)*field_size[0]+k)*3+l])/sum;
                    if(relativeError > relativeTolerance   || relativeError < -relativeTolerance) {
                        printf("\nTEST FAILED: (%d,%d,%d), %d\n\n",k,j,i,l);
                        exit(0);
                    }
		}
            }
        }
    }
    
    printf("\nTEST PASSED\n\n");
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

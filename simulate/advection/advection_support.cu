#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "advection_support.h"

/*void verify(float *initial, float *final, unsigned int x, unsigned int y, unsigned int z) {
    const float relativeTolerance = 1e-2;

    for(int i = 0; i < n; ++i) {
        float sum = A[i]+B[i];
        printf("\t%f/%f",sum,C[i]);
        float relativeError = (sum - C[i])/sum;
        if(relativeError > relativeTolerance   || relativeError < -relativeTolerance) {
            printf("\nTEST FAILED\n\n",i);
            exit(0);
        }
    }
    
    printf("\nTEST PASSED\n\n");
}

float roundUp(float val, int decimal) {
    float scale = pow(10,decimal);
    int rounded = val*scale + 1;
    return rounded/scale
}

float roundDown(float val, int decimal) {
    float scale = pow(10,decimal);
    int rounded = val*scale;
    return rounded/scale;
}

//cubes look like this:
//      3-------7
//     /|      /|
//    / |     / |
//   1--|----5  |
//   |  2----|--6
//   | /     | /
//   0-------4
// x+ 2->0
// y+ 2->6
// z+ 2->3
// x - w
// y - d
// z - h

float trilinear_interpolation(float x, float y, float z, float *field, int h, int w, int d) {
    int decimal = 6;
    int scale = pow(10,decimal);

    float x_0=roundDown(x,decimal), x_1=roundUp(x,decimal);
    float y_0=roundDown(x,decimal), y_1=roundUp(x,decimal);
    float z_0=roundDown(x,decimal), z_1=roundUp(x,decimal);

    float c8[8];
    c8[0] = field[x_1*scale + y_0*scale*w + z_0*scale*w*d];
    c8[1] = field[x_1*scale + y_0*scale*w + z_1*scale*w*d];
    c8[2] = field[x_0*scale + y_0*scale*w + z_0*scale*w*d];
    c8[3] = field[x_0*scale + y_0*scale*w + z_1*scale*w*d];
    c8[4] = field[x_1*scale + y_1*scale*w + z_0*scale*w*d];
    c8[5] = field[x_1*scale + y_1*scale*w + z_1*scale*w*d];
    c8[6] = field[x_0*scale + y_1*scale*w + z_0*scale*w*d];
    c8[7] = field[x_0*scale + y_1*scale*w + z_1*scale*w*d];

    float x_d = (x-x_0)/(x_1-x);
    float y_d = (y-y_0)/(y_1-y);
    float z_d = (z-z_0)/(z_1-z);

    float c4[4];
    for(int i=0;i<4;i++)
        c4[i] = c8[i]*(1-x_d)+c8[i+4]*x_d

    float c2[2];
    for(int i=0;i<2;i++)
        c2[i] = c4[i]*(1-y_d)+c4[i+2]*y_d;

    return c2[0]*(1-z_d)+c2[1]*z_d;
}*/

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

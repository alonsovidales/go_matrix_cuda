/***************************************************
 * Module that applay the function sigmoid to all the elements of the matrix
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_sigmoid_gradient.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_sigmoid_gradient.cu
 *
 **************************************************/

//#include <stdio.h>
//#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void matrixSigmoidGrad(double* A, int resW, int resH, int width, int finalSize)
{
	int x = threadIdx.x + (blockIdx.x * resW);
        int y = threadIdx.y + (blockIdx.y * resH);
	int resultPos = y * width + x;

	if (resultPos < finalSize && x < width) {
		//printf("IN Block %d - %d, wA: %d thread %d - %d Val: %f resultPos: %d finalSize: %d\n", x, y, wA, threadIdx.x, threadIdx.y, A[resultPos], resultPos, finalSize);
		double s = 1 / (1 + pow(M_E, (double)(-1 * A[resultPos])));
		A[resultPos] = s * (1 - s);
	}
}

#ifdef __cplusplus
}
#endif

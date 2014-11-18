/***************************************************
 * Module that applay the function sigmoid to all the elements of the matrix
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_pow_two.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_pow_two.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixPowTwo(double* A, int resW, int resH, int width, int finalSize)
{
	int x = threadIdx.x + (blockIdx.x * resW);
        int y = threadIdx.y + (blockIdx.y * resH);
        int resultPos = y * width + x;

	if (resultPos < finalSize && x < width) {
		//printf("IN Block %d - %d, wA: %d thread %d - %d Val: %f resultPos: %d finalSize: %d\n", x, y, wA, threadIdx.x, threadIdx.y, A[resultPos], resultPos, finalSize);
		A[resultPos] = A[resultPos] * A[resultPos];
	}
}

#ifdef __cplusplus
}
#endif

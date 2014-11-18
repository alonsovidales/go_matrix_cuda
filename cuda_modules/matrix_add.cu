/***************************************************
 * Module for matrix addition
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_add.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_add.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixAdd(double* C, double* A, double* B, int width, int resW, int resH, int resultSize)
{
	int x = threadIdx.x + (blockIdx.x * resW);
	int y = threadIdx.y + (blockIdx.y * resH);
	int resultPos = y * width + x;

	if (resultPos < resultSize && x < width) {
		C[resultPos] = A[resultPos] + B[resultPos];
		//printf("Block %d - %d, thread %d - %d Val: %f\n", x, y, threadIdx.x, threadIdx.y, C[resultPos]);
	}
}

#ifdef __cplusplus
}
#endif

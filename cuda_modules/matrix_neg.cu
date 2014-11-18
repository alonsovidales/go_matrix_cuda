/***************************************************
 * Module that negs all the elements on a matrix
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_neg.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_neg.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixNeg(double* A, int resW, int resH, int width, int finalSize)
{
	int x = threadIdx.x + (blockIdx.x * resW);
	int y = threadIdx.y + (blockIdx.y * resH);
	int resultPos = y * width + x;

	if (resultPos < finalSize && x <  width) {
		A[resultPos] = -A[resultPos];
		//printf("Block %d - %d, thread %d - %d Val: \n", x, y, threadIdx.x, threadIdx.y);
	}
}

#ifdef __cplusplus
}
#endif

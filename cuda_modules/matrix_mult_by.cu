/***************************************************
 * Module that multiply all the elements of a matrix by a number
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_mult_by.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_mult_by.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixMultBy(double* A, double multBy, int width, int resW, int resH, int finalSize)
{
	int x = threadIdx.x + (blockIdx.x * resW);
	int y = threadIdx.y + (blockIdx.y * resH);
	int resultPos = y * width + x;

	if (resultPos < finalSize && x < width) {
		A[resultPos] *= multBy;
		//printf("Block %d - %d, thread %d - %d Val: %f\n", x, y, threadIdx.x, threadIdx.y, A[resultPos]);
	}
}

#ifdef __cplusplus
}
#endif

/***************************************************
 * Module that applay the log to all the elements of the matrix
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_log.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_log.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixLog(double* A, int resW, int resH, int width, int finalSize)
{
	int x = threadIdx.x + (blockIdx.x * resW);
        int y = threadIdx.y + (blockIdx.y * resH);
        int resultPos = y * width + x;

	if (resultPos < finalSize && x < width) {
		A[resultPos] = (double)log((double)A[resultPos]);
		//printf("Block %d - %d, thread %d - %d Val: %f %f %f\n", x, y, threadIdx.x, threadIdx.y, A[resultPos], log((double)A[resultPos]), (double)log((double)A[resultPos]));
	}
}

#ifdef __cplusplus
}
#endif

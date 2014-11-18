/***************************************************
 * Module that negs all the elements on a matrix
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_trans.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_trans.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixTrans(double* C, double* A, int resW, int resH, int width, int height, int finalSize)
{
	int x = threadIdx.x + (blockIdx.x * resW);
	int y = threadIdx.y + (blockIdx.y * resH);
	int resultPos = y * width + x;

	if (resultPos < finalSize && x <  width) {
		C[resultPos] = A[x * height + y];
		//printf("Block %d - %d, thread %d - %d Val: %f Pos: %d Row: %d\n", x, y, threadIdx.x, threadIdx.y, C[resultPos], resultPos, resultPos - (resultPos / width + 1));
	}
}

#ifdef __cplusplus
}
#endif

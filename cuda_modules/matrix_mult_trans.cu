/***************************************************
 * Module that multiply a matrix by the transpose of other
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_mult_trans.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_mult_trans.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixMulTrans(double* C, double* A, double* B, int wA, int resW, int resH, int resultWidth, int resultSize)
{
	int x = threadIdx.x + (blockIdx.x * resW);
	int y = threadIdx.y + (blockIdx.y * resH);
	int resultPos = y * resultWidth + x;

	//printf("Thread %d - %d: %d. Final: x: %d y: %d Size: %d\n", threadIdx.x, threadIdx.y, resultPos, x, y, resultSize);
	if (resultPos < resultSize && x < resultWidth) {
		// value stores the element that is 
		// computed by the thread
		double value = 0;
		for (int i = 0; i < wA; ++i)
		{
			value += A[y * wA + i] * B[x * wA + i];

			//printf("Pos %d - %d, thread %d - %d : pos: %d %d H: %d Pos: %d Val: %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, x, y, resultWidth, resultPos, value);
		}

		// Write the matrix to device memory each 
		// thread writes one element
		C[resultPos] = value;
	}
}

#ifdef __cplusplus
}
#endif

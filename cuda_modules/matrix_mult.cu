/***************************************************
 * Module for matrix multiplication
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_mult.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_mult.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
	// CUDA Kernel
	__global__ void matrixMul(double* C, double* A, double* B, int wA, int wB, int resW, int resH, int resultSize)
	{
		int x = threadIdx.x + (blockIdx.x * resW);
		int y = threadIdx.y + (blockIdx.y * resH);
		int resultPos = y * wB + x;

		// 2014/06/28 17:43:45 Final SIZE: 17 60 Grid: 4 1 Size: 60 60 3600 Treads: 1024
		if (resultPos < resultSize && x < wB) {
			// value stores the element that is 
			// computed by the thread
			double value = 0;
			for (int i = 0; i < wA; ++i)
			{
				value += A[y * wA + i] * B[i * wB + x];
			}

			// Write the matrix to device memory each 
			// thread writes one element
			C[resultPos] = value;
			//printf("Block %d - %d, Thread %d - %d: %d. Final: x: %d y: %d %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, resultPos, x, y, value);
		}
	}

#ifdef __cplusplus
}
#endif

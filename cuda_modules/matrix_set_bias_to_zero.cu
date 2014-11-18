/***************************************************
 * Module that multiply all the elements of a matrix by a number
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_set_bias_to_zero.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_set_bias_to_zero.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixSetBiasToZero(double* A, int height, int width, int resH)
{
	int y = threadIdx.y + (blockIdx.y * resH);
	if (y < height) {
		int resultPos = y * width;

		A[resultPos] = 0;
	}
}

#ifdef __cplusplus
}
#endif

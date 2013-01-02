#include "VectorAdd.h"

__global__ void _vectorAdd(float *A, float *B, float *C, unsigned int size, unsigned int column)
{
	unsigned int blk = blockIdx.x * blockDim.x;
	unsigned int dx = threadIdx.x;
	unsigned int dy = threadIdx.y * column;
	unsigned idx = dx + dy + blk;

	//__shared__ float values[1024];

	if (idx < size)
		C[idx] = A[idx] + B[idx];
}


void vectorAdd(float *A, float *B, float *C, unsigned int size)
{
	dim3 threads(32, 32);
	dim3 blocks((size + 1023) / 1024);
	_vectorAdd<<<blocks, threads>>>(A, B, C, size, 32);
}


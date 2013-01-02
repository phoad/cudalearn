#include <stdio.h>
#include <stdlib.h>

#include "VectorAdd.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

int main(int argc, char **argv)
{
	unsigned int count = 1024 * 1;

	float *A = new float[count];
	float *B = new float[count];
	float *C = new float[count];

	for (unsigned int i = 0; i < count; ++i) {
		A[i] = (float) i;
		B[i] = (float) i;
	}

	float *dA, *dB, *dC;

	checkCudaErrors(cudaMalloc(&dA, sizeof(float) * count));
	checkCudaErrors(cudaMalloc(&dB, sizeof(float) * count));
	checkCudaErrors(cudaMalloc(&dC, sizeof(float) * count));

	checkCudaErrors(cudaMemcpy(dA, A, sizeof(float) * count, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dB, B, sizeof(float) * count, cudaMemcpyHostToDevice));

	vectorAdd(dA, dB, dC, count);

	checkCudaErrors(cudaMemcpy(C, dC, sizeof(float) * count, cudaMemcpyDeviceToHost));

	for (int i = 0; i < 10; ++i) {
		printf("%.3f ", C[i]);
	}
	printf("\n");

	delete []A;
	delete []B;
	delete []C;

	return EXIT_SUCCESS;
}

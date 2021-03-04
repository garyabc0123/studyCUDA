#define CUDACC
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <stdio.h>
#include <iostream>
#include <time.h>
#define BLOCKSIZE 32
//128 too big for shared memory
#define ARRAYSIZE 2048

__global__ void shareMem(int* in, int* out) {
	__shared__ int shareMemory[BLOCKSIZE][BLOCKSIZE];

	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	int tranX = threadIdx.x + blockIdx.y * blockDim.y;
	int tranY = threadIdx.y + blockIdx.x * blockDim.x;
	int tranIndex = tranX + tranY * ARRAYSIZE;
	int index = indexX * ARRAYSIZE + indexY;

	shareMemory[threadIdx.x][threadIdx.y] = in[index];
	
	__syncthreads();

	out[tranIndex] = shareMemory[threadIdx.y][threadIdx.x];
}

__global__ void unshareMem(int* in, int* out) {
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;
	int index = indexX * ARRAYSIZE + indexY;
	int tranIndex = indexX + indexY * ARRAYSIZE;
	out[tranIndex] = in[index];

}

void calCPU(int* in, int* out) {
	for (size_t x = 0; x < ARRAYSIZE; x++) {
		for (size_t y = 0; y < ARRAYSIZE; y++) {
			out[y * ARRAYSIZE + x] = in[x * ARRAYSIZE + y];
		}
	}
}

bool equal(size_t size, int* a, int* b, int* c) {
	for (auto it = 0; it < size; it++) {
		if (it[a] != it[b] || it[a] != it[c]) {
			std::cout << it << "\t" << it[a] << "\t" << it[b] << "\t" << it[c] << std::endl;
			return false;

		}
	}
	return true;

}

int main(void) {
	srand(time(NULL));
	int* host_a, * host_b_ubshared, *host_b_shared, *host_b_calCPU;
	int* dev_a, * dev_b;
	host_a = new int[ARRAYSIZE * ARRAYSIZE];
	host_b_ubshared = new int[ARRAYSIZE * ARRAYSIZE];
	host_b_shared = new int[ARRAYSIZE * ARRAYSIZE];
	host_b_calCPU = new int[ARRAYSIZE * ARRAYSIZE];
	cudaMalloc((void**)&dev_a, ARRAYSIZE * ARRAYSIZE * sizeof(int));
	cudaMalloc((void**)&dev_b, ARRAYSIZE * ARRAYSIZE * sizeof(int));

	for (auto it = 0; it < ARRAYSIZE * ARRAYSIZE; it++) {
		it[host_a] = rand();
	}

	cudaMemcpy(dev_a, host_a, ARRAYSIZE * ARRAYSIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	dim3 thread(BLOCKSIZE, BLOCKSIZE, 1);
	dim3 block(ARRAYSIZE / BLOCKSIZE, ARRAYSIZE / BLOCKSIZE, 1);

	unshareMem << <block, thread >> > (dev_a, dev_b);

	cudaMemcpy(host_b_ubshared, dev_b, ARRAYSIZE * ARRAYSIZE * sizeof(int), cudaMemcpyDeviceToHost);
	
	shareMem << <block, thread >> > (dev_a, dev_b);
	cudaMemcpy(host_b_shared, dev_b, ARRAYSIZE * ARRAYSIZE * sizeof(int), cudaMemcpyDeviceToHost);

	calCPU(host_a, host_b_calCPU);
	std::cout << (equal ? "true" : "false") << std::endl;
	return 0;


}
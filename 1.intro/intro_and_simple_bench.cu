
#include "cuda_runtime.h"
#include "device_launch_parameters.h"




#include <iostream>
#include <stdio.h>
#include <time.h>
#include <chrono>
#define SIZE 1048576 //1024*1024
#define BLOCKSIZE 1024
__global__ void deviceADD(int* a, int* b, int* c) {
	int off = blockIdx.x + threadIdx.x * BLOCKSIZE;
	c[off] = a[off] + b[off];
}
void fillramdom(int size, int* ptr) {
	for (int i = 0; i < size; i++) {
		i[ptr] = rand();
	}
}
//cpu版計算function

std::chrono::duration<double > calByCPU(int size, int* a, int* b, int* ans) {
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < size; i++) {
		i[ans] = i[a] + i[b];
	}
	auto end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast < std::chrono::duration<double >> (end - start);
}
//gpu版計算function
//計時器只計算扣掉搬data後的計算時間
std::chrono::duration<double > calByGPU(int size, int* a, int* b, int* ans) {
	int* gpuA, * gpuB, * gpuC;
	cudaMalloc((void**)&gpuA, size * sizeof(int));
	cudaMalloc((void**)&gpuB, size * sizeof(int));
	cudaMalloc((void**)&gpuC, size * sizeof(int));

	cudaMemcpy(gpuA, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuB, b, size * sizeof(int), cudaMemcpyHostToDevice);

	auto start = std::chrono::high_resolution_clock::now();
	deviceADD <<<SIZE / BLOCKSIZE, BLOCKSIZE >>> (gpuA, gpuB, gpuC);
	//<<<block, thread>>>
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();

	//<<<block, thread>>>
	cudaMemcpy(ans, gpuC, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(gpuA);
	cudaFree(gpuB);
	cudaFree(gpuC);
	return std::chrono::duration_cast <std::chrono::duration<double >> (end - start);

}
bool equal(int size, int* a, int* b) {
	for (int i = 0; i < size; i++) {
		if (i[a] != i[b]) {
			std::cout << i[a] << "\t" << i[b] << "\t" << i << std::endl;
			return false;
		}
		
	}
	return true;
}
void printArr(int size, int* ptr) {
	for (int i = 0; i < size; i++) {
		std::cout << i << ": " << i[ptr] << std::endl;
	}
}
void benchmark(int time, std::chrono::duration<double > (*func)(int, int*, int*, int*), int size, int *a , int *b , int * c) {
	std::cout << "start benchmark" << std::endl;
	std::cout << "Time\texecute Time" << std::endl;
	std::cout << "---------------" << std::endl;
	double total = 0;
	for (int i = 0; i < time; i++) {
		auto exeT = (*func)(size, a, b, c);
		std::cout << i + 1 << "\t" << exeT.count() << std::endl;
		total += exeT.count();
	}
	std::cout << "average : " << total / time << std::endl;
	std::cout << "------end------ " << std::endl;

}

int main(void) {
	srand(time(NULL));
	int* a;
	int* b;
	int* c_cpu;
	int* c_gpu;
	a = new int[SIZE];
	b = new int[SIZE];
	c_cpu = new int[SIZE];
	c_gpu = new int[SIZE];
	fillramdom(SIZE, a);
	fillramdom(SIZE, b);
	benchmark(10, calByCPU, SIZE, a, b, c_cpu);
	benchmark(10, calByGPU, SIZE, a, b, c_gpu);

	//calByCPU(SIZE, a, b, c_cpu);
	//calByGPU(SIZE, a, b, c_gpu);
	//printArr(SIZE, c_gpu);
	std::cout << (equal(SIZE, c_cpu, c_gpu) ? "True" : "False") << std::endl;
	delete[] a, b, c_cpu, c_gpu;
}


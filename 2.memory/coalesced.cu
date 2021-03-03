
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <iostream>

const size_t SIZE = 1024000;

const size_t BLOCKSIZE = 1024;
struct structOfArray {
	int* a;
	int* b;
	int* c;
	int* d;
	int* e;
	int* f;
	int* g;
	int* h;
};

struct arrayOfStruct {
	int a, b, c, d, e, f, g, h;
};

bool equal(size_t size, arrayOfStruct* aos, arrayOfStruct* cpu,structOfArray soa) {
	for (auto it = 0; it < size; it++) {
		if (aos[it].h != soa.h[it]) {
			std::cout << it << ": " << aos[it].h << "\t"<< soa.h[it] << "\t" << cpu[it].h << std::endl;
			return false;
		}
		if (cpu[it].h != soa.h[it]) {
			std::cout << it << ": " << aos[it].h << "\t" << soa.h[it] << "\t" << cpu[it].h << std::endl;
			return false;
		}
	}
	return true;
}
__global__ void computeSOA(structOfArray data) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	data.h[id] =
		data.a[id] +
		data.b[id] +
		data.c[id] +
		data.d[id] +
		data.e[id] +
		data.f[id] +
		data.g[id];
}
__global__ void computeAOS(arrayOfStruct* data) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	data[id].h =
		data[id].a +
		data[id].b +
		data[id].c +
		data[id].d +
		data[id].e +
		data[id].f +
		data[id].g;
}

void calbyCPU(size_t size, arrayOfStruct* data) {
	for (auto it = 0; it < size; it++) {
		data[it].h =
			data[it].a +
			data[it].b +
			data[it].c +
			data[it].d +
			data[it].e +
			data[it].f +
			data[it].g;
	}
}

int main(void) {
	srand(time(NULL));
	//initial variable
	int blocknum = SIZE / BLOCKSIZE;
	if (SIZE % BLOCKSIZE != 0)
		blocknum++;
	arrayOfStruct aos[SIZE];
	structOfArray soa;
	soa.a = new int[SIZE];
	soa.b = new int[SIZE];
	soa.c = new int[SIZE];
	soa.d = new int[SIZE];
	soa.e = new int[SIZE];
	soa.f = new int[SIZE];
	soa.g = new int[SIZE];
	soa.h = new int[SIZE];
	int num;
	for (auto it = 0; it < SIZE; it++) {
		num = rand() % 1000;
		soa.a[it] = num;
		aos[it].a = num;
	}
	for (auto it = 0; it < SIZE; it++) {
		num = rand() % 1000;
		soa.b[it] = num;
		aos[it].b = num;
	}
	for (auto it = 0; it < SIZE; it++) {
		num = rand() % 1000;
		soa.c[it] = num;
		aos[it].c = num;
	}
	for (auto it = 0; it < SIZE; it++) {
		num = rand() % 1000;
		soa.d[it] = num;
		aos[it].d = num;
	}
	for (auto it = 0; it < SIZE; it++) {
		num = rand() % 1000;
		soa.e[it] = num;
		aos[it].e = num;
	}
	for (auto it = 0; it < SIZE; it++) {
		num = rand() % 1000;
		soa.f[it] = num;
		aos[it].f = num;
	}
	for (auto it = 0; it < SIZE; it++) {
		num = rand() % 1000;
		soa.g[it] = num;
		aos[it].g = num;
	}
	

	//structOfArrayTest
	structOfArray device_soa;
	cudaMalloc((void**)&device_soa.a, SIZE * sizeof(int));
	cudaMalloc((void**)&device_soa.b, SIZE * sizeof(int));
	cudaMalloc((void**)&device_soa.c, SIZE * sizeof(int));
	cudaMalloc((void**)&device_soa.d, SIZE * sizeof(int));
	cudaMalloc((void**)&device_soa.e, SIZE * sizeof(int));
	cudaMalloc((void**)&device_soa.f, SIZE * sizeof(int));
	cudaMalloc((void**)&device_soa.g, SIZE * sizeof(int));
	cudaMalloc((void**)&device_soa.h, SIZE * sizeof(int));
	cudaMemcpy(device_soa.a, soa.a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_soa.b, soa.b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_soa.c, soa.c, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_soa.d, soa.d, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_soa.e, soa.e, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_soa.f, soa.f, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_soa.g, soa.g, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	computeSOA << <blocknum, BLOCKSIZE >> > (device_soa);
	cudaDeviceSynchronize();

	cudaMemcpy(soa.h, device_soa.h, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_soa.a);
	cudaFree(device_soa.b);
	cudaFree(device_soa.c);
	cudaFree(device_soa.d);
	cudaFree(device_soa.e);
	cudaFree(device_soa.f);
	cudaFree(device_soa.g);
	cudaFree(device_soa.h);

	//arrayofStruct test
	arrayOfStruct *device_aos;
	cudaMalloc((void**)&device_aos, SIZE * sizeof(arrayOfStruct));
	cudaMemcpy(device_aos, aos, SIZE * sizeof(arrayOfStruct), cudaMemcpyHostToDevice);
	computeAOS << <blocknum, BLOCKSIZE >> > (device_aos);
	cudaDeviceSynchronize();

	cudaMemcpy(aos, device_aos, SIZE * sizeof(arrayOfStruct), cudaMemcpyDeviceToHost);
	cudaFree(device_aos);


	//cpu
	arrayOfStruct* cpu = new arrayOfStruct[SIZE];
	for (auto it = 0; it < SIZE; it++) {
		cpu[it].a = aos[it].a;
		cpu[it].b = aos[it].b;
		cpu[it].c = aos[it].c;
		cpu[it].d = aos[it].d;
		cpu[it].e = aos[it].e;
		cpu[it].f = aos[it].f;
		cpu[it].g = aos[it].g;
	}
	calbyCPU(SIZE, cpu);
	std::cout << (equal(SIZE, aos, cpu, soa) ? "True" : "False") << std::endl;

}

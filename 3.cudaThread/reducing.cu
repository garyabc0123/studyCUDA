#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string.h>
#include <chrono>
#include "opencv4/opencv2/opencv.hpp"
#define PERBLOCK 256

double zero_calByCPU(float *ptr, int size){
    double total = 0;
    for(int i = 0 ; i < size ; i++){
        total = total + ptr[i];
    }
    return total;
}

//test 01- simple
__global__ void one_globalMemKernel(float *data, int stride, int size){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id + stride < size){
        data[id] += data[id + stride];
    }
}
float one_globalMem(float *ptr, int size){
    int block = (size + PERBLOCK - 1) / PERBLOCK;
    float *devPtr;
    cudaMalloc((void**)&devPtr, sizeof(float) * size);
    cudaMemcpy(devPtr,ptr,sizeof(float) * size, cudaMemcpyHostToDevice);
    for(int stride = 1 ; stride < size ; stride *= 2){
        one_globalMemKernel<<<block,PERBLOCK>>>(devPtr, stride, size);
        cudaDeviceSynchronize();
    }
    float ret;
    cudaMemcpy(&ret, devPtr, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devPtr);
    return ret;
}

//test 02-use share memory
__global__ void two_shareMemKernel(float *data, int size){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shareM[PERBLOCK];
    if(threadIdx.x > PERBLOCK)
        return;
    shareM[threadIdx.x] = (id < size) ? data[id] : 0.0f; // copy to shared memory
    __syncthreads();

    for(unsigned int i = 1 ; i < blockDim.x ; i *= 2){
        if(id % (2 * i) == 0){
            shareM[threadIdx.x] += shareM[threadIdx.x + i];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        data[blockIdx.x] = shareM[0];

    }

}

float two_shareMem(float *ptr, int size){
    float *devPtr;
    cudaMalloc((void**)&devPtr, sizeof(float) * size);
    cudaMemcpy(devPtr,ptr,sizeof(float) * size, cudaMemcpyHostToDevice);
    float ret = 0.f;
    while(size > 1){
        int block = (size + PERBLOCK - 1) / PERBLOCK;
        two_shareMemKernel<<<block, PERBLOCK>>>(devPtr,size);
        cudaDeviceSynchronize();
        size = block;


    }
    cudaMemcpy(&ret, devPtr, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devPtr);
    return ret;
}


//test 03-non Divergent
__global__ void three_nonDivergentKernel(float *data, int size){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shareM[PERBLOCK];
    if(threadIdx.x > PERBLOCK)
        return;
    shareM[threadIdx.x] = (id < size) ? data[id] : 0.0f; // copy to shared memory
    __syncthreads();

    for(unsigned int i = 1 ; i < blockDim.x ; i *= 2){
        //cal mod is slow
//        if(id % (2 * i) == 0){
//            shareM[threadIdx.x] += shareM[threadIdx.x + i];
//        }
        int index = 2 * i * threadIdx.x;
        if(index < blockDim.x){
            //printf("block: %d thread: %d index: %d\n",blockIdx.x, threadIdx.x,index);
            shareM[index] += shareM[index + i];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        data[blockIdx.x] = shareM[0];

    }
}
float three_nonDivergent(float *ptr, int size){
    float *devPtr;
    cudaMalloc((void**)&devPtr, sizeof(float) * size);
    cudaMemcpy(devPtr,ptr,sizeof(float) * size, cudaMemcpyHostToDevice);
    float ret = 0.f;
    while(size > 1){
        int block = (size + PERBLOCK - 1) / PERBLOCK;
        three_nonDivergentKernel<<<block, PERBLOCK>>>(devPtr,size);
        cudaDeviceSynchronize();
        size = block;


    }
    cudaMemcpy(&ret, devPtr, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devPtr);
    return ret;
}

//test 04-Sequential
__global__ void four_sequentialKernel(float *data, int size){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shareM[PERBLOCK];
    if(threadIdx.x > PERBLOCK)
        return;
    shareM[threadIdx.x] = (id < size) ? data[id] : 0.0f; // copy to shared memory
    __syncthreads();
    /*
    for(unsigned int i = 1 ; i < blockDim.x ; i *= 2){

        int index = 2 * i * threadIdx.x;
        if(index < blockDim.x){
            //printf("block: %d thread: %d index: %d\n",blockIdx.x, threadIdx.x,index);
            shareM[index] += shareM[index + i];
        }
        __syncthreads();
    }
    */
    for(unsigned int i = blockDim.x/2 ; i > 0 ; i>>=1){
        if(threadIdx.x < i){
            //printf("block: %d thread: %d i: %d\n",blockIdx.x, threadIdx.x, i);
            shareM[threadIdx.x] += shareM[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        data[blockIdx.x] = shareM[0];

    }
}
float  four_sequential(float *ptr, int size){
    float *devPtr;
    cudaMalloc((void**)&devPtr, sizeof(float) * size);
    cudaMemcpy(devPtr,ptr,sizeof(float) * size, cudaMemcpyHostToDevice);
    float ret = 0.f;
    while(size > 1){
        int block = (size + PERBLOCK - 1) / PERBLOCK;
        four_sequentialKernel<<<block, PERBLOCK>>>(devPtr,size);
        cudaDeviceSynchronize();
        size = block;


    }
    cudaMemcpy(&ret, devPtr, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devPtr);
    return ret;
}


int main(void){
    float *hostNum;
    unsigned int size = 2 << 28;
    hostNum = new float[size];
    srand(time(NULL));
    for(int i = 0 ; i < size ; i++){
        hostNum[i] =  100;
    }
    double cpu = zero_calByCPU(hostNum, size);
    float gpu01 = one_globalMem(hostNum, size);
    float gpu02 = two_shareMem(hostNum, size);
    float gpu03 = three_nonDivergent(hostNum,size);
    float gpu04 = four_sequential(hostNum,size);


    std::cout <<  cpu << std::endl;
    std::cout <<  gpu01 << std::endl;
    std::cout <<  gpu02 << std::endl;
    std::cout << gpu03 << std::endl;
    std::cout << gpu04 << std::endl;

    delete [] hostNum;
}

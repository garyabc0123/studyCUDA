#define CUDACC
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string.h>
#include <chrono>
#include "opencv4/opencv2/opencv.hpp"
#define BLOCKSIZE 256

texture<unsigned char, 1 , cudaReadModeElementType> text;

__global__ void calHisto(unsigned int* histoArr){


    __shared__ unsigned int histoP[256];
    //歸零，只需要256 threads
    if(threadIdx.x < 256)
        histoP[threadIdx.x] = 0;
    __syncthreads();


    int index  = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned char data = tex1Dfetch(text, index);
    if(data < 256 && data >= 0)
        atomicAdd(&(histoP[data]), 1);

    __syncthreads();

    if(threadIdx.x < 256){
        atomicAdd(&(histoArr[threadIdx.x]), histoP[threadIdx.x]);
    }


}

__global__ void withoutText(unsigned int* histoArr, unsigned char *img){
    __shared__ unsigned int histoP[256];
    //歸零，只需要256 threads
    if(threadIdx.x < 256)
        histoP[threadIdx.x] = 0;
    __syncthreads();
    int index  = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned char data = img[index];
    if(data < 256 && data >= 0)
        atomicAdd(&(histoP[data]), 1);

    __syncthreads();

    if(threadIdx.x < 256){
        atomicAdd(&(histoArr[threadIdx.x]), histoP[threadIdx.x]);
    }
}

int main(int argc, char** argv){

    cv::Mat orgImg =  cv::imread("/home/ascdc/Downloads/OpenCV_Logo_with_text.png",cv::IMREAD_GRAYSCALE);

    unsigned char *ptrImg = orgImg.isContinuous()? orgImg.data: orgImg.clone().data;

    /* test cupy
    for(auto it = 0 ; it < orgImg.total() ; it++){
        if(it[ptrImg] != orgImg.at<ushort>(it)){
            std::cout << it << " " << it[ptrImg] << " " << orgImg.at<ushort>(it) << std::endl;
        }
    }
    */

    unsigned char * devImg;
    unsigned int *devHistoArr;
    unsigned int *hostHistArr;
    cudaError err;
    hostHistArr = new unsigned int[256];


    auto start = std::chrono::high_resolution_clock::now();
    cudaMalloc((void**)& devImg, sizeof(unsigned char) * orgImg.total());
    cudaMalloc((void**)& devHistoArr, sizeof (unsigned int) * 256);
    err = cudaMemcpy(devImg,ptrImg,sizeof(unsigned char) * orgImg.total(), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
    }
    //bind texture with linear memory
    err = cudaBindTexture(0, text, devImg, sizeof(unsigned char) * orgImg.total());
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto period = std::chrono::duration_cast < std::chrono::duration<double >> (end - start);
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>( period ).count() << "us" << std::endl;



    dim3 block(orgImg.total() / BLOCKSIZE + 1 ,1,1);
    if(orgImg.total() % BLOCKSIZE != 0)
        block.x += 1;
    dim3 thread(BLOCKSIZE,1 ,1);

    start = std::chrono::high_resolution_clock::now();
    calHisto<<<block,thread>>>(devHistoArr);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    period = std::chrono::duration_cast < std::chrono::duration<double >> (end - start);
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>( period ).count() << "us" << std::endl;
    err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
    }



    err = cudaMemcpy(hostHistArr, devHistoArr, sizeof (unsigned int) * 256, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
    }

    for(auto it = 0 ; it < 256 ; it ++){
        std::cout << hostHistArr[it] << std::endl;
    }
    //Free
    cudaUnbindTexture(&text);
    cudaFree(devImg);
    cudaFree(devHistoArr);










    //test 2
    //without texture

    start = std::chrono::high_resolution_clock::now();
    cudaMalloc((void**)& devImg, sizeof(unsigned char) * orgImg.total());
    cudaMalloc((void**)& devHistoArr, sizeof (unsigned int) * 256);
    err = cudaMemcpy(devImg,ptrImg,sizeof(unsigned char) * orgImg.total(), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
    }
    end = std::chrono::high_resolution_clock::now();
    period = std::chrono::duration_cast < std::chrono::duration<double >> (end - start);
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>( period ).count() << "us" << std::endl;



    start = std::chrono::high_resolution_clock::now();
    withoutText<<<block,thread>>>(devHistoArr, devImg);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    period = std::chrono::duration_cast < std::chrono::duration<double >> (end - start);
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>( period ).count() << "us" << std::endl;
    err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
    }




    cudaFree(devImg);
    cudaFree(devHistoArr);


    return 0;
}

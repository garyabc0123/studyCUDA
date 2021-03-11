#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string.h>
#include <chrono>
#include "opencv4/opencv2/opencv.hpp"
texture<unsigned char, 2 , cudaReadModeElementType> textImg;
__global__ void convolution(
                            uchar *ans,
                            int8_t *mask,uint imgW, uint imgH){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x > imgW || y > imgH)
        return;
    int16_t total = 0;
    for (uint8_t i = 0 ; i < 3 ; i++){
        for(uint8_t j = 0 ; j < 3 ; j++){
            total += (int16_t)tex2D(textImg, x+i, y+j) * mask[i + j * 3];
        }
    }
    if(total < 0){
        total = 0;
    }
    __syncthreads();
    ans[x  + y * imgW] = total;
}

__global__ void noTextConvolution(uchar *ans, int8_t *mask, uint imgW, uint imgH, uchar * img){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x > imgW || y > imgH)
        return;
    int16_t total = 0;
    for (uint8_t i = 0 ; i < 3 ; i++){
        for(uint8_t j = 0 ; j < 3 ; j++){
            total += (int16_t)img[(x+i) +  (y+j) * imgW] * mask[i + j * 3];
        }
    }
    if(total < 0){
        total = 0;
    }
    __syncthreads();
    ans[x  + y * imgW] = total;
}

int main() {
    cv::Mat orgImg =  cv::imread("/home/ascdc/Downloads/1280px-RoadEcologyConference2017-17.jpg",cv::IMREAD_GRAYSCALE);
    cv::imshow("G",orgImg);
    unsigned char *ptrImg = orgImg.isContinuous()? orgImg.data: orgImg.clone().data;

    //use cudaChannelFormatDesc define struct type
    //cudaChannelFormatDesc chDesc = cudaCreateChannelDesc(8,8,0,0,cudaChannelFormatKindUnsigned);
    //or
    cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<uchar>();
    cudaArray * cuArr;
    cudaError err;
    err = cudaMallocArray(&cuArr, &chDesc, orgImg.cols, orgImg.rows);
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    err = cudaMemcpyToArray(cuArr, 0, 0, ptrImg, sizeof(uchar) * orgImg.total(), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    //binding
    cudaBindTextureToArray(&textImg, cuArr, &chDesc);

    //alloc mask space
    int8_t mask[9] = {-1,-1,-1,
                      -1,8,-1,
                      -1,-1,-1};
    int8_t* devMask;
    cudaMalloc((void**)&devMask, sizeof(int8_t) * 9);
    err = cudaMemcpy(devMask, mask, sizeof(int8_t) * 9, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
        return -1;
    }


    //alloc output space
    uchar *output = new uchar[orgImg.total()];
    uchar *devOut;
    cudaMalloc((void**)&devOut, sizeof(uchar) * orgImg.total());




    dim3 block(orgImg.cols / 32 + 1, orgImg.rows / 32 + 1);
    dim3 thread(32,32,1);
    for(auto i = 0 ; i < 100 ; i++){
        convolution<<<block,thread>>>(devOut, devMask, orgImg.cols, orgImg.rows);

    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(output, devOut, sizeof(uchar) * orgImg.total() , cudaMemcpyDeviceToHost);
    cudaFree(devOut);
    cv::Mat outMat(orgImg.rows,orgImg.cols,CV_8UC1, (unsigned char*)output);
    cv::imwrite("text.png",outMat);

    delete []output;
    outMat.release();
    cudaUnbindTexture(&textImg);
    cudaFree(cuArr);





    //test 2
    //without texture
    uchar *devImg;
    cudaMalloc((void**)& devImg, sizeof(uchar) * orgImg.total());
    cudaMemcpy(devImg, ptrImg, sizeof(uchar) * orgImg.total() , cudaMemcpyHostToDevice);
    output = new uchar[orgImg.total()];
    cudaMalloc((void**)&devOut, sizeof(uchar) * orgImg.total());
    for(auto i = 0 ; i < 100 ; i++) {

        noTextConvolution<<<block, thread>>>(devOut, devMask, orgImg.cols, orgImg.rows, devImg);
    }
    cudaDeviceSynchronize();
    err = cudaMemcpy(output, devOut, sizeof(uchar) * orgImg.total() , cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
    }
    cudaFree(devOut);
    outMat = cv::Mat(orgImg.rows,orgImg.cols,CV_8UC1, (unsigned char*)output);
    cv::imwrite("nontext.png",outMat);

    cudaFree(devImg);
    cudaFree(devOut);
    delete []output;


    cudaFree(devMask);
    orgImg.release();
    return 0;
}

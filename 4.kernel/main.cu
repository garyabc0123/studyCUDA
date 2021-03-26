#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string.h>
#include <chrono>
#include "memory.h"
#include "pthread.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "opencv4/opencv2/opencv.hpp"
///home/ascdc/Downloads/OpenCV_Logo_with_text.png /home/ascdc/Downloads/1280px-RoadEcologyConference2017-17.jpg /home/ascdc/Downloads/PIA23137.png /home/ascdc/Downloads/PIA23623_M34.tif
__global__ void Convolution(uchar *ans, int8_t *mask, uint imgW, uint imgH, uchar * img, int channel){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x > imgW || y > imgH)
        return;
    y = y * 3 + channel;
    int16_t total = 0;
    for (uint8_t i = 0 ; i < 3 ; i++){
        for(uint8_t j = 0 ; j < 3 ; j++){
            total += (int16_t)img[(x + i) +  (y+j*3) * imgW] * mask[i + j * 3];
        }
    }
    if(total < 0){
        total = 0;
    }
    __syncthreads();
    ans[x  + y * imgW] = total;
}

int main(int argc, char ** argv) {
    if(argc < 2){
        return -1;
    }
    cv::Mat *srcImg = new cv::Mat[argc - 1];
    unsigned char **pinnedPtrImg = new unsigned char *[argc - 1];
    uchar **devImg = new uchar *[argc - 1];
    unsigned int *imgSize = new unsigned int[argc - 1];
    unsigned int *imgW = new unsigned int[argc - 1];
    unsigned int *imgH = new unsigned int[argc - 1];
    cudaStream_t *stream = new cudaStream_t[argc - 1];
    uchar **devAns = new uchar *[argc - 1];

    int8_t mask[9] = {-1,-1,-1,
                      -1,8,-1,
                      -1,-1,-1};
    int8_t* devMask;
    cudaMalloc((void**)&devMask, sizeof(int8_t) * 9);
    cudaError err = cudaMemcpy(devMask, mask, sizeof(int8_t) * 9, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    for(int i = 1 ; i < argc ; i++){
        srcImg[i - 1] = cv::imread(argv[i]);
    }
    for(int i = 0 ; i < argc - 1 ; i++){
        cudaStreamCreate(&stream[i]);
        imgSize[i] = srcImg[i].total() * 3;
        imgW[i] = srcImg[i].cols;
        imgH[i] = srcImg[i].rows;
    }

    for(int i = 0 ; i < argc - 1 ;i++){
        cudaStreamCreate(&stream[i]);

        cudaMallocHost((void**)&pinnedPtrImg[i], sizeof(unsigned char) * imgSize[i]);
        memcpy(pinnedPtrImg[i], srcImg[i].data, sizeof(unsigned char) * imgSize[i]);
        srcImg[i].release();

        cudaMalloc((void**)&devImg[i], sizeof(unsigned char) * imgSize[i]);
        cudaMalloc((void**)&devAns[i], sizeof(unsigned char) * imgSize[i]);

        cudaMemcpyAsync(devImg[i], pinnedPtrImg[i] , sizeof(unsigned char) * imgSize[i],cudaMemcpyHostToDevice, stream[i]);

        //kernel
        dim3 block(imgW[i] / 32 + 1, imgH[i] / 32 + 1);
        dim3 thread(32,32,1);
        Convolution<<<block, thread , 0 , stream[i]>>>(devAns[i], devMask, imgW[i], imgH[i], devImg[i] , 0);
        Convolution<<<block, thread , 0 , stream[i]>>>(devAns[i], devMask, imgW[i], imgH[i], devImg[i] , 1);
        Convolution<<<block, thread , 0 , stream[i]>>>(devAns[i], devMask, imgW[i], imgH[i], devImg[i] , 2);


        cudaMemcpyAsync(pinnedPtrImg[i], devAns[i] , sizeof(unsigned char) * imgSize[i],cudaMemcpyDeviceToHost, stream[i]);


    }




    for(int i = 0 ; i < argc - 1 ; i++){
        cudaStreamSynchronize(stream[i]);
        auto outMat = cv::Mat(imgH[i],imgW[i],CV_8UC3, (unsigned char*)pinnedPtrImg[i]);
        cv::imwrite(std::to_string(i) + ".png",outMat);
        cudaStreamDestroy(stream[i]);
        cudaFreeHost(pinnedPtrImg[i]);
        cudaFree(devImg);
        cudaFree(devAns);
    }
    delete [] srcImg;
    delete [] pinnedPtrImg;
    delete [] imgSize;
    delete [] devImg;
    delete [] stream;
    delete [] devAns;
    delete [] imgW;
    delete [] imgH;
    return 0;
}

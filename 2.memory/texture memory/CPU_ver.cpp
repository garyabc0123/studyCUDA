#include <iostream>
#include "opencv4/opencv2/opencv.hpp"
#include <time.h>
#include <chrono>


int main() {
    cv::Mat orgImg =  cv::imread("/home/ascdc/Downloads/OpenCV_Logo_with_text.png",cv::IMREAD_GRAYSCALE);
    unsigned char *ptrImg = orgImg.isContinuous()? orgImg.data: orgImg.clone().data;
    unsigned int histoArr[256] = {0};

    auto start = std::chrono::high_resolution_clock::now();
    for(auto it = 0 ; it < orgImg.total() ; it++){
        if(it[ptrImg] < 256){
            histoArr[it[ptrImg]] ++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto period = std::chrono::duration_cast < std::chrono::duration<double >> (end - start);
    //std::cout << period.count() << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>( period ).count() << "ms" << std::endl;
    return 0;
}

#include <stdlib.h>
#include <string.h>
#include "RuntimeAPI.h"
#include <string>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
//using namespace cv;
//#include "/usr/include/opencv4/opencv2/opencv.hpp"

template <typename T>
class Neuropl{
public:
    /* The constructor function*/ 
    Neuropl(std::string& path); 
    //T predict(std::vector<uint8_t>& image);
    T predict(cv::Mat& image);

private:
    std::string model_path;

    void* runtime;
    /* The constructor function*/ 

    /* Should be called once per neuropl initialization. */
    void setup(); 
    void setModelPath(std::string path);
    
    //T predict(std::vector<uint8_t>& image);
    //T predict(cv::Mat& image);

    /* Functions for testing pruposes. */
    void print_attributes();
};
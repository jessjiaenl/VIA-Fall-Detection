#include <stdlib.h>
#include <string.h>
#include "RuntimeAPI.h"
#include <string>
#include <iostream>
#include <stdio.h>
#include <vector>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <opencv2/opencv.hpp>

namespace bp = boost::python;
namespace np = boost::python::numpy;

template <typename T>
class Neuropl{
public:
    /* The constructor function*/ 
    Neuropl(std::string path); 
    //T predict(std::vector<uint8_t>& image);
    T predict(np::ndarray image);
    //T predict(cv::Mat& image)

    /* Functions for testing pruposes. Will remove later. */
    void print_attributes();
    void setModelPath(std::string path);

private:

    std::string model_path;
    void* runtime;
    /* Should be called once per neuropl initialization. */
    void setup(); 

};

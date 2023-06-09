#include <stdlib.h>
#include <string.h>
#include "RuntimeAPI.h"
#include <string>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <dlfcn.h>
#include <getopt.h>
#include <cstdlib>

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
    void setModelPath(std::string path);
    /* Functions for testing pruposes. Will remove later. */
    void print_attributes();

private:

    std::string model_path;
    size_t required_size;
    void* runtime;
    void * handle;
    /* Should be called once per neuropl initialization. */
    void setup(); 
    void * load_func(void * handle, const char * func_name);

};

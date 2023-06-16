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
    Neuropl(std::string path, int inputLen, int outputLen); 
    void setModelPath(std::string path);
    /* Functions for testing pruposes. Will remove later. */
    void print_attributes();

    np::ndarray predict(np::ndarray image, int inputLen);;
    std::vector<std::vector<T>> predict(cv::Mat mat);

private:
    
    std::string model_path;
    size_t required_size;
    void* runtime;
    void * handle;

    /* Should be called once per neuropl initialization. */
    void * load_func(void * handle, const char * func_name);

    typedef int (*NeuronRuntime_create)(const EnvOptions* options, void** runtime);
    typedef int (*NeuronRuntime_loadNetworkFromFile)(void* runtime, const char* pathToDlaFile);
    typedef int (*NeuronRuntime_loadNetworkFromBuffer)(void* runtime, const void* buffer, size_t size);
    typedef int (*NeuronRuntime_setInput)(void* runtime, uint64_t handle, const void* buffer, size_t length, BufferAttribute attr);
    typedef int (*NeuronRuntime_setSingleInput)(void* runtime, const void* buffer, size_t length, BufferAttribute attr);
    typedef int (*NeuronRuntime_setOutput)(void* runtime, uint64_t handle, void* buffer, size_t length, BufferAttribute attr);
    typedef int (*NeuronRuntime_setSingleOutput)(void* runtime, void* buffer, size_t length, BufferAttribute attr);
    typedef int (*NeuronRuntime_setQoSOption)(void* runtime, const QoSOptions* qosOption);
    typedef int (*NeuronRuntime_getInputSize)(void* runtime, uint64_t handle, size_t* size);
    typedef int (*NeuronRuntime_getSingleInputSize)(void* runtime, size_t* size);
    typedef int (*NeuronRuntime_getOutputSize)(void* runtime, uint64_t handle, size_t* size);
    typedef int (*NeuronRuntime_getSingleOutputSize)(void* runtime, size_t* size);
    typedef int (*NeuronRuntime_getProfiledQoSData)(void* runtime, ProfiledQoSData** profiledQoSData, uint8_t* execBoostValue);
    typedef int (*NeuronRuntime_inference)(void* runtime);
    typedef void (*NeuronRuntime_release)(void* runtime);

    NeuronRuntime_create rt_create;
    NeuronRuntime_loadNetworkFromFile loadNetworkFromFile;
    NeuronRuntime_setInput setInput;
    NeuronRuntime_setSingleInput setSingleInput;
    NeuronRuntime_setOutput setOutput;
    NeuronRuntime_setSingleOutput setSingleOutput;
    NeuronRuntime_setQoSOption setQoSOption;
    NeuronRuntime_inference inference;
    NeuronRuntime_release release;
    NeuronRuntime_getInputSize getInputSize;
    NeuronRuntime_getSingleInputSize getSingleInputSize;
    NeuronRuntime_getOutputSize getOutputSize;
    NeuronRuntime_getSingleOutputSize getSingleOutputSize;
    NeuronRuntime_getProfiledQoSData getProfiledQoSData;
};

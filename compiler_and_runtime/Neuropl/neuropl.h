#include <dlfcn.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "RuntimeAPI.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

class Neuropl {
public:
    /* The constructor function */
    Neuropl(std::string path);
    /* The destructor function */
    ~Neuropl();

    /* Functions */
    bp::list predict(np::ndarray data);
    const std::vector<std::vector<uint8_t>>& predict(uint8_t* byte_buffer);
    void print_profiled_qos_data();

private:
    std::string model_path;

    size_t input_size;
    void* runtime;
    void* handle;
    std::vector<std::vector<uint8_t>> output_buf;

    /* Should be called once per neuropl initialization. */
    void* load_func(void* handle, const char* func_name);

    /* typedef all the NeuronRuntime library functions */
    typedef int (*NeuronRuntime_create)(const EnvOptions* options,
        void** runtime);
    typedef int (*NeuronRuntime_loadNetworkFromFile)(void* runtime,
        const char* pathToDlaFile);
    typedef int (*NeuronRuntime_loadNetworkFromBuffer)(void* runtime,
        const void* buffer,
        size_t size);
    typedef int (*NeuronRuntime_setInput)(void* runtime, uint64_t handle,
        const void* buffer, size_t length,
        BufferAttribute attr);
    typedef int (*NeuronRuntime_setSingleInput)(void* runtime, const void* buffer,
        size_t length,
        BufferAttribute attr);
    typedef int (*NeuronRuntime_setOutput)(void* runtime, uint64_t handle,
        void* buffer, size_t length,
        BufferAttribute attr);
    typedef int (*NeuronRuntime_setSingleOutput)(void* runtime, void* buffer,
        size_t length,
        BufferAttribute attr);
    typedef int (*NeuronRuntime_setQoSOption)(void* runtime,
        const QoSOptions* qosOption);
    typedef int (*NeuronRuntime_getInputSize)(void* runtime, uint64_t handle,
        size_t* size);
    typedef int (*NeuronRuntime_getSingleInputSize)(void* runtime, size_t* size);
    typedef int (*NeuronRuntime_getOutputSize)(void* runtime, uint64_t handle,
        size_t* size);
    typedef int (*NeuronRuntime_getSingleOutputSize)(void* runtime, size_t* size);
    typedef int (*NeuronRuntime_getProfiledQoSData)(
        void* runtime, ProfiledQoSData** profiledQoSData,
        uint8_t* execBoostValue);
    typedef int (*NeuronRuntime_inference)(void* runtime);
    typedef void (*NeuronRuntime_release)(void* runtime);
    typedef int (*NeuronRuntime_getOutputPaddedDimensions)(
        void* runtime, uint64_t handle, RuntimeAPIDimensions* dims);

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
    NeuronRuntime_getOutputPaddedDimensions getOutputPaddedDimensions;
};

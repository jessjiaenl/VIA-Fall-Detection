#include "neuropl.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

/* Constructor function. */
/* Specify both loop_count and num_results. */

/* Using default arguements for loop_count and num_results*/
template <typename T>
Neuropl<T>::Neuropl(std::string path){ 
    std::cout << "constructor " << std::endl;
    model_path = path;
    setup();
}

/* Function implementations. */

/* This function should set up all the needed enviroments for the Neuron Runtime. */
template <typename T>
void Neuropl<T>::setup(){
    std::cout << "setup " << std::endl;
    Py_Initialize();
    np::initialize();
    /* I realized I cannot set up here or else all the functions will go out o scope in predict. */
}

/* These function is for testing purposes only. */
template <typename T>
void Neuropl<T>::print_attributes(void){
    std::cout << "print_attributes " << std::endl;
    std::cout << "model_path: " + model_path << std::endl;

}

template <typename T>
void Neuropl<T>::setModelPath(std::string path){
    std::cout << "SetModelPath " << std::endl;
    model_path = path;

}

//T predict(uint8_t* image) {
template <typename T>
T Neuropl<T>::predict(np::ndarray image) {    
    std::cout << "predict " << std::endl;
    //uint8_t* ptr = image.data;
    /* Call all the NeuroRuntime functions just like how runtime.cpp does. */
     // typedef to the functions pointer signatures.
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
    /* Functions from  /usr/lib/libneuron_runtime.so. */

    // Open the share library
    handle = dlopen("/usr/lib/libneuron_runtime.so", RTLD_LAZY);
    if (handle == nullptr) {
        std::cerr << "Failed to open /usr/lib/libneuron_runtime.so." << std::endl;
        exit(1);
    }

        // Setup the environment options for the Neuron Runtime
    EnvOptions envOptions = {
        .deviceKind = kEnvOptHardware,
        .MDLACoreOption = 1,  // Force single MDLA
        .suppressInputConversion = false,  // Please refer to the explanation above
        .suppressOutputConversion = false,  // Please refer to the explanation above
    };

    // Setup the QoS options for the Neuron Runtime
    // Please refer to the explanation above for details
    QoSOptions qosOptions = {
        NEURONRUNTIME_PREFER_PERFORMANCE,  // Preference
        NEURONRUNTIME_PRIORITY_MED,  // Priority
        NEURONRUNTIME_BOOSTVALUE_MAX,  // Boost value
        0,  // Deadline
        0,  // Abort time
        nullptr,  // Profiled QoS Data
    };
    // Declare function pointer to each functions,
    // and load the function address into function pointer
    #define LOAD_FUNCTIONS(FUNC_NAME, VARIABLE_NAME) \
            FUNC_NAME VARIABLE_NAME = reinterpret_cast<FUNC_NAME>(load_func(handle, #FUNC_NAME));
        LOAD_FUNCTIONS(NeuronRuntime_create, rt_create)
        LOAD_FUNCTIONS(NeuronRuntime_loadNetworkFromFile, loadNetworkFromFile)
        LOAD_FUNCTIONS(NeuronRuntime_setInput, setInput)
        LOAD_FUNCTIONS(NeuronRuntime_setSingleInput, setSingleInput)
        LOAD_FUNCTIONS(NeuronRuntime_setOutput, setOutput)
        LOAD_FUNCTIONS(NeuronRuntime_setSingleOutput, setSingleOutput)
        LOAD_FUNCTIONS(NeuronRuntime_setQoSOption, setQoSOption);
        LOAD_FUNCTIONS(NeuronRuntime_inference, inference)
        LOAD_FUNCTIONS(NeuronRuntime_release, release)
        LOAD_FUNCTIONS(NeuronRuntime_getInputSize, getInputSize)
        LOAD_FUNCTIONS(NeuronRuntime_getSingleInputSize, getSingleInputSize)
        LOAD_FUNCTIONS(NeuronRuntime_getOutputSize, getOutputSize)
        LOAD_FUNCTIONS(NeuronRuntime_getSingleOutputSize, getSingleOutputSize)
        LOAD_FUNCTIONS(NeuronRuntime_getProfiledQoSData, getProfiledQoSData);
    #undef LOAD_FUNCTIONS

    // Step 1. Create neuron runtime environment
    int err_code = (*rt_create)(&envOptions, &runtime);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to create Neuron runtime." << std::endl;
        exit(3);
    }
    // Step 2. Load the compiled network(*.dla) from file
    /* c_str(): converts a string to an array of characters and terminates this array with a null character at the end*/
    err_code = (*loadNetworkFromFile)(runtime, model_path.c_str());
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to load network from file." << std::endl;
        exit(3);
    }

    // Check the required input buffer size
    err_code = (*getSingleInputSize)(runtime, &required_size);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to get single input size for network." << std::endl;
        exit(3);
    }
    std::cout << "The required size of the input buffer is " << required_size << std::endl;

    //uint8_t* byte_buffer = new uint8_t[image.get_data()];
    np::ndarray npArray = bp::extract<np::ndarray>(image);

    // Get the buffer pointer and size
    uint8_t* byte_buffer = reinterpret_cast<uint8_t*>(npArray.get_data());


    // Set the input buffer with our memory buffer (pixels inside)
    BufferAttribute attr {-1};
    err_code = (*setSingleInput)(runtime, static_cast<void *>(byte_buffer), required_size, attr);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to set single input for network." << std::endl;
        exit(3);
    }
    
    // (Optional) Check the required output buffer size
    err_code = (*getSingleOutputSize)(runtime, &required_size);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to get single output size for network." << std::endl;
        exit(3);
    }
    
    std::vector<std::vector<unsigned char>> ret;
    //std::vector<std::vector<uint8_t>> outbuf;
    ret.push_back(std::vector<unsigned char>(required_size));
    unsigned char* out_buf = ret[0].data();

    std::cout << "The required size of the output buffer is " << required_size << std::endl;

    // Step 4. Set the output buffer
    //unsigned char * out_buf = new unsigned char[1001];
    err_code = (*setSingleOutput)(runtime, static_cast<void *>(out_buf), 1001, attr);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to set single output for network." << std::endl;
        exit(3);
    }

    // (Optional) Set QoS Option
    err_code = (*setQoSOption)(runtime, &qosOptions);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to set QoS option, using default setting instead" << std::endl;
    }

    // Step 5. Do the inference with Neuron Runtime
    err_code = (*inference)(runtime);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to inference the input." << std::endl;
        exit(3);
    }

    // (Optional) Get profiled QoS Data
    // Neuron Rutime would allocate ProfiledQoSData instance when the input profiledQoSData is nullptr.
    /* QoS stands for Quality of Service. We want these numbers. */
    ProfiledQoSData* profiledQoSData = nullptr;
    uint8_t executingBoostValue;
    err_code = (*getProfiledQoSData)(runtime, &profiledQoSData, &executingBoostValue);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to Get QoS Data" << std::endl;
    }

    // (Optional) Print out profiled QoS Data and executing boost value
    if (profiledQoSData != nullptr) {
        std::cout << "Dump the profiled QoS Data:" << std::endl;
        std::cout << "executing boost value = " << +executingBoostValue << std::endl;
        for (uint32_t i = 0u; i < profiledQoSData->numSubgraph; ++i) {
            for (uint32_t j = 0u; j < profiledQoSData->numSubCmd[i]; ++j) {
                std::cout << "SubCmd[" << i << "][" << j << "]:" << std::endl;
                std::cout << "execution time = " << profiledQoSData->qosData[i][j].execTime << std::endl;
                std::cout << "boost value = " << +profiledQoSData->qosData[i][j].boostValue << std::endl;
                std::cout << "bandwidth = " << profiledQoSData->qosData[i][j].bandwidth << std::endl;
            }
        }
    } else {
        std::cerr << "profiledQoSData is nullptr" << std::endl;
    }

    // Step 6. Release the runtime resource
    (*release)(runtime);

    //result.push_back(std::vector<T>{out_buf});
    // result.push_back(std::vector<T>{out_buf1});
    // result.push_back(std::vector<T>{out_buf2});
    // result.push_back(std::vector<T>{out_buf3});
   return ret;
   //return T {};
}

template <typename T>
void *  Neuropl<T>::load_func(void * handle, const char * func_name) {
    /* Load the function specified by func_name, and exit if the loading is failed. */
    void * func_ptr = dlsym(handle, func_name); /* Find the run-time address in the shared object HANDLE refers to of the symbol called NAME.  */

    if (func_name == nullptr) {
        std::cerr << "Find " << func_name << " function failed." << std::endl;
        exit(2);
    }
    return func_ptr;
}

/* For C++ */
// template <typename T>
// T Neuropl<T>::predict(cv::Mat& image) {    
//     //uint8_t* ptr = image.data;
//     std::cout << "predict " << std::endl;
//     return T {};
// }

//template <typename T>
//T Neuropl<T>::predict(std::vector& image) {    
    //uint8_t* ptr = image.data;
//    return T {};
//}

int main(void){
    // Py_Initialize();
    // np::initialize();
    std::cout << "Using 2 default arguments" << std::endl;
    //neuropl *n = new neuropl("I scream :O");
    typedef std::vector<std::vector<unsigned char>> outfmt;

    std::string model_path {"model.dla"};
    
    /* 2 ways to call a function in C++. */
    Neuropl<outfmt> model{model_path};
    Neuropl<outfmt> m2 = Neuropl<outfmt>(model_path);
    model.print_attributes();
    //std::vector<uint8_t> output {10};
    const int rows = 224;
    const int cols = 224;
    // Create a NumPy shape tuple
    bp::tuple shape = bp::make_tuple(rows, cols);
    const np::dtype dtype = np::dtype::get_builtin<unsigned char>();
    np::ndarray img = np::zeros(shape, dtype);
    //std::vector<std::vector<uint8_t>> output = model.predict(image); 
    //output.size;

    outfmt output = model.predict(img);

    // for (auto v : output) {
    //     for (auto vv : v) {
    //         std::cout << vv << " ";
    //     }
    //     std::cout << std::endl;
    // }

}


BOOST_PYTHON_MODULE(neuropl)
{
    using namespace boost::python;
    
    class_<Neuropl<uint8_t>>("Neuropl",  init<std::string>())
         .def("predict", &Neuropl<uint8_t>::predict)
         .def("print_attributes", &Neuropl<uint8_t>::print_attributes)
         .def("setModelPath", &Neuropl<uint8_t>::setModelPath)
        ;
}
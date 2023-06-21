#include "neuropl.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

/* Disables error messages */
int disable(FILE* stream)
{
    int saved = dup(fileno(stream));
    int tmpfd = ::open("/dev/null", O_WRONLY);
    ::dup2(tmpfd, fileno(stream));
    ::close(tmpfd);

    return saved;
}

/* Restores error messages */
void restore(FILE* stream, int saved)
{
    dup2(saved, fileno(stream));
    close(saved);
}

/* Constructor function. */
Neuropl::Neuropl(std::string path){ 

    /* Setup NeuroRuntime */
    model_path = path;
    Py_Initialize();
    np::initialize();
    // Open the shared library
    handle = dlopen("/usr/lib/libneuron_runtime.so", RTLD_LAZY);
    if (handle == nullptr) {
        throw std::runtime_error("Failed to open /usr/lib/libneuron_runtime.so.");
    }

    // Load all the NeuroRuntime functions
    #define LOAD_FUNCTIONS(FUNC_NAME, VARIABLE_NAME) \
        VARIABLE_NAME = reinterpret_cast<FUNC_NAME>(load_func(handle, #FUNC_NAME));

    LOAD_FUNCTIONS(NeuronRuntime_create, rt_create);
    LOAD_FUNCTIONS(NeuronRuntime_loadNetworkFromFile, loadNetworkFromFile);
    LOAD_FUNCTIONS(NeuronRuntime_setInput, setInput);
    LOAD_FUNCTIONS(NeuronRuntime_setSingleInput, setSingleInput);
    LOAD_FUNCTIONS(NeuronRuntime_setOutput, setOutput);
    LOAD_FUNCTIONS(NeuronRuntime_setSingleOutput, setSingleOutput);
    LOAD_FUNCTIONS(NeuronRuntime_setQoSOption, setQoSOption);
    LOAD_FUNCTIONS(NeuronRuntime_inference, inference);
    LOAD_FUNCTIONS(NeuronRuntime_release, release);
    LOAD_FUNCTIONS(NeuronRuntime_getInputSize, getInputSize);
    LOAD_FUNCTIONS(NeuronRuntime_getSingleInputSize, getSingleInputSize);
    LOAD_FUNCTIONS(NeuronRuntime_getOutputSize, getOutputSize);
    LOAD_FUNCTIONS(NeuronRuntime_getSingleOutputSize, getSingleOutputSize);
    LOAD_FUNCTIONS(NeuronRuntime_getProfiledQoSData, getProfiledQoSData);
    LOAD_FUNCTIONS(NeuronRuntime_getOutputPaddedDimensions, getOutputPaddedDimensions);

    // Setup the environment options for the Neuron Runtime
    EnvOptions envOptions = {
        .deviceKind = kEnvOptHardware,
        .MDLACoreOption = 1,  // Force single MDLA
        .suppressInputConversion = false,  // Please refer to the explanation above
        .suppressOutputConversion = false,  // Please refer to the explanation above
    };

    // Setup the QoS options for the Neuron Runtime
    // Please refer to RuntimeAPI.h above for details
    QoSOptions qosOptions = {
        NEURONRUNTIME_PREFER_PERFORMANCE,  // Preference
        NEURONRUNTIME_PRIORITY_MED,  // Priority
        NEURONRUNTIME_BOOSTVALUE_MAX,  // Boost value
        0,  // Deadline
        0,  // Abort time
        nullptr,  // Profiled QoS Data
    };

    // Create neuron runtime environment
    int err_code = (*rt_create)(&envOptions, &runtime);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        throw std::runtime_error("Failed to create Neuron runtime.");
    }

    // Load the compiled network(*.dla) from file
    err_code = (*loadNetworkFromFile)(runtime, model_path.c_str());
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        throw std::runtime_error("Failed to load network from file.");
    }

    // Check the required input buffer size
    err_code = (*getSingleInputSize)(runtime, &input_size);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        throw std::runtime_error("Failed to get single input size for network.");
    }
    std::cout << "The required size of the input buffer is " << input_size << std::endl;

    BufferAttribute attr {-1};
    //Set the output buffer
    for(int i = 0; i < 10; i++){
        size_t size;
        int fd = disable(stderr);
        err_code = (*getOutputSize)(runtime, i, &size);
        if (err_code != NEURONRUNTIME_NO_ERROR) {
            break;
        }
        output_buf.push_back(std::vector<uint8_t>(size));
        restore(stderr, fd);
        err_code = (*setOutput)(runtime, i, static_cast<void *>(output_buf[i].data()), size, attr);
        if (err_code != NEURONRUNTIME_NO_ERROR) {
            throw std::runtime_error("Failed to set single output for network.");
        }
    }

}

/*Destructor function */
Neuropl::~Neuropl() {
    // Releases runtime resources
    (*release)(runtime);
}

/* Function implementations */

/* This function converts a vector<vector<usigned char>> into a python list of ndarrays. */
bp::list convertToListOfNdarrays(const std::vector<std::vector<unsigned char>>& data)
{
    typename std::vector<std::vector<unsigned char>>::const_iterator iter;
    boost::python::list list;
    for (iter = data.begin(); iter != data.end(); ++iter) {
        np::ndarray ndarray = np::zeros(bp::make_tuple((*iter).size()), np::dtype::get_builtin<unsigned char>());
        for (std::size_t i = 0; i < (*iter).size(); ++i)
        {
            *reinterpret_cast<unsigned char*>(ndarray.get_data() + i * ndarray.strides(0)) = (*iter)[i];
        }
        list.append(ndarray);
    }
    return list;
}

/* For C++ users */
const std::vector<std::vector<uint8_t>> & Neuropl::predict(uint8_t* byte_buffer)
{
    // Check for invalid arguement
    if(byte_buffer == nullptr){
        throw std::invalid_argument("byte_buffer is nullptr");
    }   

    int err_code;
    // Set the input buffer with our memory buffer (pixels inside)
    BufferAttribute attr {-1};
    err_code = (*setSingleInput)(runtime, static_cast<void *>(byte_buffer), input_size, attr);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        throw std::runtime_error("Failed to set single input for network.");
    }

    // Do the inference with Neuron Runtime
    err_code = (*inference)(runtime);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        throw std::runtime_error("Failed to inference the input.");
    }

    // Get profiled QoS Data
    // Neuron Rutime would allocate ProfiledQoSData instance when the input profiledQoSData is nullptr.
    ProfiledQoSData* profiledQoSData = nullptr;
    uint8_t executingBoostValue;
    err_code = (*getProfiledQoSData)(runtime, &profiledQoSData, &executingBoostValue);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to Get QoS Data" << std::endl;
    }
    
    // Print out profiled QoS Data and executing boost value
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

	return output_buf;
    
}

/* For Python users */
bp::list Neuropl::predict(np::ndarray data) {    
    // Verify input size
    np::ndarray npArray = bp::extract<np::ndarray>(data);
    int array_size = 1;
    for(int i = 0; i < data.get_nd(); i++){
        array_size*= (int)(npArray.shape(i));
    }

    if(array_size != input_size){
        std::cerr << "Actual input size =  "<< array_size  <<"Expected input size = "<< input_size<< std::endl;
        throw std::invalid_argument("Input dimension mismatch.");
    }

    // Get the buffer pointer and size
    uint8_t* byte_buffer = reinterpret_cast<uint8_t*>(npArray.get_data());
    auto ret = Neuropl::predict(byte_buffer);
    // Convert ret from vector vector to 2d list of numpy arrays
    auto retNumpy = convertToListOfNdarrays(ret);
    return retNumpy;
}

void *  Neuropl::load_func(void * handle, const char * func_name) {
    /* Load the function specified by func_name, and exit if the loading failed. */
    void * func_ptr = dlsym(handle, func_name); /* Find the run-time address in the shared object HANDLE refers to of the symbol called NAME. */

    if (func_name == nullptr) {
        std::cerr << "Find " << func_name << " function failed." << std::endl;
        exit(2);
    }
    return func_ptr;
}

/* Export using Boost.Python */
BOOST_PYTHON_MODULE(neuropl)
{
    using namespace boost::python;

    bp::list (Neuropl::*predict)(np::ndarray data) = &Neuropl::predict;

    class_<Neuropl>("Neuropl",  init<std::string>())
         .def("predict", predict)
        ;
}

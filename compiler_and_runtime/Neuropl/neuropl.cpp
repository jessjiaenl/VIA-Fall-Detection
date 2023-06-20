#include "neuropl.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

/* Constructor function. */

/* Using default arguements for loop_count and num_results*/
Neuropl::Neuropl(std::string path, int num_of_inputs, int num_of_outputs):
	num_of_inputs (num_of_inputs),
	num_of_outputs (num_of_outputs)
{ 
    std::cout << "constructor " << std::endl;
    model_path = path;
    Py_Initialize();
    np::initialize();
    // Open the share library
    handle = dlopen("/usr/lib/libneuron_runtime.so", RTLD_LAZY);
    if (handle == nullptr) {
        std::cerr << "Failed to open /usr/lib/libneuron_runtime.so." << std::endl;
        return;
    }

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

    /* Call all the NeuroRuntime functions just like how runtime.cpp does. */
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
    err_code = (*getSingleInputSize)(runtime, &input_size);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to get single input size for network." << std::endl;
        exit(3);
    }
    std::cout << "The required size of the input buffer is " << input_size << std::endl;

}

/*Destructor function */
Neuropl::~Neuropl() {
    /* Releases runtime resources */
    std::cout << "Destructor invoked" << std::endl;
    (*release)(runtime);
}

/* Function implementations. */

/* These function is for testing purposes only. */
void Neuropl::print_attributes(void){
    std::cout << "print_attributes " << std::endl;
    std::cout << "model_path: " + model_path << std::endl;

}

void Neuropl::setModelPath(std::string path){
    std::cout << "SetModelPath " << std::endl;
    model_path = path;

}

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

np::ndarray convertToNdarray(const std::vector<std::vector<unsigned char>>& data)
{
    // Get the shape of the data
    const std::size_t rows = data.size();
    const std::size_t cols = data.empty() ? 0 : data[0].size();

    // Create a NumPy ndarray with the same shape
    np::ndarray ndarray = np::zeros(bp::make_tuple(rows, cols), np::dtype::get_builtin<unsigned char>());

    // Copy the data to the ndarray
    for (std::size_t i = 0; i < rows; ++i)
    {
        const std::vector<unsigned char>& row = data[i];
        for (std::size_t j = 0; j < cols; ++j)
        {
            *reinterpret_cast<unsigned char*>(ndarray.get_data() + i * ndarray.strides(0) + j * ndarray.strides(1)) = row[j];
        }
    }
    return ndarray;
}

/* For C++ users */
template <typename T>
std::vector<std::vector<T>> Neuropl::predict(uint8_t* byte_buffer)
{
    //uint8_t* byte_buffer;
    int err_code;
    // Set the input buffer with our memory buffer (pixels inside)
    BufferAttribute attr {-1};
    err_code = (*setSingleInput)(runtime, static_cast<void *>(byte_buffer), input_size, attr);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to set single input for network." << std::endl;
        exit(3);
    }

    std::vector<std::vector<T>> ret;
    
    T* out_buf;
    for(int i = 0; i < 10; i++){
        err_code = (*getOutputSize)(runtime, i, &required_size);
        if (err_code != NEURONRUNTIME_NO_ERROR) {
            break;
        }
        std::vector<T> vect;
        vect.push_back(required_size);
        std::cout << "Output size " << i <<" is " << required_size << std::endl;
        ret.push_back(vect);
        out_buf = ret[i].data();
        err_code = (*setOutput)(runtime, i, static_cast<void *>(out_buf), required_size, attr);
        if (err_code != NEURONRUNTIME_NO_ERROR) {
            std::cerr << "Failed to set single output for network." << std::endl;
            exit(3);
        }
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
    //(*release)(runtime);

	return ret;
}

/* For Python users */
bp::list Neuropl::predict(np::ndarray image) {    
    std::cout << "predict " << std::endl;

    /* Verify input size. */
    np::ndarray npArray = bp::extract<np::ndarray>(image);
    int array_size = 1;
    for(int i = 0; i < image.get_nd(); i++){
        std::cout<< i << std::endl;
        array_size*= (int)(npArray.shape(i));
    }

    if(array_size != input_size){
        std::cerr << "Actual input size =  "<< array_size  <<"Expected input size = "<< input_size<< std::endl;
        std::cerr << "Input dimension mismatch. " << std::endl;
        exit(1);
    }

    // Get the buffer pointer and size
    uint8_t* byte_buffer = reinterpret_cast<uint8_t*>(npArray.get_data());

    auto ret = Neuropl::predict<unsigned char>(byte_buffer);
    // Step 7. convert ret from vector vector to 2d numpy array
    auto retNumpy = convertToListOfNdarrays(ret);
    
    return retNumpy;
}

void *  Neuropl::load_func(void * handle, const char * func_name) {
    /* Load the function specified by func_name, and exit if the loading is failed. */
    void * func_ptr = dlsym(handle, func_name); /* Find the run-time address in the shared object HANDLE refers to of the symbol called NAME. */

    if (func_name == nullptr) {
        std::cerr << "Find " << func_name << " function failed." << std::endl;
        exit(2);
    }
    return func_ptr;
}

int main(void){
    std::cout << "Welcome" << std::endl;

    std::string model_path {"./../model1.dla"};
    Neuropl model {model_path, 1, 2};
    
    // C++ example
    cv::Mat image(224, 224, CV_8UC3);
    // Get the buffer pointer and size
    uint8_t* byte_buffer = reinterpret_cast<uint8_t*>(image.data);
    auto result = model.predict<uint8_t>(byte_buffer);

    for (auto v : result) {
        for (auto vv : v) {
            std::cout << vv << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

BOOST_PYTHON_MODULE(neuropl)
{
    using namespace boost::python;

    bp::list (Neuropl::*predict)(np::ndarray image) = &Neuropl::predict;

    class_<Neuropl>("Neuropl",  init<std::string, int, int>())
         .def("predict", predict)
        ;
}

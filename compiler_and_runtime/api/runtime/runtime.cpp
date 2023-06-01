/*
 * Copyright (C) 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <getopt.h>
#include <cstdlib>

#include "RuntimeAPI.h"

struct Settings {
    int loop_count = 1;
    std::string model_path;
    std::string input_bin_path;
    std::string labels_path;
    int number_of_results = 1;
};

Settings process_options(int argc, char** argv) {
    Settings s;
    int c = 0;
    static struct option long_options[] = {{"count", required_argument, nullptr, 'c'},
                                           {"image_bin", required_argument, nullptr, 'i'},
                                           {"labels", required_argument, nullptr, 'l'},
                                           {"tflite_model", required_argument, nullptr, 'm'},
                                           {"num_results", required_argument, nullptr, 'r'},
                                           {nullptr, 0, nullptr, 0}};

    while (1) {
        /* getopt_long stores the option index here. */
        int option_index = 0;
        c = getopt_long(argc, argv, "c:i:l:m:r:", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1) break;

        switch (c) {
            case 'c':
                s.loop_count = strtol(optarg, nullptr, 10);
                break;
            case 'i':
                s.input_bin_path = optarg;
                break;
            case 'l':
                s.labels_path = optarg;
                break;
            case 'm':
                s.model_path = optarg;
                break;
            case 'r':
                s.number_of_results = strtol(optarg, nullptr, 10);
                break;
            default:
                break;
        }
    }
    return s;
}

void * load_func(void * handle, const char * func_name) {
    /* Load the function specified by func_name, and exit if the loading is failed. */
    void * func_ptr = dlsym(handle, func_name);

    if (func_name == nullptr) {
        std::cerr << "Find " << func_name << " function failed." << std::endl;
        exit(2);
    }
    return func_ptr;
}

int main(int argc, char * argv[]) {
    void * handle;
    void * runtime;

    Settings s = process_options(argc, argv);
    if (s.model_path.empty() || s.input_bin_path.empty() || s.labels_path.empty()) {
        std::cerr << "please set model_path and input_bin" << std::endl;
        exit(1);
    }

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

    // read input
    std::ifstream file(s.input_bin_path);
    if (!file) {
        std::cerr << "Input bin not found" << std::endl;
        exit(1);
    }

    // Read file into buffer
    file.seekg(0, std::ios::end);
    std::size_t input_file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    uint8_t* byte_buffer = new uint8_t[input_file_size];
    if (byte_buffer == nullptr) {
        std::cerr << "Fail to allocate buffer to read input bin" << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(byte_buffer), input_file_size);
    file.close();

    // Open the share library
    handle = dlopen("lib/libneuron_runtime.so", RTLD_LAZY);
    if (handle == nullptr) {
        std::cerr << "Failed to open lib/libneuron_runtime.so." << std::endl;
        exit(1);
    }

    // Setup the environment options for the Neuron Runtime
    EnvOptions envOptions = {
        .deviceKind = kEnvOptHardware,
        .suppressInputConversion = false,  // Please refer to the explanation above
        .suppressOutputConversion = false,  // Please refer to the explanation above
        .MDLACoreOption = 1,  // Force single MDLA
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
    err_code = (*loadNetworkFromFile)(runtime, s.model_path.c_str());
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to load network from file." << std::endl;
        exit(3);
    }

    // (Optional) Check the required input buffer size
    size_t required_size;
    err_code = (*getSingleInputSize)(runtime, &required_size);
    if (err_code != NEURONRUNTIME_NO_ERROR) {
        std::cerr << "Failed to get single input size for network." << std::endl;
        exit(3);
    }
    std::cout << "The required size of the input buffer is " << required_size << std::endl;

    // Step 3. Set the input buffer with our memory buffer (pixels inside)
    BufferAttribute attr {-1};
    err_code = (*setSingleInput)(runtime, static_cast<void *>(byte_buffer), 3 * 224 * 224, attr);
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
    std::cout << "The required size of the output buffer is " << required_size << std::endl;

    // Step 4. Set the output buffer
    unsigned char * out_buf = new unsigned char[1001];
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

    // NOTE: profiledQoSData would be destroyed after NeuronRuntime_release.
    // You should never de-reference profiledQoSData again.

    // We check the inference result by finding the highest score in the 1001 classes
    unsigned char top = out_buf[0];
    size_t top_idx = 0;
    for (size_t i = 1; i < 1001; ++i) {
        if (out_buf[i] > top) {
            top = out_buf[i];
            top_idx = i;
        }
    }

    // read labels
    std::ifstream file_lables(s.labels_path);
    if (!file_lables) {
        std::cerr << "File " << s.labels_path.c_str() << " not found" << std::endl;
        exit(1);
    }
    std::string line;
    size_t idx = 0;
    while (std::getline(file_lables, line)) {
        if (idx == top_idx) {
            break;
        }
        idx++;
    }
    file_lables.close();

    std::cout << "The top index is " << top_idx << ": " << line.c_str() << std::endl;
    return 0;
}
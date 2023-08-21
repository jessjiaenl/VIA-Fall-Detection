/*
 * Copyright (C) 2019 The Android Open Source Project
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

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <sys/cdefs.h>

// The Runtime user should include this header to use Runtime API.
// Note that some APIs that set input and output info need the user to specify the handle of
// the input/output tensor that he/she wants to set. The user may
// 1) Acts as ANN or TFLite, which always know the handle
// 2) Run a precompiled network. The user should understand the model in the beginning.
// 3) Run a precompiled network without knowing what the network look like. In this case,
//    it is impossible for the user to do inference without taking a glance at the network
//    IO map info. Otherwise, the user cannot even give a valid input with valid input shape.
//    After the user checks the IO map, they would also acquire the handle and the corresponding
//    shape.

__BEGIN_DECLS

typedef enum {
    N = 0,
    H = 1,
    W = 2,
    C = 3,
    Invalid = C + 1,
    DimensionSize = Invalid,
} RuntimeAPIDimIndex;

// Execution preference.
typedef enum {
    // Prefer performance.
    NEURONRUNTIME_PREFER_PERFORMANCE = 0,
    // Prefer low power.
    NEURONRUNTIME_PREFER_POWER,
} RuntimeAPIQoSPreference;

// Task priority.
typedef enum {
    NEURONRUNTIME_PRIORITY_LOW = 0,
    NEURONRUNTIME_PRIORITY_MED,
    NEURONRUNTIME_PRIORITY_HIGH,
} RuntimeAPIQoSPriority;

// Special boost value hint.
typedef enum {
    // Hint to notify the underlying scheduler to use the profiled boost value.
    NEURONRUNTIME_BOOSTVALUE_PROFILED = 101,
    NEURONRUNTIME_BOOSTVALUE_MAX = 100,
    NEURONRUNTIME_BOOSTVALUE_MIN = 0,
} RuntimeAPIQoSBoostValue;

// Raw data for QoS configuration.
// All of those fields should be filled with the profiled data.
typedef struct {
    // Profiled execution time: the profiled execution time (in usec).
    uint64_t execTime;
    // Suggested time: the suggested time (in msec).
    uint32_t suggestedTime;
    // Profled bandwidh: the profiled bandwidh (in MB/s).
    uint32_t bandwidth;
    // Profiled boost value: the profiled executing boost value (range in 0 to 100).
    uint8_t boostValue;
} QoSData;

// Maintain the profiled QoS raw data.
typedef struct {
    // Maintain profiled QoS raw data in a pointer of pointer.
    // This field could be nullptr if there is no previous profiled data.
    QoSData** qosData;
    // Number of sub-command in *qosData.
    // This field could be nullptr if there is no previous profiled data.
    uint32_t* numSubCmd;
    // Number of subgraph.
    // This field should be zero if there is no previous profiled data.
    uint32_t numSubgraph;
} ProfiledQoSData;

// QoS Option for configuration.
typedef struct {
    // Execution preference: NEURONRUNTIME_PREFER_PERFORMANCE or NEURONRUNTIME_PREFER_POWER.
    RuntimeAPIQoSPreference preference;
    // Task priority: NEURONRUNTIME_PRIORITY_HIGH, NEURONRUNTIME_PRIORITY_MED,
    // or NEURONRUNTIME_PRIORITY_LOW.
    RuntimeAPIQoSPriority priority;
    // Boost value hint: hint for the device frequency, ranged between 0 (lowest) to 100 (highest).
    // This value is the hint for baseline boost value in the underlying scheduler, which sets
    // the executing boost value (the actual boot value set in device) based on scheduling policy.
    // For the inferences with preference set as NEURONRUNTIME_PREFER_PERFORMANCE, scheduler
    // guarantees that the executing boost value would not be lower than the boost value hint.
    // On the other hand, for the inferences with preference set as NEURONRUNTIME_PREFER_POWER,
    // scheduler would try to save power by configuring the executing boost value with some value
    // that is not higher than the boost value hint.
    uint8_t boostValue;
    // Deadline: deadline for the inference (in msec).
    // Setting any non-zero value would nofity the underlying scheduler that this inference is
    // a real-time task.
    // This field should be left zero, unless this inference is a real-time task.
    uint16_t deadline;
    // Abort time: the maximun inference time for the inference (in msec).
    // If the inference is not completed before the abort time, the underlying scheduler would
    // abort the inference.
    // This field should be left zero, unless you wish to abort the inference.
    uint16_t abortTime;
    // Profiled QoS Data: pointer to the historical QoS data of previous inferences.
    // If there is no profiled data, this field could be left nullptr.
    // For the details, please check the ProfiledQoSData part.
    ProfiledQoSData* profiledQoSData;
} QoSOptions;

typedef struct {
    uint32_t deviceKind;

    // Set MDLA core option.
    // MDLACoreOption = 0 : Scheduler decide
    // MDLACoreOption = 1 : Force single MDLA
    // MDLACoreOption = 2 : Force multi MDLA
    int MDLACoreOption;

    // Set this to true to bypass preprocess and feed data in the format that the device demands.
    bool suppressInputConversion;

    // Set this to true to bypass postprocess and retrieve raw device output.
    bool suppressOutputConversion;
} EnvOptions;

typedef struct {
    int ionFd;  // -1: Non-ION buffer.
} BufferAttribute;

typedef struct {
    uint32_t dimensions[RuntimeAPIDimIndex::DimensionSize];
} RuntimeAPIDimensions;

#define NON_ION_FD -1

// For unsigned char deviceKind;
const unsigned char kEnvOptNullDevice = 1 << 0;
const unsigned char kEnvOptCModelDevice = 1 << 1;
const unsigned char kEnvOptHardware = 1 << 2;

// Return 1 to indicate user-specified EnvOptions use a NullDevice. Otherwise, return 0.
inline int IsNullDevice(const EnvOptions* option) {
    return option->deviceKind & kEnvOptNullDevice;
}

// Return 1 to indicate user-specified EnvOptions use a CModelDevice. Otherwise, return 0.
inline int IsCModelDevice(const EnvOptions* option) {
    return option->deviceKind & kEnvOptCModelDevice;
}

// Return 1 to indicate user-specified EnvOptions use a MDLADevice. Otherwise, return 0.
inline int IsHardware(const EnvOptions* option) {
    return option->deviceKind & kEnvOptHardware;
}

typedef enum {
    NEURONRUNTIME_NO_ERROR        = 0,
    NEURONRUNTIME_OUT_OF_MEMORY   = 1,
    NEURONRUNTIME_INCOMPLETE      = 2,
    NEURONRUNTIME_UNEXPECTED_NULL = 3,
    NEURONRUNTIME_BAD_DATA        = 4,
    NEURONRUNTIME_BAD_STATE       = 5,
    NEURONRUNTIME_RUNTIME_ERROR   = 6,
} RuntimeAPIErrorCode;

// Create a Neuron Runtime based on the setting specified in options. The address of the created
// instance will be passed back in *runtime. The return value indicate whether the creation
// succeeds.
int NeuronRuntime_create(const EnvOptions* options, void** runtime);

// Load network from file.
int NeuronRuntime_loadNetworkFromFile(void* runtime, const char* pathToDlaFile);

// Load network from memory buffer.
int NeuronRuntime_loadNetworkFromBuffer(void* runtime, const void* buffer, size_t size);

// Set the buffer of the tensor which hold the specified input handle in the original network.
int NeuronRuntime_setInput(void* runtime, uint64_t handle, const void* buffer, size_t length,
                           BufferAttribute attr);

// If there is only one input, this function sets its buffer.
// Otherwise, NEURONRUNTIME_INCOMPLETE is returned.
int NeuronRuntime_setSingleInput(void* runtime, const void* buffer, size_t length,
                                 BufferAttribute attr);

// Set the buffer of the tensor which hold the specified output handle in the original network.
int NeuronRuntime_setOutput(void* runtime, uint64_t handle, void* buffer, size_t length,
                            BufferAttribute attr);

// If there is only one output, this function sets its buffer.
// Otherwise, NEURONRUNTIME_INCOMPLETE is returned.
int NeuronRuntime_setSingleOutput(void* runtime, void* buffer, size_t length,
                                  BufferAttribute attr);

// Set the QoS configuration for Neuron Runtime.
// If qosOption.profiledQoSData is not nullptr,
// Neuron Runtime would use it as the profiled QoS data.
int NeuronRuntime_setQoSOption(void* runtime, const QoSOptions* qosOption);

// Pass back the expected buffer size (byte) in *size for the tensor which holds the specified
// input handle.
int NeuronRuntime_getInputSize(void* runtime, uint64_t handle, size_t* size);

// If there is only one input, this function passes back the expected size (byte) of its buffer
// in *size. Otherwise, NEURONRUNTIME_INCOMPLETE is returned.
int NeuronRuntime_getSingleInputSize(void* runtime, size_t* size);

// Pass back the expected buffer size (byte) in *size for the tensor which holds the specified input
// handle. The value in *size has been aligned to hardware required size, and it can be used as
// ION buffer size for the specified input when suppressInputConversion is enabled.
int NeuronRuntime_getInputPaddedSize(void* runtime, uint64_t handle, size_t* size);

// If there is only one input, this function passes back the expected size (byte) of its buffer in
// *size. The value in *size has been aligned to hardware required size, and it can be used as ION
// buffer size for input when suppressInputConversion is enabled. Otherwise, the returned value is
// NEURONRUNTIME_INCOMPLETE.
int NeuronRuntime_getSingleInputPaddedSize(void* runtime, size_t* size);

// Pass back the expected size (in pixels) of each dimensions in *dim for the tensor which holds the
// specified input handle. The sizes of each dimensions in *dim have been aligned to hardware
// required sizes. When suppressInputConversion is enabled, the values in *dim are the required
// sizes of each dimensions for the specified input.
int NeuronRuntime_getInputPaddedDimensions(void* runtime, uint64_t handle,
                                           RuntimeAPIDimensions* dims);

// If there is only one input, this function passes back the expected size (in pixels) of each
// dimensions in *dim. The sizes of each dimensions in *dim have been aligned to hardware required
// sizes. when suppressInputConversion is enabled, the values in *dim are the required sizes of each
// dimensions for input. Otherwise NEURONRUNTIME_INCOMPLETE is returned.
int NeuronRuntime_getSingleInputPaddedDimensions(void* runtime, RuntimeAPIDimensions* dims);

// Pass back the expected buffer size (byte) in *size for the tensor which holds the specified
// output handle.
int NeuronRuntime_getOutputSize(void* runtime, uint64_t handle, size_t* size);

// If there is only one Output, this function passes back the expected size (byte) of its buffer
// in *size. Otherwise, NEURONRUNTIME_INCOMPLETE is returned.
int NeuronRuntime_getSingleOutputSize(void* runtime, size_t* size);

// Pass back the expected buffer size (byte) in *size for the tensor which holds the specified
// output handle. The value in *size has been aligned to hardware required size, and it can be used
// as ION buffer size for the specified output when suppressOutputConversion is enabled.
int NeuronRuntime_getOutputPaddedSize(void* runtime, uint64_t handle, size_t* size);

// If there is only one Output, this function passes back the expected size (byte) of its buffer in
// *size. The value in *size has been aligned to hardware required size, and it can be used as ION
// buffer size for output when suppressOutputConversion is enabled. Otherwise, the returned value is
// NEURONRUNTIME_INCOMPLETE.
int NeuronRuntime_getSingleOutputPaddedSize(void* runtime, size_t* size);

// Pass back the expected size (in pixels) of each dimensions in *dim for the tensor which holds the
// specified output handle. The sizes of each dimensions in *dim have been aligned to hardware
// required sizes. When suppressOutputConversion is enabled, the values in *dim are the required
// sizes of each dimensions for the specified output.
int NeuronRuntime_getOutputPaddedDimensions(void* runtime, uint64_t handle,
                                            RuntimeAPIDimensions* dims);

// If there is only one Output, this function passes back the expected size (in pixels) of each
// dimensions in *dim. The sizes of each dimensions in *deim have been aligned to hardware required
// sizes. When suppressOutputConversion is enabled, the values in *dim are the required sizes of
// each dimensions for output. Otherwise, NEURONRUNTIME_INCOMPLETE is returned.
int NeuronRuntime_getSingleOutputPaddedDimensions(void* runtime, RuntimeAPIDimensions* dims);

// Get the profiled QoS data and executing boost value (the actual boost value during execution).
// If *profiledQoSData is nullptr, Neuron Runtime would allocate *profiledQoSData.
// Otherwise, Neuron Runtime would only update its fields.
// *profiledQoSData is actually allocated as a smart pointer in Neuron Runtime instance,
// so the lifetime of *profiledQoSData is the same as Neuron Runtime.
// Caller should be careful about the usage of *profiledQoSData,
// and never touch the allocated *profiledQoSData after NeuronRuntime_release.
int NeuronRuntime_getProfiledQoSData(void* runtime, ProfiledQoSData** profiledQoSData,
                                     uint8_t* execBoostValue);

// Do inference.
int NeuronRuntime_inference(void* runtime);

// Release Neuron Runtime.
void NeuronRuntime_release(void* runtime);

__END_DECLS

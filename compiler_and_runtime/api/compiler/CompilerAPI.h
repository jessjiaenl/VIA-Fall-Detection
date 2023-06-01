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

#pragma once

#include <stdint.h>

__BEGIN_DECLS

typedef enum {
    NCC_NO_ERROR                = 0,
    NCC_INVALID_MODEL           = 1,
    NCC_INVALID_OPTIONS         = 2,
    NCC_UNEXPECTED_NULL         = 3,
    NCC_BAD_DATA                = 4,
    NCC_BAD_STATE               = 5,
    NCC_COMPILATION_ERROR       = 6,
} CompilerAPIErrorCode;

typedef struct {
    // Specify available deviceKind for compiler.
    uint64_t targetArchs;

    // Hint compiler with the size of L1 memory.
    uint32_t l1MemorySizeKb;

    // Hint compiler with the size of L2 memory.
    uint32_t l2MemorySizeKb;

    // Hint compiler to use <num> MDLA cores. For normal cases, this value is 1.
    uint8_t mdlaCoreNum;

    // Hint compiler to run fp32 models using fp16. This option doesn't change input and output
    // data types. For a fp32 models, it still needs fp32 input data in input buffer and returns
    // fp32 output to output buffer.
    bool relaxFp32ToFp16;

    // Rewrite input and output data types to FP16 for FP32 models. Neuron users need to prepare
    // fp16 input data if a fp32 model is compiled with rewriteFp32ToFp16 == true. Also, the
    // data type of output in output buffer is fp16 when this options is true.
    bool rewriteFp32ToFp16;

    // Disable EDMA target explicitly.
    bool disableEDMA;

    // Suppress the input conversion. set this option false if you don't know the input data
    // format requirement the MDLA needs.
    bool suppressInputConversion;

    // Suppress the Output conversion. set this option false if you don't know the input data
    // format requirement the MDLA needs.
    bool suppressOutputConversion;

    // Show tensors and nodes in the tflite model.
    bool showNetwork;

    // Show execution plan.
    bool showExecPlan;

    // Show memory allocation summary.
    bool showMemorySummary;

    // Verbose mode.
    bool verboseMode;
} NeuronCompilerOptions;

//===------------------------------------------------------------------------------------------===//
// Available Targets
//===------------------------------------------------------------------------------------------===//
// Use MDLA 1.5.
const uint64_t kArchMDLA_1_5    = 1 << 0;

// Enable VPU with uint8. No floating point support.
const uint64_t kArchVPU         = 1 << 2;

// Only available when VPU on the platform has uint8 and fp16 capability.
const uint64_t kArchVPU_FPU     = 1 << 3;

// Enable CPU with TFLite framework.
const uint64_t kArchTFLite_CPU  = 1 << 4;

// Enable CPU with ARM NEON implementation.
const uint64_t kArchNeon        = 1 << 5;

// Enable GPUNN.
const uint64_t kArchGPU         = 1 << 6;


//===------------------------------------------------------------------------------------------===//
// Compilation API
//===------------------------------------------------------------------------------------------===//
// Compile model.tflite to model.dla
CompilerAPIErrorCode NeuronCompiler_compileNetworkFromFile(const char* model, const char* dla,
                                                                    const NeuronCompilerOptions* options);

__END_DECLS

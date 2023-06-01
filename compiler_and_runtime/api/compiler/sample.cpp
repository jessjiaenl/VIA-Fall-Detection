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

#include "CompilerAPI.h"

#include <iostream>
#include <fstream>
#include <dlfcn.h>

typedef CompilerAPIErrorCode
(*FnNeuronCompiler_compileNetworkFromFile)(const char* model, const char* dla,
                                                const NeuronCompilerOptions* options);

static FnNeuronCompiler_compileNetworkFromFile fnNeuronCompiler_compileNetworkFromFile;

void* LoadLib(const char* name) {
    auto handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        std::cerr << "Unable to open Neuron Runtime library " << dlerror();
        exit(3);
    }
    return handle;
}

void* GetLibHandle() {
    return LoadLib("lib/libncc_compiler.so");
}

inline void* LoadFunc(void* libHandle, const char* name) {
    if (libHandle == nullptr) {
        std::cerr << "Invalid library handle " << dlerror();
        exit(3);
    }

    void* fn = dlsym(libHandle, name);
    if (fn == nullptr) {
        std::cerr << "Unable to open Neuron Runtime function [" << name
                  << "] Because " << dlerror();
        exit(3);
    }
    return fn;
}

int main() {
    // load api from libncc_compiler.so
    const auto libHandle = GetLibHandle();
#define LOAD(name) fn##name = reinterpret_cast<Fn##name>(LoadFunc(libHandle, #name))
    LOAD(NeuronCompiler_compileNetworkFromFile);

    // Configure options.
    NeuronCompilerOptions options;
    options.targetArchs = kArchVPU;
    options.l1MemorySizeKb = -1;
    options.l2MemorySizeKb = -1;
    options.mdlaCoreNum = 0;
    options.relaxFp32ToFp16 = false;
    options.rewriteFp32ToFp16 = false;
    options.disableEDMA = false;
    options.suppressInputConversion = false;
    options.suppressOutputConversion = false;
    options.showNetwork = false;
    options.showExecPlan = false;
    options.showMemorySummary = false;
    options.verboseMode = false;

    // Compile model.tflite to model.dla
    CompilerAPIErrorCode error_code =
        fnNeuronCompiler_compileNetworkFromFile("/data/local/tmp/model/mobilenet_v1_1.0_224_quant.tflite", "/data/local/tmp/model/mobilenet_v1_1.0_224_quant.dla", &options);

    if (error_code != NCC_NO_ERROR) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

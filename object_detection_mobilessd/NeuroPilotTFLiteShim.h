/* Copyright Statement:
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws. The information contained herein
 * is confidential and proprietary to MediaTek Inc. and/or its licensors.
 * Without the prior written permission of MediaTek inc. and/or its licensors,
 * any reproduction, modification, use or disclosure of MediaTek Software,
 * and information contained herein, in whole or in part, shall be strictly prohibited.
 */
/* MediaTek Inc. (C) 2018. All rights reserved.
 *
 * BY OPENING THIS FILE, RECEIVER HEREBY UNEQUIVOCALLY ACKNOWLEDGES AND AGREES
 * THAT THE SOFTWARE/FIRMWARE AND ITS DOCUMENTATIONS ("MEDIATEK SOFTWARE")
 * RECEIVED FROM MEDIATEK AND/OR ITS REPRESENTATIVES ARE PROVIDED TO RECEIVER ON
 * AN "AS-IS" BASIS ONLY. MEDIATEK EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE OR NONINFRINGEMENT.
 * NEITHER DOES MEDIATEK PROVIDE ANY WARRANTY WHATSOEVER WITH RESPECT TO THE
 * SOFTWARE OF ANY THIRD PARTY WHICH MAY BE USED BY, INCORPORATED IN, OR
 * SUPPLIED WITH THE MEDIATEK SOFTWARE, AND RECEIVER AGREES TO LOOK ONLY TO SUCH
 * THIRD PARTY FOR ANY WARRANTY CLAIM RELATING THERETO. RECEIVER EXPRESSLY ACKNOWLEDGES
 * THAT IT IS RECEIVER'S SOLE RESPONSIBILITY TO OBTAIN FROM ANY THIRD PARTY ALL PROPER LICENSES
 * CONTAINED IN MEDIATEK SOFTWARE. MEDIATEK SHALL ALSO NOT BE RESPONSIBLE FOR ANY MEDIATEK
 * SOFTWARE RELEASES MADE TO RECEIVER'S SPECIFICATION OR TO CONFORM TO A PARTICULAR
 * STANDARD OR OPEN FORUM. RECEIVER'S SOLE AND EXCLUSIVE REMEDY AND MEDIATEK'S ENTIRE AND
 * CUMULATIVE LIABILITY WITH RESPECT TO THE MEDIATEK SOFTWARE RELEASED HEREUNDER WILL BE,
 * AT MEDIATEK'S OPTION, TO REVISE OR REPLACE THE MEDIATEK SOFTWARE AT ISSUE,
 * OR REFUND ANY SOFTWARE LICENSE FEES OR SERVICE CHARGE PAID BY RECEIVER TO
 * MEDIATEK FOR SUCH MEDIATEK SOFTWARE AT ISSUE.
 *
 * The following software/firmware and/or related documentation ("MediaTek Software")
 * have been modified by MediaTek Inc. All revisions are subject to any receiver's
 * applicable license agreements with MediaTek Inc.
 */

#ifndef ANDROID_ML_NN_RUNTIME_NEURO_PILOT_TFLITE_SHIM_H
#define ANDROID_ML_NN_RUNTIME_NEURO_PILOT_TFLITE_SHIM_H


#if __ANDROID_API__ >= __ANDROID_API_O_MR1__

#include <dlfcn.h>
#include <android/log.h>
#include <vector>

#define TFLITE_LOG(format, ...) \
    __android_log_print(ANDROID_LOG_ERROR, "NeuroPilotTFLiteShim", format "\n", ##__VA_ARGS__);

#define LOAD_TFLITE_FUNCTION(name) \
  static name##_fn fn = reinterpret_cast<name##_fn>(loadTFLiteFunction(#name));

#define EXECUTE_TFLITE_FUNCTION(...) \
  if (fn != nullptr) {        \
    fn(__VA_ARGS__);          \
  }

#define EXECUTE_TFLITE_FUNCTION_RETURN_INT(...) \
    return fn != nullptr ? fn(__VA_ARGS__) : ANEURALNETWORKS_BAD_STATE;

#define EXECUTE_TFLITE_FUNCTION_RETURN_BOOL(...) \
    return fn != nullptr ? fn(__VA_ARGS__) : false;

#define EXECUTE_TFLITE_FUNCTION_RETURN_POINTER(...) \
    return fn != nullptr ? fn(__VA_ARGS__) : nullptr;

/************************************************************************************************/

typedef struct ANeuralNetworksTFLite ANeuralNetworksTFLite;
typedef struct TfLiteContext TfLiteContext;

#ifndef TENSORFLOW_CONTRIB_LITE_CONTEXT_H_
typedef enum { kTfLiteOk = 0, kTfLiteError = 1 } TfLiteStatus;
typedef struct TfLiteNode TfLiteNode;
typedef struct {
  int size;
  int data[];
} TfLiteIntArray;

typedef struct {
    // Initializes the op from serialized data.
    // If a built-in op:
    //   `buffer` is the op's params data (TfLiteLSTMParams*).
    //   `length` is zero.
    // If custom op:
    //   `buffer` is the op's `custom_options`.
    //   `length` is the size of the buffer.
    //
    // Returns a type-punned (i.e. void*) opaque data (e.g. a primitive pointer
    // or an instance of a struct).
    //
    // The returned pointer will be stored with the node in the `user_data` field,
    // accessible within prepare and invoke functions below.
    // NOTE: if the data is already in the desired format, simply implement this
    // function to return `nullptr` and implement the free function to be a no-op.
    void* (*init)(TfLiteContext* context, const char* buffer, size_t length);

    // The pointer `buffer` is the data previously returned by an init invocation.
    void (*free)(TfLiteContext* context, void* buffer);

    // prepare is called when the inputs this node depends on have been resized.
    // context->ResizeTensor() can be called to request output tensors to be
    // resized.
    //
    // Returns kTfLiteOk on success.
    TfLiteStatus(*prepare)(TfLiteContext* context, TfLiteNode* node);

    // Execute the node (should read node->inputs and output to node->outputs).
    // Returns kTfLiteOk on success.
    TfLiteStatus(*invoke)(TfLiteContext* context, TfLiteNode* node);

    // Builtin codes. If this kernel refers to a builtin this is the code
    // of the builtin. This is so we can do marshaling to other frameworks like
    // NN API. Note, it is the responsibility of the registration binder to
    // set this properly.
    int32_t builtin_code;
} TfLiteRegistration;

#endif

typedef enum {
    TFLITE_BUFFER_TYPE_INPUT = 0,
    TFLITE_BUFFER_TYPE_OUTPUT = 1
} TFLiteBufferType;

typedef enum {
    TFLITE_TENSOR_TYPE_NONE = 0,
    TFLITE_TENSOR_TYPE_FLOAT = 1,
    TFLITE_TENSOR_TYPE_UINT8 = 2
} TFLiteTensorType;

#define TFLITE_TENSOR_MAX_DIMENSTIONS    4

typedef struct {
    // The data type specification for data stored in `data`. This affects
    // what member of `data` union should be used.
    TFLiteTensorType type;
    // Tensor shapes
    int dimsSize;
    int dims[TFLITE_TENSOR_MAX_DIMENSTIONS];
    // Data pointer. The appropriate type should be used for a typed
    // tensor based on `type`.
    // The memory pointed by this data pointer is managed by ANeuralNetworksTFLite instance.
    // Caller should not try to free this pointer.
    void* buffer;

    // Correct the error naming from TFLiteTensor, this is actual buffer size in byte.
    size_t bufferSize;
} TFLiteTensorExt;

typedef struct {
    const char* op_name;
    const char* target_name;
    const char* vendor_name;
    void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
    void (*free)(TfLiteContext* context, void* buffer);
    TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
    TfLiteStatus (*add_params)(void*, ANeuralNetworksModel*, std::vector<uint32_t>&, uint32_t&);
} TFLiteCustomOpExt;

typedef enum {
    NP_INFERENCE_TYPE_NONE = 0,
    NP_INFERENCE_TYPE_QNAUT = 1,
    NP_INFERENCE_TYPE_FLOAT = 2,
} NpInferenceType;

/*************************************************************************************************/
typedef int (*ANeuroPilotTFLite_create_fn)
        (ANeuralNetworksTFLite** tflite, const char* modelPath);

typedef int (*ANeuroPilotTFLite_createWithBuffer_fn)
        (ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize);

typedef int (*ANeuroPilotTFLite_createCustom_fn)
        (ANeuralNetworksTFLite** tflite, const char* modelPath,
         const std::vector<TFLiteCustomOpExt>& customOperations);

typedef int (*ANeuroPilotTFLite_createCustomWithBuffer_fn)
        (ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize,
         const std::vector<TFLiteCustomOpExt>& customOperations);

typedef int (*ANeuroPilotTFLite_getTensorByIndex_fn)
        (ANeuralNetworksTFLite* tflite, TFLiteBufferType btype,
         TFLiteTensorExt *tfliteTensor, int tensorIndex);

typedef int (*ANeuroPilotTFLite_invoke_fn)(ANeuralNetworksTFLite* tflite);

typedef int (*ANeuroPilotTFLite_free_fn)(ANeuralNetworksTFLite* tflite);

typedef int (*ANeuroPilotTFLite_bindToDeivce_fn)
        (ANeuralNetworksTFLite* tflite, uint32_t device);

typedef int (*ANeuroPilotTFLiteCustomOp_getIntAttribute_fn)
        (const char* buffer, size_t length, const char* attr, int32_t* outValue);

typedef int (*ANeuroPilotTFLiteCustomOp_getFloatAttribute_fn)
        (const char* buffer, size_t length, const char* attr, float* outValue);

typedef void* (*ANeuroPilotTFLiteCustomOp_getUserData_fn)(TfLiteNode* node);

typedef int (*ANeuroPilotTFLiteCustomOp_getInput_fn)
            (TfLiteContext* context, TfLiteNode* node, int index, TFLiteTensorExt *tfliteTensor);

typedef int (*ANeuroPilotTFLiteCustomOp_getOutput_fn)
            (TfLiteContext* context, TfLiteNode* node, int index, TFLiteTensorExt *tfliteTensor);

typedef int (*ANeuroPilotTFLiteCustomOp_resizeOutput_fn)
            (TfLiteContext* context, TfLiteNode* node, int index, TfLiteIntArray* new_size);

typedef TfLiteIntArray* (*ANeuroPilotTFLite_createIntArray_fn)(int size);

typedef int (*ANeuroPilotTFLite_freeIntArray_fn)(TfLiteIntArray* v);

typedef int (*ANeuroPilot_getInferencePreference_fn)(void);

/*************************************************************************************************/
// For add-on
static void* sTFLiteHandle;
inline void* loadTFLiteLibrary(const char* name) {
    sTFLiteHandle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (sTFLiteHandle == nullptr) {
        TFLITE_LOG("TFLite error: unable to open library %s", name);
    }
    return sTFLiteHandle;
}

inline void* getTFLiteLibraryHandle() {
    if (sTFLiteHandle == nullptr) {
        sTFLiteHandle = loadTFLiteLibrary("libtflite_mtk.so");
    }
    return sTFLiteHandle;
}

inline void* loadTFLiteFunction(const char* name) {
    void* fn = nullptr;
    if (getTFLiteLibraryHandle() != nullptr) {
        fn = dlsym(getTFLiteLibraryHandle(), name);
    }

    if (fn == nullptr) {
        TFLITE_LOG("TFLite error: unable to open function %s", name);
    }

    return fn;
}

/*************************************************************************************************/
/**
 * Create an {@link ANeuralNetworksTFLite} with the TFlite model stored in a file.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLiteWrapper_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLiteWrapper_free} should be called once the instance
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *               Set to NULL if unsuccessful.
 * @param modelPath The full path of the tflite model file.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeTFLite(ANeuralNetworksTFLite** tflite, const char* modelPath) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_create);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, modelPath);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFLite model stored in a data buffer pointer.
 * The data buffer will be duplicated in ANeuralNetworksTFLite instance.
 * Caller could free the input data buffer after calling this API.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLiteWrapper_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLiteWrapper_free} should be called once the instance
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *              Set to NULL if unsuccessful.
 * @param buffer The pointer to the tflite model buffer.
 * @param bufferSize The number of bytes of the tflite model buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeTFLiteWithBuffer(
        ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createWithBuffer);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, buffer, bufferSize);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFlite model stored in a file.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLiteWrapper_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLiteWrapper_free} should be called once the instance
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *               Set to NULL if unsuccessful.
 * @param modelPath The full path of the tflite model file.
 * @param customOperations Custom defined operation list.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeCustomTFLite(ANeuralNetworksTFLite** tflite,
                                       const char* modelPath,
                                       const std::vector<TFLiteCustomOpExt>& customOperations) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createCustom);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, modelPath, customOperations);
}


/**
 * Create an {@link ANeuralNetworksTFLite} with the TFLite model stored in a data buffer pointer.
 * The data buffer will be duplicated in ANeuralNetworksTFLite instance.
 * Caller could free the input data buffer after calling this API.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLiteWrapper_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLiteWrapper_free} should be called once the instance
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *              Set to NULL if unsuccessful.
 * @param buffer The pointer to the tflite model buffer.
 * @param bufferSize The number of bytes of the tflite model buffer.
 * @param customOperations Custom defined operation list.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeCustomTFLiteWithBuffer(ANeuralNetworksTFLite** tflite,
                                       const char* buffer, size_t bufferSize,
                                       const std::vector<TFLiteCustomOpExt>& customOperations) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createCustomWithBuffer);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, buffer, bufferSize, customOperations);
}

/**
 * Get a tensor data structure.
 * This function returns the input or output tensor by the given index.
 *
 * @param tflite The instance to get input/out tensor.
 * @param btype Input or output tensor.
 * @param tfliteTensor A pointer to store the tensor data structure.
 * @param tensorIndex Zero-based index of tensor.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_getTensorByIndex(ANeuralNetworksTFLite* tflite,
                                       TFLiteBufferType btype, TFLiteTensorExt *tfliteTensor,
                                       int tensorIndex) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_getTensorByIndex);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, btype, tfliteTensor, tensorIndex);
}

/**
 * Invoke inference. (run the whole graph in dependency order).
 *
 * @param tflite The instance to invoke inference.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the operation is failed.
 */
inline int ANeuroPilotTFLiteWrapper_invoke(ANeuralNetworksTFLite* tflite) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_invoke);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite);
}

/**
 * Delete a memory object.
 *
 * Destroys the object used by the run time to keep track of the memory.
 * This will free the underlying actual memory if no other code has open
 * handles to this memory.
 *
 * @param memory The memory object to be freed.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_free(ANeuralNetworksTFLite* tflite) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_free);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite);
}

/**
 * Bind a {@link ANeuralNetworksTFLite} instance to the specified device.(CPU/GPU/APU)
 *
 * @param tflite The instance.
 * @param device Device ID.(ANEUROPILOT_CPU/ANEUROPILOT_GPU/ANEUROPILOT_APU)
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
*/
inline int ANeuroPilotTFLiteWrapper_bindToDeivce(
        ANeuralNetworksTFLite* tflite, uint32_t device) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_bindToDeivce);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, device);
}

inline int ANeuroPilotTFLiteWrapper_getCustomOpIntAttribute(const char* buffer, size_t length,
                                        const char* attr, int32_t* outValue) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOp_getIntAttribute);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(buffer, length, attr, outValue);
}

inline int ANeuroPilotTFLiteWrapper_getCustomOpFloatAttribute(const char* buffer, size_t length,
                                        const char* attr, float* outValue) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOp_getFloatAttribute);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(buffer, length, attr, outValue);
}

inline void* ANeuroPilotTFLiteWrapper_getCustomOpUserData(TfLiteNode* node) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOp_getUserData);
    EXECUTE_TFLITE_FUNCTION_RETURN_POINTER(node);
}

inline int ANeuroPilotTFLiteWrapper_getCustomOpInput(TfLiteContext* context, TfLiteNode* node,
                              int index, TFLiteTensorExt *tfliteTensor) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOp_getInput);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(context, node, index, tfliteTensor);
}

inline int ANeuroPilotTFLiteWrapper_getCustomOpOutput(TfLiteContext* context, TfLiteNode* node,
                              int index, TFLiteTensorExt *tfliteTensor) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOp_getOutput);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(context, node, index, tfliteTensor);
}

inline int ANeuroPilotTFLiteWrapper_resizeCustomOpOutput(TfLiteContext* context,
                                       TfLiteNode* node,
                                       int index,
                                       TfLiteIntArray* new_size) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOp_resizeOutput);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(context, node, index, new_size);
}

/**
 * Create a copy of an array passed as `src`.
 * Developers are expected to free memory with ANeuroPilotTFLiteWrapper_freeIntArray.
 *
 * @param size The array size to be created.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
*/
inline TfLiteIntArray* ANeuroPilotTFLiteWrapper_createIntArray(int size) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createIntArray);
    EXECUTE_TFLITE_FUNCTION_RETURN_POINTER(size);
}

/**
 * Free memory of array `v`.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
*/
inline int ANeuroPilotTFLiteWrapper_freeIntArray(TfLiteIntArray* v) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_freeIntArray);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(v);
}

/**
 * Get inference preference of current platform.
 *
 * @return NP_INFERENCE_TYPE_NONE if NeuroPilot is not supported.
 *         NP_INFERENCE_TYPE_QNAUT if quantization inference is preferred.
 *         NP_INFERENCE_TYPE_FLOAT if float inference is preferred.
*/

inline int ANeuroPilotWrapper_getInferencePreference(void) {
    LOAD_TFLITE_FUNCTION(ANeuroPilot_getInferencePreference);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT();
}



/*************************************************************************************************/
// Legacy
typedef struct {
    // The data type specification for data stored in `data`. This affects
    // what member of `data` union should be used.
    TFLiteTensorType type;
    // Tensor shapes
    int dimsSize;
    int dims[TFLITE_TENSOR_MAX_DIMENSTIONS];
    // Data pointer. The appropriate type should be used for a typed
    // tensor based on `type`.
    // The memory pointed by this data pointer is managed by ANeuralNetworksTFLite instance.
    // Caller should not try to free this pointer.
    void* buffer;
    // The number of elements in this tensor.
    // For a float tensor with dimsSize = 4, dims = {1,3,3,1}
    // TThe actual buffer size is 36 bytes (1 x 3 x 3 x 1 x sizeof(float)),
    // but the value of bufferSize is 9.
    size_t bufferSize;
} TFLiteTensor;

typedef TfLiteStatus(*ParameterFunc)(void*, ANeuralNetworksModel*,
        std::vector<uint32_t>&, uint32_t&);

typedef struct {
    const char* name;
    int32_t opCode;
    TfLiteRegistration* opRegistration;
    ParameterFunc parameterFunc;
} TFLiteCustomOp;

typedef struct {
  int size;
  int data[];
} TFLiteIntArray;

typedef int (*ANeuroPilotTFLiteLegacy_createCustom_fn)(
        ANeuralNetworksTFLite** tflite, const char* modelPath,
        const std::vector<TFLiteCustomOp>& customOperations);

typedef int (*ANeuroPilotTFLiteLegacy_createCustomWithBuffer_fn)(
        ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize,
        std::vector<TFLiteCustomOp>& customOperations);

typedef void* (*ANeuroPilotTFLiteCustomOpLegacy_getUserData_fn)(TfLiteNode* node);

typedef int (*ANeuroPilotTFLiteCustomOpLegacy_getOutput_fn)(TfLiteContext* context,
        TfLiteNode* node, int index, TFLiteTensor *tfliteTensor);

typedef int (*ANeuroPilotTFLiteCustomOpLegacy_getInput_fn)(TfLiteContext* context,
        TfLiteNode* node, int index, TFLiteTensor *tfliteTensor);

typedef int (*ANeuroPilotTFLiteCustomOpLegacy_resizeOutput_fn)(TfLiteContext* context,
        TfLiteNode* node, int index, TFLiteIntArray* new_size);

typedef int (*ANeuroPilotTFLiteCustomOpLegacy_getFloatAttribute_fn)(
        const char* buffer, size_t length, const char* attr, float* outValue);

typedef int (*ANeuroPilotTFLiteCustomOpLegacy_getIntAttribute_fn)(
        const char* buffer, size_t length, const char* attr, int32_t* outValue);

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFlite model stored in a file.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuralNetworksTFLite_invoke} is invoked.
 *
 * <p>{@link ANeuralNetworksTFLite_free} should be called once the instance
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *               Set to NULL if unsuccessful.
 * @param modelPath The full path of the tflite model file.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
// Deprecated: Use ANeuroPilotTFLiteWrapper_makeTFLite
inline int ANeuralNetworksTFLite_create(
        ANeuralNetworksTFLite** tflite, const char* modelPath) {
    return ANeuroPilotTFLiteWrapper_makeTFLite(tflite, modelPath);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFlite model stored in a file.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuralNetworksTFLite_invoke} is invoked.
 *
 * <p>{@link ANeuralNetworksTFLite_free} should be called once the instance
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *               Set to NULL if unsuccessful.
 * @param modelPath The full path of the tflite model file.
 * @param customOperations Custom defined operation list.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
// Deprecated: Use ANeuroPilotTFLiteWrapper_makeCustomTFLite
inline int ANeuralNetworksTFLite_createCustom(
        ANeuralNetworksTFLite** tflite, const char* modelPath,
        const std::vector<TFLiteCustomOp>& customOperations) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteLegacy_createCustom);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, modelPath, customOperations);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFLite model stored in a data buffer pointer.
 * The data buffer will be duplicated in ANeuralNetworksTFLite instance.
 * Caller could free the input data buffer after calling this API.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuralNetworksTFLite_invoke} is invoked.
 *
 * <p>{@link ANeuralNetworksTFLite_free} should be called once the instance
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *              Set to NULL if unsuccessful.
 * @param buffer The pointer to the tflite model buffer.
 * @param bufferSize The number of bytes of the tflite model buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
// Deprecated: Use ANeuroPilotTFLiteWrapper_makeTFLiteWithBuffer
inline int ANeuralNetworksTFLite_createWithBuffer(
        ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize) {
    return ANeuroPilotTFLiteWrapper_makeTFLiteWithBuffer(tflite, buffer, bufferSize);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFLite model stored in a data buffer pointer.
 * The data buffer will be duplicated in ANeuralNetworksTFLite instance.
 * Caller could free the input data buffer after calling this API.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuralNetworksTFLite_invoke} is invoked.
 *
 * <p>{@link ANeuralNetworksTFLite_free} should be called once the instance
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *              Set to NULL if unsuccessful.
 * @param buffer The pointer to the tflite model buffer.
 * @param bufferSize The number of bytes of the tflite model buffer.
 * @param customOperations Custom defined operation list.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
// Deprecated: Use ANeuroPilotTFLiteWrapper_makeTFLiteWithBuffer
inline int ANeuralNetworksTFLite_createCustomWithBuffer(ANeuralNetworksTFLite** tflite,
        const char* buffer, size_t bufferSize, std::vector<TFLiteCustomOp>& customOperations) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteLegacy_createCustomWithBuffer);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, buffer, bufferSize, customOperations);
}

/**
 * Get a tensor data structure. This function returns the first input or output tensor.
 *
 * @param tflite The instance to get input/out tensor.
 * @param btype Input or output tensor.
 * @param tfliteTensor A pointer to store the tensor data structure.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
// Deprecated: Use ANeuroPilotTFLiteWrapper_getTensorByIndex
inline int ANeuralNetworksTFLite_getTensor(ANeuralNetworksTFLite* tflite,
        TFLiteBufferType btype, TFLiteTensor *tfliteTensor) {
    int ret = ANeuroPilotTFLiteWrapper_getTensorByIndex(tflite, btype,
            reinterpret_cast<TFLiteTensorExt *>(tfliteTensor), 0);
    if (tfliteTensor != nullptr) {
        if (tfliteTensor->type == TFLITE_TENSOR_TYPE_FLOAT) {
            tfliteTensor->bufferSize = tfliteTensor->bufferSize / sizeof(float);
        } else if (tfliteTensor->type == TFLITE_TENSOR_TYPE_UINT8) {
            tfliteTensor->bufferSize = tfliteTensor->bufferSize / sizeof(uint8_t);
        }
    }
    return ret;
}

/**
 * Get a tensor data structure.
 * This function returns the input or output tensor by the given index.
 *
 * @param tflite The instance to get input/out tensor.
 * @param btype Input or output tensor.
 * @param tfliteTensor A pointer to store the tensor data structure.
 * @param tensorIndex Zero-based index of tensor.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
// Deprecated: Use ANeuroPilotTFLiteWrapper_getTensorByIndex
inline int ANeuralNetworksTFLite_getTensorByIndex(ANeuralNetworksTFLite* tflite,
        TFLiteBufferType btype, TFLiteTensor *tfliteTensor, int tensorIndex) {
    int ret = ANeuroPilotTFLiteWrapper_getTensorByIndex(tflite, btype,
            reinterpret_cast<TFLiteTensorExt *>(tfliteTensor), tensorIndex);
    if (tfliteTensor != nullptr) {
        if (tfliteTensor->type == TFLITE_TENSOR_TYPE_FLOAT) {
            tfliteTensor->bufferSize = tfliteTensor->bufferSize / sizeof(float);
        } else if (tfliteTensor->type == TFLITE_TENSOR_TYPE_UINT8) {
            tfliteTensor->bufferSize = tfliteTensor->bufferSize / sizeof(uint8_t);
        }
    }
    return ret;
}

/**
 * Invoke inference. (run the whole graph in dependency order).
 *
 * @param tflite The instance to invoke inference.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the operation is failed.
 */
// Deprecated: Use ANeuroPilotTFLiteWrapper_invoke
inline int ANeuralNetworksTFLite_invoke(ANeuralNetworksTFLite* tflite) {
    return ANeuroPilotTFLiteWrapper_invoke(tflite);
}

/**
 * Delete a memory object.
 *
 * Destroys the object used by the run time to keep track of the memory.
 * This will free the underlying actual memory if no other code has open
 * handles to this memory.
 *
 * @param memory The memory object to be freed.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
// Deprecated: Use ANeuroPilotTFLiteWrapper_free
inline int ANeuralNetworksTFLite_free(ANeuralNetworksTFLite* tflite) {
    return ANeuroPilotTFLiteWrapper_free(tflite);
}

/**
 * Bind a {@link ANeuralNetworksTFLite} instance to the specified device.(CPU/GPU/APU)
 *
 * @param tflite The instance.
 * @param device Device ID.(ANEURALNETWORKS_CPU/ANEURALNETWORKS_GPU/ANEURALNETWORKS_APU)
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
*/
// Deprecated: Use ANeuroPilotTFLiteWrapper_bindToDeivce
inline int ANeuralNetworksTFLite_bindToDeivce(
        ANeuralNetworksTFLite* tflite, uint32_t device) {
    return ANeuroPilotTFLiteWrapper_bindToDeivce(tflite, device);
}

/**
 * Create a copy of an array passed as `src`.
 * Developers are expected to free memory with ANeuralNetworksTFLite_freeIntArray.
 *
 * @param size The array size to be created.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
*/
// Deprecated: Use ANeuroPilotTFLiteWrapper_createIntArray
inline TFLiteIntArray* ANeuralNetworksTFLite_createIntArray(int size) {
    return reinterpret_cast<TFLiteIntArray*>(ANeuroPilotTFLiteWrapper_createIntArray(size));
}

/**
 * Free memory of array `v`.
 *
*/
// Deprecated: Use ANeuroPilotTFLiteWrapper_freeIntArray
inline void ANeuralNetworksTFLite_freeIntArray(TFLiteIntArray* v) {
    ANeuroPilotTFLiteWrapper_freeIntArray(reinterpret_cast<TfLiteIntArray*>(v));
}

inline void* ANeuralNetworksTFLiteCustomOp_getUserData(TfLiteNode* node) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOpLegacy_getUserData);
    EXECUTE_TFLITE_FUNCTION_RETURN_POINTER(node);
}

inline int ANeuralNetworksTFLiteCustomOp_getInput(TfLiteContext* context, TfLiteNode* node,
        int index, TFLiteTensor *tfliteTensor) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOpLegacy_getInput);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(context, node, index, tfliteTensor);
}

inline int ANeuralNetworksTFLiteCustomOp_getOutput(TfLiteContext* context, TfLiteNode* node,
        int index, TFLiteTensor *tfliteTensor) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOpLegacy_getOutput);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(context, node, index, tfliteTensor);
}

inline int ANeuralNetworksTFLiteCustomOp_resizeOutput(TfLiteContext* context,
        TfLiteNode* node, int index, TFLiteIntArray* new_size) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOpLegacy_resizeOutput);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(context, node, index, new_size);
}

inline int ANeuralNetworksTFLiteCustomOp_getFloatAttribute(const char* buffer, size_t length,
        const char* attr, float* outValue) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOpLegacy_getFloatAttribute);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(buffer, length, attr, outValue);
}

inline int ANeuralNetworksTFLiteCustomOp_getIntAttribute(const char* buffer, size_t length,
        const char* attr, int32_t* outValue) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOpLegacy_getIntAttribute);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(buffer, length, attr, outValue);
}

#endif  //  __ANDROID_API__ >= 27
#endif  // ANDROID_ML_NN_RUNTIME_NEURO_PILOT_TFLITE_SHIM_H
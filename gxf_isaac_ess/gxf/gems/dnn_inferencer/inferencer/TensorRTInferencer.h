#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <NvInferRuntimeCommon.h>
#include <NvInfer.h>
#include "Errors.h"
#include "IInferenceBackend.h"
#include "Inferencer.h"

namespace cvcore {
namespace inferencer {

// Alias for tensor base class used for input/output buffers
using TensorBase = cvcore::tensor_ops::TensorBase;

// TensorRTInferencer provides a high-performance backend for running deep learning inference using TensorRT
class TensorRTInferencer : public IInferenceBackendClient {
 public:
    // Constructor: initializes the TensorRT engine and context from parameters (ONNX or engine file)
    TensorRTInferencer(const TensorRTInferenceParams& params);

    // Set the input tensor buffer for a specific input layer
    std::error_code setInput(const TensorBase& trtInputBuffer,
        std::string inputLayerName) override;

    // Set the output tensor buffer for a specific output layer
    std::error_code setOutput(TensorBase& trtOutputBuffer,
        std::string outputLayerName) override;

    // Get parsed model metadata (input/output layer info, shapes, types, etc.)
    ModelMetaData getModelMetaData() const override;

    // Convert an ONNX model to a TensorRT engine plan (optionally using DLA, FP16, INT8, etc.)
    std::error_code convertModelToEngine(int32_t dla_core,
        const char* model_file, int64_t max_workspace_size, int32_t buildFlags,
        std::size_t max_batch_size);

    // Run inference for the given batch size (enqueue on CUDA stream)
    std::error_code infer(size_t batchSize = 1) override;

    // Set the CUDA stream to be used for inference
    std::error_code setCudaStream(cudaStream_t) override;

    // Unregister shared memory for a specific layer (no-op in this implementation)
    std::error_code unregister(std::string layerName) override;

    // Unregister all shared memory (no-op in this implementation)
    std::error_code unregister() override;

 private:
    // Destructor: cleans up resources
    ~TensorRTInferencer();

    // Logger for TensorRT messages and errors
    std::unique_ptr<TRTLogger> m_logger;

    // TensorRT runtime object for engine deserialization
    std::unique_ptr<nvinfer1::IRuntime> m_inferenceRuntime;

    // Maximum batch size supported by the engine
    size_t m_maxBatchSize;

    // Names of input layers
    std::vector<std::string> m_inputLayers;

    // Names of output layers
    std::vector<std::string> m_outputLayers;

    // CUDA stream used for inference execution
    cudaStream_t m_cudaStream;

    // Pointer to the TensorRT engine (may be owned or provided externally)
    nvinfer1::ICudaEngine* m_inferenceEngine;

    // Owned TensorRT engine (if created internally)
    std::unique_ptr<nvinfer1::ICudaEngine> m_ownedInferenceEngine;

    // Execution context for running inference
    std::unique_ptr<nvinfer1::IExecutionContext> m_inferenceContext;

    // Number of input/output tensors
    size_t m_ioTensorsCount;

    // Parsed model metadata (input/output layer info)
    ModelMetaData m_modelInfo;

    // Whether the engine uses implicit batch mode
    bool m_hasImplicitBatch;

    // Buffer holding the serialized engine plan
    std::vector<char> m_modelEngineStream;

    // Size of the serialized engine plan
    size_t m_modelEngineStreamSize = 0;

    // Parse the TensorRT engine to extract model metadata
    std::error_code ParseTRTModel();

    // Helper to extract layer info (shape, type, layout) from the engine
    std::error_code getLayerInfo(LayerInfo& layer, std::string layerName);
};

}  // namespace inferencer
}  // namespace cvcore

/*
This header file declares the TensorRTInferencer class, which serves as a high-performance inference backend 
for running deep learning models using NVIDIA TensorRT. The class provides methods for loading and managing 
TensorRT engines (from ONNX models or pre-built engine files), setting up input and output tensor buffers, 
configuring CUDA streams, and executing inference. It also includes functionality for extracting model metadata, 
such as input and output layer information, and supports advanced features like dynamic batch sizes and hardware 
acceleration options (FP16, INT8, DLA). This interface enables efficient integration of neural network inference 
into larger computer vision or robotics pipelines.
*/
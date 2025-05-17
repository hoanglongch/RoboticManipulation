#include "gems/dnn_inferencer/inferencer/TensorRTInferencer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "gems/dnn_inferencer/inferencer/Errors.h"
#include "gems/dnn_inferencer/inferencer/IInferenceBackend.h"
#include "gems/dnn_inferencer/inferencer/Inferencer.h"
#include "gems/dnn_inferencer/inferencer/TensorRTUtils.h"
#include "gxf/core/expected.hpp"
#include <NvOnnxParser.h>

namespace cvcore {
namespace inferencer {

namespace {
// Helper function to compute the total data size for a tensor given its shape and data type
using TensorBase = cvcore::tensor_ops::TensorBase;
using ChannelType = cvcore::tensor_ops::ChannelType;
using ChannelCount =  cvcore::tensor_ops::ChannelCount;
using ErrorCode = cvcore::tensor_ops::ErrorCode;
size_t getDataSize(const std::vector<int64_t>& shape, ChannelType dataType) {
    size_t layerShape = 1;
    for (size_t k = 0; k < shape.size(); k++)
        layerShape *= shape[k] <= 0 ? 1 : shape[k];

    return layerShape * GetChannelSize(dataType);
}
}  // namespace

// Retrieve layer information (shape, layout, data type, etc.) from the TensorRT engine
std::error_code TensorRTInferencer::getLayerInfo(LayerInfo& layer, std::string layerName) {
    layer.name = layerName;
    // Get the shape of the tensor from the engine
    auto dim = m_inferenceEngine->getTensorShape(layer.name.c_str());
    // Get the tensor format (e.g., NCHW, NHWC)
    nvinfer1::TensorFormat tensorFormat = m_inferenceEngine->getTensorFormat(layer.name.c_str());

    std::error_code err;
    // Convert TensorRT tensor format to cvcore layout
    err = getCVCoreChannelLayoutFromTensorRT(layer.layout, tensorFormat);
    if (err != make_error_code(ErrorCode::SUCCESS)) {
        return ErrorCode::INVALID_ARGUMENT;
    }

    // Store the shape dimensions
    for (int32_t cnt = 0; cnt < dim.nbDims; cnt++) {
        layer.shape.push_back(dim.d[cnt]);
    }

    // Convert TensorRT data type to cvcore data type
    err = getCVCoreChannelTypeFromTensorRT(layer.dataType,
        m_inferenceEngine->getTensorDataType(layer.name.c_str()));
    // Compute the total size of the layer in bytes
    layer.layerSize = getDataSize(layer.shape, layer.dataType);
    if (err != make_error_code(ErrorCode::SUCCESS)) {
        return ErrorCode::INVALID_ARGUMENT;
    }

    return ErrorCode::SUCCESS;
}

// Parse the TensorRT model and populate model metadata (input/output layers, etc.)
std::error_code TensorRTInferencer::ParseTRTModel() {
    m_modelInfo.modelName    = m_inferenceEngine->getName();
    m_modelInfo.modelVersion = "";
    m_modelInfo.maxBatchSize = m_maxBatchSize;
    std::error_code err;
    // Parse input layers
    for (size_t i = 0; i < m_inputLayers.size(); i++) {
        LayerInfo layer;
        err = getLayerInfo(layer, m_inputLayers[i]);
        if (err != make_error_code(ErrorCode::SUCCESS)) {
            return err;
        }
        m_modelInfo.inputLayers[layer.name] = layer;
    }
    // Parse output layers
    for (size_t i = 0; i < m_outputLayers.size(); i++) {
        LayerInfo layer;
        err = getLayerInfo(layer, m_outputLayers[i]);
        if (err != make_error_code(ErrorCode::SUCCESS)) {
            return err;
        }
        m_modelInfo.outputLayers[layer.name] = layer;
    }

    return ErrorCode::SUCCESS;
}

// Convert an ONNX model to a TensorRT engine plan, with support for DLA, FP16, INT8, etc.
std::error_code TensorRTInferencer::convertModelToEngine(int32_t dla_core,
  const char* model_file, int64_t max_workspace_size, int32_t buildFlags,
  std::size_t max_batch_size) {
  GXF_LOG_INFO("Convert to engine from onnx file: %s", model_file);
  // Create the TensorRT engine builder
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(*(m_logger.get())));

  // Builder config for memory and optimization options
  std::unique_ptr<nvinfer1::IBuilderConfig> builderConfig(builder->createBuilderConfig());
  builderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size);

  // Set DLA core if requested
  if (dla_core >= 0) {
    builderConfig->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    builderConfig->setDLACore(dla_core);
  }
  // Enable INT8 or FP16 precision if requested
  if (buildFlags & kINT8) {
    builderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
    builderConfig->setInt8Calibrator(nullptr);
  }
  if (buildFlags & OnnxModelBuildFlag::kFP16) {
    builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  if (buildFlags & kGPU_FALLBACK) {
    builderConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
  }

  // Parse ONNX model with explicit batch support
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

  std::unique_ptr<nvonnxparser::IParser> onnx_parser(
      nvonnxparser::createParser(*network, *(m_logger.get())));
  // Parse the ONNX file into the TensorRT network
  if (!onnx_parser->parseFromFile(model_file,
      static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    GXF_LOG_ERROR("Failed to parse ONNX file %s", model_file);
    return ErrorCode::INVALID_ARGUMENT;
  }

  // Create optimization profile for dynamic input shapes
  nvinfer1::IOptimizationProfile* optimization_profile = builder->createOptimizationProfile();
  const int number_inputs = network->getNbInputs();
  for (int i = 0; i < number_inputs; ++i) {
    auto* bind_tensor = network->getInput(i);
    const char* bind_name = bind_tensor->getName();
    nvinfer1::Dims dims = bind_tensor->getDimensions();

    // Validate input tensor dimensions
    if (dims.nbDims <= 0) {
      GXF_LOG_ERROR("Invalid input tensor dimensions for binding %s", bind_name);
      return ErrorCode::INVALID_ARGUMENT;
    }
    for (int j = 1; j < dims.nbDims; ++j) {
      if (dims.d[j] <= 0) {
        GXF_LOG_ERROR(
            "Input binding %s requires dynamic size on dimension No.%d which is not supported",
            bind_tensor->getName(), j);
        return ErrorCode::INVALID_ARGUMENT;
      }
    }
    if (dims.d[0] == -1) {
      // Only support dynamic batch size as first dimension
      dims.d[0] = 1;
      optimization_profile->setDimensions(bind_name, nvinfer1::OptProfileSelector::kMIN, dims);
      optimization_profile->setDimensions(bind_name, nvinfer1::OptProfileSelector::kOPT, dims);
      dims.d[0] = max_batch_size;
      if (max_batch_size <= 0) {
        dims.d[0] = 1;
      }
      optimization_profile->setDimensions(bind_name, nvinfer1::OptProfileSelector::kMAX, dims);
    }
  }
  builderConfig->addOptimizationProfile(optimization_profile);

  // Build the serialized TensorRT engine
  std::unique_ptr<nvinfer1::IHostMemory> model_stream(
      builder->buildSerializedNetwork(*network, *builderConfig));
  if (!model_stream) {
    GXF_LOG_ERROR("Failed to build TensorRT engine from model %s.", model_file);
    return InferencerErrorCode::INVALID_ARGUMENT;
  }

  if (model_stream->size() == 0 || model_stream->data() == nullptr) {
    GXF_LOG_ERROR("Fail to serialize TensorRT Engine.");
    return InferencerErrorCode::INVALID_ARGUMENT;
  }

  // Copy engine plan to internal buffer
  const char* data = static_cast<const char*>(model_stream->data());
  m_modelEngineStream.resize(model_stream->size());
  std::copy(data, data + model_stream->size(), m_modelEngineStream.data());
  m_modelEngineStreamSize =  model_stream->size();
  return InferencerErrorCode::SUCCESS;
}

// Serialize the engine plan to disk at the specified path
std::error_code SerializeEnginePlan(const std::vector<char>& plan, const std::string path) {
  std::ofstream out_stream(path.c_str(), std::ofstream::binary);
  if (!out_stream.is_open()) {
    GXF_LOG_ERROR("Failed to open engine file %s.", path.c_str());
    return InferencerErrorCode::TENSORRT_ENGINE_ERROR;
  }
  out_stream.write(plan.data(), plan.size());
  if (out_stream.bad()) {
    GXF_LOG_ERROR("Failed writing engine file %s.", path.c_str());
    return InferencerErrorCode::TENSORRT_ENGINE_ERROR;
  }
  out_stream.close();
  GXF_LOG_INFO("TensorRT engine serialized at %s", path.c_str());
  return InferencerErrorCode::SUCCESS;
}

// Constructor: loads or builds a TensorRT engine from ONNX or engine file, sets up context
TensorRTInferencer::TensorRTInferencer(const TensorRTInferenceParams& params)
    : m_logger(new TRTLogger())
    , m_maxBatchSize(params.maxBatchSize)
    , m_inputLayers(params.inputLayerNames)
    , m_outputLayers(params.outputLayerNames)
    , m_cudaStream(0)
    , m_inferenceEngine(nullptr) {
    if (params.inferType == TRTInferenceType::TRT_ENGINE) {
        // Try to open the engine file
        std::ifstream trtModelFStream(params.engineFilePath, std::ios::binary);
        const bool shouldRebuild = params.force_engine_update || !trtModelFStream.good();
        const bool canRebuild = params.onnxFilePath.size() != 0;
        if (canRebuild && shouldRebuild) {
            // Delete engine plan file if forced update
            std::remove(params.engineFilePath.c_str());
            if (std::ifstream(params.engineFilePath).good()) {
                GXF_LOG_ERROR("Failed to remove engine plan file %s for forced engine update.",
                    params.engineFilePath.c_str());
            }
            GXF_LOG_INFO(
                "Rebuilding CUDA engine %s%s. "
                "Note: this process may take up to several minutes.",
                params.force_engine_update ? " (forced by config)" : "",
                params.engineFilePath.c_str());
            // Build engine from ONNX
            auto result = convertModelToEngine(params.dlaID, params.onnxFilePath.c_str(),
                params.max_workspace_size, params.buildFlags, params.maxBatchSize);
            if (result != InferencerErrorCode::SUCCESS) {
                GXF_LOG_INFO("Failed to create engine plan for model %s.",
                    params.onnxFilePath.c_str());
            }

            // Try to serialize the plan to disk
            if (SerializeEnginePlan(m_modelEngineStream, params.engineFilePath) !=
                InferencerErrorCode::SUCCESS) {
                GXF_LOG_INFO(
                    "Engine plan serialization failed. Proceeds with in-memory"
                    "engine plan anyway.");
            }
        } else {
            // Load engine from file
            GXF_LOG_INFO("Using CUDA engine %s. ", params.engineFilePath.c_str());

            trtModelFStream.seekg(0, trtModelFStream.end);
            m_modelEngineStreamSize = trtModelFStream.tellg();
            m_modelEngineStream.resize(m_modelEngineStreamSize);
            trtModelFStream.seekg(0, trtModelFStream.beg);
            trtModelFStream.read(m_modelEngineStream.data(), m_modelEngineStreamSize);
            trtModelFStream.close();
        }

        // Create TensorRT runtime and engine from serialized plan
        m_inferenceRuntime.reset(nvinfer1::createInferRuntime(*(m_logger.get())));
        if (params.dlaID != -1 && params.dlaID < m_inferenceRuntime->getNbDLACores()) {
            m_inferenceRuntime->setDLACore(params.dlaID);
        }
        m_inferenceEngine = m_inferenceRuntime->deserializeCudaEngine(m_modelEngineStream.data(),
            m_modelEngineStreamSize);
        m_ownedInferenceEngine.reset(m_inferenceEngine);
        m_inferenceContext.reset(m_inferenceEngine->createExecutionContext());
        m_inferenceContext->setOptimizationProfileAsync(0, m_cudaStream);
    } else {
        // Use provided engine pointer (for native TRT)
        if (params.engine == nullptr) {
            throw ErrorCode::INVALID_ARGUMENT;
        }
        m_inferenceEngine = params.engine;
        m_inferenceContext.reset(m_inferenceEngine->createExecutionContext());
    }

    if (m_inferenceEngine == nullptr || m_inferenceContext == nullptr) {
        throw ErrorCode::INVALID_ARGUMENT;
    }

    m_hasImplicitBatch = m_inferenceEngine->hasImplicitBatchDimension();
    m_ioTensorsCount    = m_inferenceEngine->getNbIOTensors();
    // Set input shapes for explicit batch mode
    if (!m_hasImplicitBatch) {
        for (size_t i = 0; i < m_ioTensorsCount; i++) {
            const char* name = m_inferenceEngine->getIOTensorName(i);
            if (m_inferenceEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                nvinfer1::Dims dims_i(m_inferenceEngine->getTensorShape(name));
                nvinfer1::Dims4 inputDims{1, dims_i.d[1], dims_i.d[2], dims_i.d[3]};
                m_inferenceContext->setInputShape(name, inputDims);
            }
        }
    }
    std::error_code err;
    err = ParseTRTModel();
    if (err != make_error_code(ErrorCode::SUCCESS)) {
        throw err;
    }
}

// Set the input tensor buffer for a given input layer
std::error_code TensorRTInferencer::setInput(const TensorBase& trtInputBuffer,
    std::string inputLayerName) {
    if (m_modelInfo.inputLayers.find(inputLayerName) == m_modelInfo.inputLayers.end()) {
        return ErrorCode::INVALID_ARGUMENT;
    }
    LayerInfo layer        = m_modelInfo.inputLayers[inputLayerName];
    // Set the address of the input buffer for the given layer
    m_inferenceContext->setTensorAddress(inputLayerName.c_str(),
                                          trtInputBuffer.getData());
    return ErrorCode::SUCCESS;
}

// Set the output tensor buffer for a given output layer
std::error_code TensorRTInferencer::setOutput(TensorBase& trtOutputBuffer,
    std::string outputLayerName) {
    if (m_modelInfo.outputLayers.find(outputLayerName) == m_modelInfo.outputLayers.end()) {
        return ErrorCode::INVALID_ARGUMENT;
    }
    LayerInfo layer        = m_modelInfo.outputLayers[outputLayerName];
    // Set the address of the output buffer for the given layer
    m_inferenceContext->setTensorAddress(outputLayerName.c_str(),
                                         trtOutputBuffer.getData());
    return ErrorCode::SUCCESS;
}

// Get the parsed model metadata (input/output layer info, etc.)
ModelMetaData TensorRTInferencer::getModelMetaData() const {
    return m_modelInfo;
}

// Run inference for the given batch size
std::error_code TensorRTInferencer::infer(size_t batchSize) {
    bool err = true;
    if (!m_hasImplicitBatch) {
        size_t ioTensorsCount = m_inferenceEngine->getNbIOTensors();
        for (size_t i = 0; i < ioTensorsCount; i++) {
            const char* name = m_inferenceEngine->getIOTensorName(i);
            if (m_inferenceEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                nvinfer1::Dims dims_i(m_inferenceEngine->getTensorShape(name));
                nvinfer1::Dims4 inputDims{1, dims_i.d[1], dims_i.d[2], dims_i.d[3]};
                m_inferenceContext->setInputShape(name, inputDims);
            }
        }
        // Enqueue the inference execution on the CUDA stream
        err = m_inferenceContext->enqueueV3(m_cudaStream);
    } else {
        return InferencerErrorCode::INVALID_ARGUMENT;
    }
    if (!err) {
        return InferencerErrorCode::TENSORRT_INFERENCE_ERROR;
    }
    return ErrorCode::SUCCESS;
}

// Set the CUDA stream to be used for inference
std::error_code TensorRTInferencer::setCudaStream(cudaStream_t cudaStream) {
    m_cudaStream = cudaStream;
    return ErrorCode::SUCCESS;
}

// Unregister a layer (no-op in this implementation)
std::error_code TensorRTInferencer::unregister(std::string layerName) {
    return ErrorCode::SUCCESS;
}

// Unregister all layers (no-op in this implementation)
std::error_code TensorRTInferencer::unregister() {
    return ErrorCode::SUCCESS;
}

// Destructor
TensorRTInferencer::~TensorRTInferencer() {
}

}  // namespace inferencer
}  // namespace cvcore

/*
This file implements the TensorRTInferencer class, which provides a high-performance inference backend for 
running deep learning models using NVIDIA TensorRT. It handles loading and parsing ONNX models, building or 
loading optimized TensorRT engine plans (with support for FP16, INT8, and DLA acceleration), managing input 
and output tensor buffers, and executing inference on CUDA streams. The class also extracts and manages model 
metadata, such as input and output layer information, and supports dynamic batch sizes. This infrastructure 
enables efficient, real-time neural network inference for tasks like semantic segmentation or stereo depth 
estimation on NVIDIA hardware platforms.
*/
// Core logic for the ESS (Efficient Stereo Supervision) inference, including CUDA-based execution and tensor management.
#include "extensions/ess/inference/ESS.h"

#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "extensions/tensorops/core/CVError.h"
#include "extensions/tensorops/core/Image.h"
#include "extensions/tensorops/core/ImageUtils.h"
#include "extensions/tensorops/core/Memory.h"
#include "gems/dnn_inferencer/inferencer/IInferenceBackend.h"
#include "gems/dnn_inferencer/inferencer/Inferencer.h"

namespace nvidia {
namespace isaac {
namespace ess {

// Type aliases for convenience
using ImagePreProcessingParams = cvcore::tensor_ops::ImagePreProcessingParams;
using TensorRTInferenceParams = cvcore::inferencer::TensorRTInferenceParams;
using InferenceBackendClient = cvcore::inferencer::InferenceBackendClient;
using InferenceBackendFactory = cvcore::inferencer::InferenceBackendFactory;
using TRTInferenceType = cvcore::inferencer::TRTInferenceType;
using ErrorCode = cvcore::tensor_ops::ErrorCode;

// Default parameters for preprocessing and model input
const ImagePreProcessingParams defaultPreProcessorParams = {
    ImageType::BGR_U8,   // Input image type
    {-128, -128, -128},  // Mean value for normalization
    {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0}, // Normalization factor
    {0.5, 0.5, 0.5}      // Standard deviation
};

const ModelInputParams defaultModelInputParams = {
    1,                   // Max batch size supported
    ImageType::RGB_U8    // Input image type the network is trained with
};

const ModelInferenceParams defaultInferenceParams = {
    "models/ess.engine",           // Engine file path
    {"input_left", "input_right"}, // Input layer names for the model
    {"output_left", "output_conf"} // Output layer names for the model
};

const ESSPreProcessorParams defaultESSPreProcessorParams = {
    PreProcessType::RESIZE // Preprocessing type: resize input image to network input dimensions
};

const int InferencerVersion = NV_TENSORRT_VERSION;

// Implementation struct for the ESS class
struct ESS::ESSImpl {
    TensorRTInferenceParams tensorrtParams; // TensorRT inference parameters
    InferenceBackendClient client;          // Backend client for inference

    // Model tensors for input and output
    Tensor_NHWC_C1_F32 m_outputDevice;      // Output tensor (disparity map)
    Tensor_NHWC_C1_F32 m_confMap;           // Output tensor (confidence map)
    Tensor_NCHW_C3_F32 m_inputLeftPlanar;   // Preprocessed left input (planar format)
    Tensor_NCHW_C3_F32 m_inputRightPlanar;  // Preprocessed right input (planar format)

    // Preprocessing object
    std::unique_ptr<ESSPreProcessor> m_preprocess;

    // Maximum batch size supported by the model
    size_t m_maxBatchSize;

    // Layer names for model input/output
    std::string m_leftInputLayerName, m_rightInputLayerName;
    std::string m_outputName;
    std::string m_confMapName;

    // Network input/output dimensions
    size_t m_networkInputWidth, m_networkInputHeight;
    size_t m_networkOutputWidth, m_networkOutputHeight;

    // Constructor: initializes all parameters, loads the TensorRT engine, and allocates tensors
    ESSImpl(const ImagePreProcessingParams & imgParams,
        const ModelInputParams & modelInputParams,
        const ModelBuildParams & modelBuildParams,
        const ModelInferenceParams & modelInferParams,
        const ESSPreProcessorParams & essParams)
        : m_maxBatchSize(modelInputParams.maxBatchSize) {
        // Validate model input/output configuration
        if (modelInferParams.inputLayers.size() != 2 || modelInferParams.outputLayers.size() != 2 ||
            modelInputParams.maxBatchSize <= 0) {
            throw ErrorCode::INVALID_ARGUMENT;
        }

        // Set build flags for TensorRT engine (e.g., FP16 support)
        cvcore::inferencer::OnnxModelBuildFlag buildFlags =
            cvcore::inferencer::OnnxModelBuildFlag::NONE;
        if (modelBuildParams.enable_fp16) {
            buildFlags = cvcore::inferencer::OnnxModelBuildFlag::kFP16;
        }

        // Fill in TensorRT inference parameters
        tensorrtParams = {
            TRTInferenceType::TRT_ENGINE,           // Use TensorRT engine
            nullptr,                                // No ONNX model pointer
            modelBuildParams.onnx_file_path,        // Path to ONNX file (if needed)
            modelInferParams.engineFilePath,        // Path to TensorRT engine file
            modelBuildParams.force_engine_update,   // Force engine rebuild if true
            buildFlags,                            // Build flags (e.g., FP16)
            modelBuildParams.max_workspace_size,    // Workspace size for TensorRT
            modelInputParams.maxBatchSize,          // Max batch size
            modelInferParams.inputLayers,           // Input layer names
            modelInferParams.outputLayers,          // Output layer names
            modelBuildParams.dla_core               // DLA core (if used)
        };

        // Create the TensorRT inference backend client
        std::error_code err =
            InferenceBackendFactory::CreateTensorRTInferenceBackendClient(client, tensorrtParams);

        if (err != cvcore::tensor_ops::make_error_code(ErrorCode::SUCCESS)) {
            throw err;
        }

        // Get model metadata (input/output shapes, batch size, etc.)
        cvcore::inferencer::ModelMetaData modelInfo = client->getModelMetaData();
        if (modelInfo.maxBatchSize != modelInputParams.maxBatchSize) {
            throw ErrorCode::INVALID_ARGUMENT;
        }

        // Store input/output layer names
        m_leftInputLayerName  = modelInferParams.inputLayers[0];
        m_rightInputLayerName = modelInferParams.inputLayers[1];
        m_outputName = modelInferParams.outputLayers[0];
        m_confMapName = modelInferParams.outputLayers[1];

        // Get input dimensions from model metadata
        m_networkInputHeight  = modelInfo.inputLayers[m_leftInputLayerName].shape[2];
        m_networkInputWidth   = modelInfo.inputLayers[m_leftInputLayerName].shape[3];

        // Allocate input tensors for left and right images
        m_inputLeftPlanar     = {m_networkInputWidth, m_networkInputHeight,
            modelInputParams.maxBatchSize, false};
        m_inputRightPlanar    = {m_networkInputWidth, m_networkInputHeight,
            modelInputParams.maxBatchSize, false};

        // Get output dimensions from model metadata
        m_networkOutputHeight = modelInfo.outputLayers[modelInferParams.outputLayers[0]].shape[1];
        m_networkOutputWidth  = modelInfo.outputLayers[modelInferParams.outputLayers[0]].shape[2];

        // Allocate output tensors for disparity and confidence maps
        m_outputDevice = {m_networkOutputWidth, m_networkOutputHeight,
            modelInputParams.maxBatchSize, false};
        m_confMap = {m_networkOutputWidth, m_networkOutputHeight,
            modelInputParams.maxBatchSize, false};

        // Register input tensors with the inference backend
        CHECK_ERROR_CODE(client->setInput(m_inputLeftPlanar, modelInferParams.inputLayers[0]));
        CHECK_ERROR_CODE(client->setInput(m_inputRightPlanar, modelInferParams.inputLayers[1]));

        // Initialize the preprocessor for input images
        m_preprocess.reset(new ESSPreProcessor(imgParams, modelInputParams,
            m_networkInputWidth, m_networkInputHeight, essParams));
    }

    // Destructor: unregisters the backend client and releases resources
    ~ESSImpl() {
        CHECK_ERROR_CODE(client->unregister());
        InferenceBackendFactory::DestroyTensorRTInferenceBackendClient(client);
    }

    // Getters for output dimensions
    size_t getModelOutputHeight() const {
        return m_networkOutputHeight;
    }
    size_t getModelOutputWidth() const {
        return m_networkOutputWidth;
    }

    // Execute inference using NHWC uint8 input (raw images)
    void execute(Tensor_NHWC_C1_F32 & output, Tensor_NHWC_C1_F32 & confMap,
                 const Tensor_NHWC_C3_U8 & inputLeft, const Tensor_NHWC_C3_U8 & inputRight,
                 cudaStream_t stream) {
        size_t batchSize = inputLeft.getDepth();
        // Validate input types and batch sizes
        if (inputLeft.isCPU() || inputRight.isCPU()) {
            throw std::invalid_argument("ESS : Input type should be GPU buffer");
        }
        if (inputLeft.getDepth() > m_maxBatchSize || inputRight.getDepth() > m_maxBatchSize) {
            throw std::invalid_argument("ESS : Input batch size cannot exceed max batch size\n");
        }
        if (inputLeft.getDepth() != inputRight.getDepth() ||
            output.getDepth() != inputLeft.getDepth()) {
            throw std::invalid_argument("ESS : Batch size of input and output images don't match!\n");
        }
        // Preprocess input images (resize, normalize, etc.)
        m_preprocess->execute(m_inputLeftPlanar, m_inputRightPlanar, inputLeft,
            inputRight, stream);

        // Register output tensors and set CUDA stream
        CHECK_ERROR_CODE(client->setOutput(output,  m_outputName));
        CHECK_ERROR_CODE(client->setOutput(confMap, m_confMapName));
        CHECK_ERROR_CODE(client->setCudaStream(stream));
        // Run inference
        CHECK_ERROR_CODE(client->infer(batchSize));
        // Synchronize CUDA stream to ensure inference is complete
        CHECK_ERROR(cudaStreamSynchronize(stream));
    }

    // Execute inference using NCHW float32 input (already preprocessed)
    void execute(Tensor_NHWC_C1_F32 & output, Tensor_NHWC_C1_F32 & confMap,
                 const Tensor_NCHW_C3_F32 & inputLeft, const Tensor_NCHW_C3_F32 & inputRight,
                 cudaStream_t stream) {
        size_t batchSize = inputLeft.getDepth();
        // Validate input types and batch sizes
        if (inputLeft.isCPU() || inputRight.isCPU()) {
            throw std::invalid_argument("ESS : Input type should be GPU buffer");
        }
        if (inputLeft.getDepth() > m_maxBatchSize || inputRight.getDepth() > m_maxBatchSize) {
            throw std::invalid_argument("ESS : Input batch size cannot exceed max batch size\n");
        }
        if (inputLeft.getDepth() != inputRight.getDepth() ||
            output.getDepth() != inputLeft.getDepth()) {
            throw std::invalid_argument("ESS : Batch size of input and output images don't match!\n");
        }
        if (inputLeft.getWidth() != m_networkInputWidth ||
            inputLeft.getHeight() != m_networkInputHeight) {
            throw std::invalid_argument("ESS : Left preprocessed input does not match network input dimensions!\n");
        }
        if (inputRight.getWidth() != m_networkInputWidth ||
            inputRight.getHeight() != m_networkInputHeight) {
            throw std::invalid_argument("ESS : Right preprocessed input does not match network input dimensions!\n");
        }
        // Register input/output tensors and set CUDA stream
        CHECK_ERROR_CODE(client->setInput(inputLeft, m_leftInputLayerName));
        CHECK_ERROR_CODE(client->setInput(inputRight, m_rightInputLayerName));
        CHECK_ERROR_CODE(client->setOutput(output,  m_outputName));
        CHECK_ERROR_CODE(client->setOutput(confMap, m_confMapName));
        CHECK_ERROR_CODE(client->setCudaStream(stream));
        // Run inference
        CHECK_ERROR_CODE(client->infer(batchSize));
        // Synchronize CUDA stream to ensure inference is complete
        CHECK_ERROR(cudaStreamSynchronize(stream));
    }
};

// ESS class constructor: creates the implementation object
ESS::ESS(const ImagePreProcessingParams & imgParams,
         const ModelInputParams & modelInputParams,
         const ModelBuildParams & modelBuildParams,
         const ModelInferenceParams & modelInferParams,
         const ESSPreProcessorParams & essParams)
    : m_pImpl(new ESSImpl(imgParams, modelInputParams,
              modelBuildParams, modelInferParams, essParams)) {}

// ESS class destructor
ESS::~ESS() {}

// Execute inference (overload for preprocessed input)
void ESS::execute(Tensor_NHWC_C1_F32 & output, Tensor_NHWC_C1_F32 & confMap,
                  const Tensor_NCHW_C3_F32 & inputLeft, const Tensor_NCHW_C3_F32 & inputRight,
                  cudaStream_t stream) {
    m_pImpl->execute(output, confMap, inputLeft, inputRight, stream);
}

// Execute inference (overload for raw input)
void ESS::execute(Tensor_NHWC_C1_F32 & output, Tensor_NHWC_C1_F32 & confMap,
                  const Tensor_NHWC_C3_U8 & inputLeft, const Tensor_NHWC_C3_U8 & inputRight,
                  cudaStream_t stream) {
    m_pImpl->execute(output, confMap, inputLeft, inputRight, stream);
}

// Get output height
size_t ESS::getModelOutputHeight() {
    return m_pImpl->getModelOutputHeight();
}

// Get output width
size_t ESS::getModelOutputWidth() {
    return m_pImpl->getModelOutputWidth();
}

}  // namespace ess
}  // namespace isaac
}  // namespace nvidia

/*
This file implements the core logic for the ESS (Efficient Supervised Stereo) inference engine. 
It manages the entire stereo depth estimation process, including preprocessing input images, running 
inference using a TensorRT-accelerated neural network, and producing both disparity and confidence 
maps. The code supports both raw and preprocessed image inputs, handles GPU memory and CUDA streams, 
and ensures all input and output tensors match the expected dimensions and batch sizes. By encapsulating 
the setup, execution, and resource management for stereo inference, this file serves as the computational 
backbone for generating depth information from stereo camera images in real time.
*/
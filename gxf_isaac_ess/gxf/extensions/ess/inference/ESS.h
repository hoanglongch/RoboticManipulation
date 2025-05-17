#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "extensions/tensorops/core/Array.h"
#include "extensions/tensorops/core/Core.h"
#include "extensions/tensorops/core/Image.h"
#include "extensions/tensorops/core/Tensor.h"

namespace nvidia {
namespace isaac {
namespace ess {

// Type aliases for tensor layouts and types used throughout the ESS pipeline.
using TensorLayout = cvcore::tensor_ops::TensorLayout;
using ChannelType = cvcore::tensor_ops::ChannelType;
using ChannelCount =  cvcore::tensor_ops::ChannelCount;
using TensorDimension = cvcore::tensor_ops::TensorDimension;

// Common tensor types for image and network data.
using Tensor_NHWC_C3_U8 = cvcore::tensor_ops::Tensor<TensorLayout::NHWC, ChannelCount::C3, ChannelType::U8>;
using Tensor_NHWC_C3_F32 = cvcore::tensor_ops::Tensor<TensorLayout::NHWC, ChannelCount::C3, ChannelType::F32>;
using Tensor_NHWC_C1_F32 = cvcore::tensor_ops::Tensor<TensorLayout::NHWC, ChannelCount::C1, ChannelType::F32>;
using Tensor_NCHW_C3_F32 = cvcore::tensor_ops::Tensor<TensorLayout::NCHW, ChannelCount::C3, ChannelType::F32>;
using Tensor_HWC_C3_U8 = cvcore::tensor_ops::Tensor<TensorLayout::HWC, ChannelCount::C3, ChannelType::U8>;
using DisparityLevels = cvcore::tensor_ops::Array<int>;
using TensorBase = cvcore::tensor_ops::TensorBase;
using ImagePreProcessingParams = cvcore::tensor_ops::ImagePreProcessingParams;
using ImageType = cvcore::tensor_ops::ImageType;

// Struct describing the input type required by the model (e.g., batch size, image format).
struct ModelInputParams {
    size_t maxBatchSize;      // Maximum batch size supported by the network
    cvcore::tensor_ops::ImageType modelInputType; // Input image layout/type
};

// Struct describing parameters for building the TensorRT engine.
struct ModelBuildParams {
    bool force_engine_update;      // Force engine rebuild even if engine file exists
    std::string onnx_file_path;    // Path to ONNX model file
    bool enable_fp16;              // Enable FP16 precision
    int64_t max_workspace_size;    // Max workspace size for TensorRT
    int64_t dla_core;              // DLA core index (if using DLA)
};

// Struct describing the model inference parameters (engine file, input/output layer names).
struct ModelInferenceParams {
    std::string engineFilePath;             // Path to TensorRT engine file
    std::vector<std::string> inputLayers;   // Names of input layers
    std::vector<std::string> outputLayers;  // Names of output layers
};

// Enum for preprocessing algorithm: resize or center crop.
enum class PreProcessType : uint8_t {
    RESIZE = 0,  // Resize to network dimensions without maintaining aspect ratio
    CENTER_CROP  // Crop to network dimensions from center of image
};

// Struct describing the parameters for ESS preprocessing.
struct ESSPreProcessorParams {
    PreProcessType preProcessType; // Which preprocessing algorithm to use
};

// Default parameter declarations (defined elsewhere).
CVCORE_API extern const ImagePreProcessingParams defaultPreProcessorParams;
CVCORE_API extern const ModelInputParams defaultModelInputParams;
CVCORE_API extern const ModelInferenceParams defaultInferenceParams;
CVCORE_API extern const ESSPreProcessorParams defaultESSPreProcessorParams;
CVCORE_API extern const int InferencerVersion;

/*
 * Interface for running pre-processing on ESS model.
 * Converts raw stereo images into the format required by the neural network.
 */
class CVCORE_API ESSPreProcessor {
 public:
    ESSPreProcessor() = delete; // Default constructor not allowed

    // Constructor: sets up preprocessing pipeline with all required parameters.
    ESSPreProcessor(const ImagePreProcessingParams & preProcessorParams,
        const ModelInputParams & modelInputParams,
        size_t output_width, size_t output_height,
        const ESSPreProcessorParams & essPreProcessorParams);

    ~ESSPreProcessor();

    // Main interface to run pre-processing on a batch of stereo images.
    // Converts left/right input images (uint8, NHWC) to network-ready format (float32, NCHW).
    void execute(Tensor_NCHW_C3_F32& leftOutput, Tensor_NCHW_C3_F32& rightOutput,
                 const Tensor_NHWC_C3_U8& leftInput, const Tensor_NHWC_C3_U8& rightInput,
                 cudaStream_t stream = 0);

 private:
    struct ESSPreProcessorImpl; // Opaque pointer to implementation details (PIMPL idiom)
    std::unique_ptr<ESSPreProcessorImpl> m_pImpl;
};

/**
 * Main class for Efficient Supervised Stereo (ESS) inference.
 * Handles model loading, preprocessing, inference, and output formatting.
 */
class CVCORE_API ESS {
 public:
    // Constructor: initializes the ESS pipeline with all required parameters.
    ESS(const ImagePreProcessingParams & imgparams,
        const ModelInputParams & modelInputParams,
        const ModelBuildParams & modelBuildParams,
        const ModelInferenceParams & modelInferParams,
        const ESSPreProcessorParams & essPreProcessorParams);

    ESS() = delete; // Default constructor not allowed

    ~ESS();

    // Inference function for raw BGR/RGB images (NHWC uint8).
    // Outputs disparity and confidence maps (NHWC float32).
    void execute(Tensor_NHWC_C1_F32 & disparityMap, Tensor_NHWC_C1_F32 & confMap,
                 const Tensor_NHWC_C3_U8 & leftInput, const Tensor_NHWC_C3_U8 & rightInput,
                 cudaStream_t stream = 0);

    // Inference function for preprocessed images (NCHW float32).
    // Outputs disparity and confidence maps (NHWC float32).
    void execute(Tensor_NHWC_C1_F32& disparityMap,  Tensor_NHWC_C1_F32 & confMap,
                 const Tensor_NCHW_C3_F32& leftInput, const Tensor_NCHW_C3_F32& rightInput,
                 cudaStream_t stream = 0);

    // Helper to get model output height (disparity/confidence map height).
    size_t getModelOutputHeight();

    // Helper to get model output width (disparity/confidence map width).
    size_t getModelOutputWidth();

 private:
    struct ESSImpl; // Opaque pointer to implementation details (PIMPL idiom)
    std::unique_ptr<ESSImpl> m_pImpl;
};

/**
 * Post-processing class for ESS output.
 * Handles resizing or scaling of the disparity map to match original image resolution.
 */
class CVCORE_API ESSPostProcessor {
 public:
    // Constructor: sets up post-processing with model parameters and output size.
    ESSPostProcessor(const ModelInputParams & modelParams,
                     size_t output_width, size_t output_height);

    ESSPostProcessor() = delete; // Default constructor not allowed

    ~ESSPostProcessor();

    // Post-processes the network output disparity map (e.g., resize to original image size).
    void execute(Tensor_NHWC_C1_F32 & outputdisparityMap,
        const Tensor_NHWC_C1_F32 & inputdisparityMap,
        cudaStream_t stream = 0);

 private:
    struct ESSPostProcessorImpl; // Opaque pointer to implementation details (PIMPL idiom)
    std::unique_ptr<ESSPostProcessorImpl> m_pImpl;
};

}  // namespace ess
}  // namespace isaac
}  // namespace nvidia

/*
This header file defines the core C++ interfaces and data structures for the (ESS) inference pipeline. 
It provides class declarations for the main ESS model, including preprocessing (ESSPreProcessor), 
inference (ESS), and post-processing (ESSPostProcessor). These classes manage the conversion of raw 
stereo images into network-ready tensors, execute the neural network to produce disparity and confidence 
maps, and resize or scale the output as needed. The file also defines supporting types, parameters, and 
configuration structures, enabling flexible and efficient stereo depth estimation using CUDA and TensorRT 
on NVIDIA hardware.
*/
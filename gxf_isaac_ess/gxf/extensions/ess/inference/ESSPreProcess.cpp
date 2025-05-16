// Implements stereo image preprocessing, including resizing, normalization, and color conversion,
// all using CUDA.
// ESS =  Efficient Supervised Stereo model (ESS)
// Standard libraries for algorithms and error handling.
#include <algorithm>
#include <stdexcept>

// Header dependencies for custom ESS and tensor operations.
#include "extensions/ess/inference/ESS.h"
#include "extensions/tensorops/core/BBoxUtils.h"
#include "extensions/tensorops/core/CVError.h"
#include "extensions/tensorops/core/Image.h"
#include "extensions/tensorops/core/ImageUtils.h"
#include "extensions/tensorops/core/Memory.h"

namespace nvidia {
namespace isaac {
namespace ess {

// Create shortcuts for image types and bounding box.
using ImageType = cvcore::tensor_ops::ImageType;
using BBox = cvcore::tensor_ops::BBox;

// Implementation struct for the ESSPreProcessor.
struct ESSPreProcessor::ESSPreProcessorImpl {
    // Maximum batch size for processing.
    size_t m_maxBatchSize;
    // Expected output dimensions.
    size_t m_outputWidth;
    size_t m_outputHeight;
    // Type of preprocessing to perform (simple resize vs. crop-and-resize).
    PreProcessType m_processType;
    // Parameters for input image preprocessing (includes normalization means and stddev).
    ImagePreProcessingParams m_preProcessorParams;
    // Temporary device (GPU) tensors for resized images.
    Tensor_NHWC_C3_U8 m_resizedDeviceLeftInput;
    Tensor_NHWC_C3_U8 m_resizedDeviceRightInput;
    // Temporary device tensors for normalized images (converted to float32).
    Tensor_NHWC_C3_F32 m_normalizedDeviceLeftInput;
    Tensor_NHWC_C3_F32 m_normalizedDeviceRightInput;
    // Flag to determine if red-blue channel swap is needed.
    bool m_swapRB;

    // Constructor: Initializes processing parameters and allocates GPU tensors.
    ESSPreProcessorImpl(const ImagePreProcessingParams & imgParams,
        const ModelInputParams & modelParams,
        size_t output_width, size_t output_height,
        const ESSPreProcessorParams & essParams)
        : m_maxBatchSize(modelParams.maxBatchSize)
        , m_outputHeight(output_height)
        , m_outputWidth(output_width)
        , m_processType(essParams.preProcessType)
        , m_preProcessorParams(imgParams) {
        // Validate that the input image type is supported.
        if (imgParams.imgType != ImageType::BGR_U8 &&
            imgParams.imgType != ImageType::RGB_U8) {
            throw std::invalid_argument("ESSPreProcessor : Only image types RGB_U8/BGR_U8 are supported\n");
        }
        // Allocate device tensors for both the left and right inputs.
        m_resizedDeviceLeftInput = {output_width, output_height, modelParams.maxBatchSize, false};
        m_resizedDeviceRightInput = {output_width, output_height, modelParams.maxBatchSize, false};
        m_normalizedDeviceLeftInput  = {output_width, output_height, modelParams.maxBatchSize, false};
        m_normalizedDeviceRightInput = {output_width, output_height, modelParams.maxBatchSize, false};
        // Determine if color channel swap (BGR to RGB conversion) is required.
        m_swapRB = imgParams.imgType != modelParams.modelInputType;
    }

    // Process method: Performs resizing, optional cropping, color conversion,
    // normalization, and rearranges data from interleaved to planar format.
    void process(Tensor_NCHW_C3_F32 & outputLeft, Tensor_NCHW_C3_F32 & outputRight,
        const Tensor_NHWC_C3_U8 & inputLeft, const Tensor_NHWC_C3_U8 & inputRight,
        cudaStream_t stream) {
        // Ensure that all tensors are allocated on the GPU.
        if (inputLeft.isCPU() || inputRight.isCPU() || outputLeft.isCPU() ||
            outputRight.isCPU()) {
            throw std::invalid_argument("ESSPreProcessor : Input/Output Tensor must be GPU Tensor.");
        }

        // Check that the output dimensions match expected dimensions.
        if (outputLeft.getWidth() != m_outputWidth || outputLeft.getHeight() != m_outputHeight ||
            outputRight.getWidth() != m_outputWidth || outputRight.getHeight() != m_outputHeight) {
            throw std::invalid_argument(
                "ESSPreProcessor : Output Tensor dimension does not match network input requirement");
        }

        // Validate that both left and right inputs have the same dimensions.
        if (inputLeft.getWidth() != inputRight.getWidth() ||
            inputLeft.getHeight() != inputRight.getHeight()) {
            throw std::invalid_argument("ESSPreProcessor : Input tensor dimensions don't match");
        }

        // Validate batch sizes between input and output tensors.
        if (outputLeft.getDepth() != inputLeft.getDepth() ||
            outputRight.getDepth() != inputRight.getDepth() ||
            inputLeft.getDepth() != inputRight.getDepth()) {
            throw std::invalid_argument("ESSPreProcessor : Input/Output Tensor batchsize mismatch.");
        }
        if (outputLeft.getDepth() > m_maxBatchSize) {
            throw std::invalid_argument("ESSPreProcessor : Input/Output batchsize exceeds max batch size.");
        }

        // Retrieve common dimensions and batch size.
        const size_t batchSize   = inputLeft.getDepth();
        const size_t inputWidth  = inputLeft.getWidth();
        const size_t inputHeight = inputLeft.getHeight();

        // Depending on the pre-processing type, either perform a straight resize
        // or a more complex center-crop followed by resizing.
        if (m_processType == PreProcessType::RESIZE) {
            cvcore::tensor_ops::Resize(m_resizedDeviceLeftInput, inputLeft,
                false, cvcore::tensor_ops::INTERP_LINEAR, stream);
            cvcore::tensor_ops::Resize(m_resizedDeviceRightInput, inputRight,
                false, cvcore::tensor_ops::INTERP_LINEAR, stream);
        } else {
            // Compute center and offsets for cropping.
            const float centerX = static_cast<float>(inputWidth) / 2.0;
            const float centerY = static_cast<float>(inputHeight) / 2.0;
            const float offsetX = static_cast<float>(m_outputWidth) / 2.0;
            const float offsetY = static_cast<float>(m_outputHeight) / 2.0;
            BBox srcCrop, dstCrop;
            // Destination crop covers the whole output image.
            dstCrop = {0, 0, static_cast<int>(m_outputWidth - 1), static_cast<int>(m_outputHeight - 1)};
            // Calculate source crop ensuring it stays within image boundaries.
            srcCrop.xmin = std::max(0, static_cast<int>(centerX - offsetX));
            srcCrop.ymin = std::max(0, static_cast<int>(centerY - offsetY));
            srcCrop.xmax = std::min(static_cast<int>(m_outputWidth - 1), static_cast<int>(centerX + offsetX));
            srcCrop.ymax = std::min(static_cast<int>(m_outputHeight - 1), static_cast<int>(centerY + offsetY));
            // Loop over each batch element to perform crop-and-resize.
            for (size_t i = 0; i < batchSize; i++) {
                // Create view tensors for the current batch element.
                Tensor_HWC_C3_U8 inputLeftCrop(
                    inputWidth, inputHeight,
                    const_cast<uint8_t *>(inputLeft.getData()) + i * inputLeft.getStride(TensorDimension::DEPTH),
                    false);
                Tensor_HWC_C3_U8 outputLeftCrop(
                    m_outputWidth, m_outputHeight,
                    m_resizedDeviceLeftInput.getData() + i * m_resizedDeviceLeftInput.getStride(TensorDimension::DEPTH),
                    false);
                Tensor_HWC_C3_U8 inputRightCrop(
                    inputWidth, inputHeight,
                    const_cast<uint8_t *>(inputRight.getData()) + i * inputRight.getStride(TensorDimension::DEPTH),
                    false);
                Tensor_HWC_C3_U8 outputRightCrop(
                    m_outputWidth, m_outputHeight,
                    m_resizedDeviceRightInput.getData() + i * m_resizedDeviceRightInput.getStride(TensorDimension::DEPTH),
                    false);
                // Perform cropping and resizing using linear interpolation.
                cvcore::tensor_ops::CropAndResize(outputLeftCrop, inputLeftCrop, dstCrop, srcCrop,
                    cvcore::tensor_ops::InterpolationType::INTERP_LINEAR, stream);
                cvcore::tensor_ops::CropAndResize(outputRightCrop, inputRightCrop, dstCrop, srcCrop,
                    cvcore::tensor_ops::InterpolationType::INTERP_LINEAR, stream);
            }
        }

        // If required, swap red and blue channels (BGR to RGB conversion).
        if (m_swapRB) {
            cvcore::tensor_ops::ConvertColorFormat(m_resizedDeviceLeftInput,
                m_resizedDeviceLeftInput, cvcore::tensor_ops::BGR2RGB, stream);
            cvcore::tensor_ops::ConvertColorFormat(m_resizedDeviceRightInput,
                m_resizedDeviceRightInput, cvcore::tensor_ops::BGR2RGB, stream);
        }

        // Compute scaling factors for normalization based on provided mean/std values.
        float scale[3];
        for (size_t i = 0; i < 3; i++) {
            scale[i] = m_preProcessorParams.normalization[i] / m_preProcessorParams.stdDev[i];
        }

        // Normalize the resized images using the computed scale and provided pixel mean.
        cvcore::tensor_ops::Normalize(m_normalizedDeviceLeftInput, m_resizedDeviceLeftInput,
            scale, m_preProcessorParams.pixelMean, stream);
        cvcore::tensor_ops::Normalize(m_normalizedDeviceRightInput, m_resizedDeviceRightInput,
            scale, m_preProcessorParams.pixelMean, stream);
        // Convert images from interleaved (C3) to planar format for network input requirements.
        cvcore::tensor_ops::InterleavedToPlanar(outputLeft, m_normalizedDeviceLeftInput, stream);
        cvcore::tensor_ops::InterleavedToPlanar(outputRight, m_normalizedDeviceRightInput, stream);
    }
};

// Public method: Executes the preprocessing pipeline on stereo image pairs.
void ESSPreProcessor::execute(Tensor_NCHW_C3_F32 & outputLeft,
    Tensor_NCHW_C3_F32 & outputRight,
    const Tensor_NHWC_C3_U8 & inputLeft, const Tensor_NHWC_C3_U8 & inputRight,
    cudaStream_t stream) {
    m_pImpl->process(outputLeft, outputRight, inputLeft, inputRight, stream);
}

// Constructor: Creates an implementation instance with provided parameters.
ESSPreProcessor::ESSPreProcessor(const ImagePreProcessingParams & preProcessorParams,
    const ModelInputParams & modelInputParams,
    const size_t output_width, const size_t output_height,
    const ESSPreProcessorParams & essParams)
    : m_pImpl(new ESSPreProcessor::ESSPreProcessorImpl(preProcessorParams,
        modelInputParams, output_width, output_height, essParams)) {}

// Destructor: Clean up resources.
ESSPreProcessor::~ESSPreProcessor() {}

}  // namespace ess
}  // namespace isaac
}  // namespace nvidia
#include <iostream>
#include <stdexcept>

#include "extensions/ess/inference/ESS.h"
#include "extensions/tensorops/core/BBoxUtils.h"
#include "extensions/tensorops/core/CVError.h"
#include "extensions/tensorops/core/ImageUtils.h"
#include "extensions/tensorops/core/Memory.h"

namespace nvidia {
namespace isaac {
namespace ess {

// Implementation details for the ESSPostProcessor class.
// This struct handles resizing and scaling of the disparity map output from the network.
struct ESSPostProcessor::ESSPostProcessorImpl {
    // Constructor: initializes internal buffers and parameters.
    ESSPostProcessorImpl(const ModelInputParams & modelParams,
        const size_t output_width, const size_t output_height)
        : m_maxBatchSize(modelParams.maxBatchSize)
        , m_networkWidth(output_width)
        , m_networkHeight(output_height) {
        // Allocate device (GPU) buffers for scaled and output disparity maps.
        m_scaledDisparityDevice = {m_networkWidth, m_networkHeight, m_maxBatchSize, false};
        m_outputDisparityDevice = {m_networkWidth, m_networkHeight, m_maxBatchSize, false};
    }

    // Ensures the output buffer matches the requested output size.
    void resizeBuffers(std::size_t width, std::size_t height) {
        if (m_outputDisparityDevice.getWidth() == width &&
            m_outputDisparityDevice.getHeight() == height) {
            return;
        }
        // Reallocate output buffer if size changed.
        m_outputDisparityDevice = {width, height, m_maxBatchSize, false};
    }

    // Main post-processing function.
    // Scales and resizes the disparity map to match the desired output size.
    void process(Tensor_NHWC_C1_F32 & outputDisparity, const Tensor_NHWC_C1_F32 & inputDisparity,
                 cudaStream_t stream) {
        // Ensure input is on the GPU.
        if (inputDisparity.isCPU()) {
            throw std::invalid_argument("ESSPostProcessor : Input Tensor must be GPU Tensor.");
        }

        // Check that input dimensions match the network's output size.
        if (inputDisparity.getWidth() != m_networkWidth ||
            inputDisparity.getHeight() != m_networkHeight) {
          std::cerr << "input disparity: " << inputDisparity.getWidth() << "x"
                    << inputDisparity.getHeight() << ", network size: "
                    << m_networkWidth << "x" << m_networkHeight << std::endl;
            throw std::invalid_argument(
                "ESSPostProcessor : Input Tensor dimension "
                "does not match network input "
                "requirement");
        }

        // Check that batch sizes match.
        if (inputDisparity.getDepth() != outputDisparity.getDepth()) {
            throw std::invalid_argument("ESSPostProcessor : Input/Output Tensor batchsize"
                 "mismatch.");
        }

        const size_t batchSize = inputDisparity.getDepth();
        if (batchSize > m_maxBatchSize) {
            throw std::invalid_argument("ESSPostProcessor : Input batchsize exceeds Max"
                "Batch size.");
        }
        const size_t outputWidth  = outputDisparity.getWidth();
        const size_t outputHeight = outputDisparity.getHeight();

        // Compute scaling factor for disparity values based on output vs. network width.
        const float scale = static_cast<float>(outputWidth) / m_networkWidth;
        Tensor_NHWC_C1_F32 scaledDisparity(m_scaledDisparityDevice.getWidth(),
            m_scaledDisparityDevice.getHeight(), batchSize,
            m_scaledDisparityDevice.getData(), false);

        // Scale the disparity values.
        cvcore::tensor_ops::Normalize(scaledDisparity, inputDisparity, scale, 0, stream);

        // If output is on GPU, resize directly.
        if (!outputDisparity.isCPU()) {
            cvcore::tensor_ops::Resize(outputDisparity, m_scaledDisparityDevice, stream);
        } else {
            // If output is on CPU, resize on GPU then copy to CPU.
            resizeBuffers(outputWidth, outputHeight);
            Tensor_NHWC_C1_F32 outputDisparityDevice(m_outputDisparityDevice.getWidth(),
                m_outputDisparityDevice.getHeight(), batchSize,
                m_outputDisparityDevice.getData(), false);
            cvcore::tensor_ops::Resize(outputDisparityDevice, m_scaledDisparityDevice, stream);
            cvcore::tensor_ops::Copy(outputDisparity, outputDisparityDevice, stream);
            CHECK_ERROR(cudaStreamSynchronize(stream));
        }
    }

    // Internal state: batch size, network dimensions, and device buffers.
    size_t m_maxBatchSize;
    size_t m_networkWidth, m_networkHeight;
    Tensor_NHWC_C1_F32 m_scaledDisparityDevice;
    Tensor_NHWC_C1_F32 m_outputDisparityDevice;
};

// Public interface: calls the internal process function.
void ESSPostProcessor::execute(Tensor_NHWC_C1_F32 & outputDisparity,
    const Tensor_NHWC_C1_F32 & inputDisparity, cudaStream_t stream) {
    m_pImpl->process(outputDisparity, inputDisparity, stream);
}

// Constructor: creates the implementation object with model parameters and output size.
ESSPostProcessor::ESSPostProcessor(const ModelInputParams & modelInputParams,
    size_t output_width, size_t output_height)
    : m_pImpl(new ESSPostProcessor::ESSPostProcessorImpl(modelInputParams,
              output_width, output_height)) {}

// Destructor: cleans up resources.
ESSPostProcessor::~ESSPostProcessor() {}

}  // namespace ess
}  // namespace isaac
}  // namespace nvidia

/*
This file implements the post-processing logic for the ESS stereo depth estimation pipeline. 
Specifically, it takes the raw disparity map output from the neural network (which estimates 
pixel-wise differences between left and right stereo images) and performs scaling and resizing 
operations to match the desired output resolution. The code ensures that all tensor dimensions 
and batch sizes are valid, handles both GPU and CPU output cases, and uses CUDA streams for 
efficient processing. This post-processing step is essential for converting the neural network's 
output into a usable disparity map for downstream tasks such as depth calculation, navigation, 
or robotic manipulation.
*/

#pragma once

#include "NvInferRuntime.h"
#include "extensions/tensorops/core/Tensor.h"
#include "Errors.h"

namespace cvcore {
namespace inferencer {

/*
 * Maps tensorrt datatype to cvcore Channel type.
 * @param channelType cvcore channel type.
 * @param dtype tensorrt datatype
 * return error code
 */
std::error_code getCVCoreChannelTypeFromTensorRT(
    cvcore::tensor_ops::ChannelType& channelType,
    nvinfer1::DataType dtype);

/*
 * Maps tensorrt datatype to cvcore Channel type.
 * @param channelLayout cvcore channel type.
 * @param dtype tensorrt layout
 * return error code
 */
std::error_code getCVCoreChannelLayoutFromTensorRT(
    cvcore::tensor_ops::TensorLayout& channelLayout,
    nvinfer1::TensorFormat tensorFormat);

}  // namespace inferencer
}  // namespace cvcore

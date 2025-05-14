

#include "gems/dnn_inferencer/inferencer/TensorRTUtils.h"
#include <iostream>

namespace cvcore {
namespace inferencer {
using TensorLayout = cvcore::tensor_ops::TensorLayout;
using ChannelType = cvcore::tensor_ops::ChannelType;
using ChannelCount =  cvcore::tensor_ops::ChannelCount;
using ErrorCode = cvcore::tensor_ops::ErrorCode;

std::error_code getCVCoreChannelTypeFromTensorRT(ChannelType& channelType,
    nvinfer1::DataType dtype) {
    if (dtype == nvinfer1::DataType::kINT8) {
        channelType = ChannelType::U8;
    } else if (dtype == nvinfer1::DataType::kHALF) {
        channelType = ChannelType::F16;
    } else if (dtype == nvinfer1::DataType::kFLOAT) {
        channelType = ChannelType::F32;
    } else {
        return ErrorCode::INVALID_OPERATION;
    }

    return ErrorCode::SUCCESS;
}

std::error_code getCVCoreChannelLayoutFromTensorRT(TensorLayout& channelLayout,
    nvinfer1::TensorFormat tensorFormat) {
    if (tensorFormat == nvinfer1::TensorFormat::kLINEAR) {
        channelLayout = TensorLayout::NCHW;
    } else if (tensorFormat == nvinfer1::TensorFormat::kHWC) {
        channelLayout = TensorLayout::HWC;
    } else {
        return ErrorCode::INVALID_OPERATION;
    }

    return ErrorCode::SUCCESS;
}

}  // namespace inferencer
}  // namespace cvcore

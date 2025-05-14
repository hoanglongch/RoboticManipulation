
#ifdef ENABLE_TRITON

#include <iostream>
#include <string>

#include "gems/dnn_inferencer/inferencer/TritonUtils.h"

namespace cvcore {
namespace inferencer {
using ChannelType = cvcore::tensor_ops::ChannelType;
bool getCVCoreChannelType(ChannelType& channelType, std::string dtype) {
    if (dtype.compare("UINT8") == 0) {
        channelType = ChannelType::U8;
    } else if (dtype.compare("UINT16") == 0) {
        channelType = ChannelType::U16;
    } else if (dtype.compare("FP16") == 0) {
        channelType = ChannelType::F16;
    } else if (dtype.compare("FP32") == 0) {
        channelType = ChannelType::F32;
    } else if (dtype.compare("FP64") == 0) {
        channelType = ChannelType::F64;
    } else {
        return false;
    }

    return true;
}

bool getTritonChannelType(std::string& dtype, ChannelType channelType) {
    if (channelType == ChannelType::U8) {
        dtype = "UINT8";
    } else if (channelType == ChannelType::U16) {
        dtype = "UINT16";
    } else if (channelType == ChannelType::F16) {
        dtype = "FP16";
    } else if (channelType == ChannelType::F32) {
        dtype = "FP32";
    } else if (channelType == ChannelType::F64) {
        dtype = "FP64";
    } else {
        return false;
    }

    return true;
}

}  // namespace inferencer
}  // namespace cvcore
#endif

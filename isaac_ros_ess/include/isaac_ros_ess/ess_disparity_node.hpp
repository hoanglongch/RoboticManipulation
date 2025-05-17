// The purpose of this file: This file defines the ESSDisparityNode class, which is a ROS 2 node for processing disparity images using the NVIDIA Isaac SDK.
// It includes the necessary headers, defines the class and its methods, and provides a constructor for initializing the node with specific parameters.

#ifndef ISAAC_ROS_ESS__ESS_DISPARITY_NODE_HPP_
#define ISAAC_ROS_ESS__ESS_DISPARITY_NODE_HPP_

#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"                  // ROS 2 C++ client library
#include "isaac_ros_nitros/nitros_node.hpp"   // Base class for Nitros-enabled nodes

using StringList = std::vector<std::string>;  // Alias for a list of strings

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

// Main class for the stereo disparity node, inheriting from NitrosNode
class ESSDisparityNode : public nitros::NitrosNode
{
public:
  // Constructor: initializes the node with ROS 2 node options
  explicit ESSDisparityNode(const rclcpp::NodeOptions &);

  // Destructor: cleans up resources
  ~ESSDisparityNode();

  // Delete copy constructor to prevent copying
  ESSDisparityNode(const ESSDisparityNode &) = delete;

  // Delete assignment operator to prevent copying
  ESSDisparityNode & operator=(const ESSDisparityNode &) = delete;

  // Callback executed before the GXF graph is loaded (for parameter setup)
  void preLoadGraphCallback() override;

  // Callback executed after the GXF graph is loaded (for parameter setup)
  void postLoadGraphCallback() override;

private:
  // Image type (e.g., "rgb8", "bgr8") expected by the model
  const std::string image_type_;

  // Width of the model's input layer (in pixels)
  const int input_layer_width_;

  // Height of the model's input layer (in pixels)
  const int input_layer_height_;

  // Model input type (e.g., "NCHW", "NHWC")
  const std::string model_input_type_;

  // Path to the ONNX model file
  const std::string onnx_file_path_;

  // Path to the TensorRT engine file
  const std::string engine_file_path_;

  // Names of the model's input layers
  const std::vector<std::string> input_layers_name_;

  // Names of the model's output layers
  const std::vector<std::string> output_layers_name_;

  // Threshold value for post-processing (e.g., confidence threshold)
  const float threshold_;

  // Number of frames to skip for throttling (controls processing rate)
  const int throttler_skip_;
};

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_ESS__ESS_DISPARITY_NODE_HPP_

/*
This header file defines the ESSDisparityNode class, which implements a ROS 2 node for stereo disparity 
estimation using NVIDIA's Isaac SDK and the Nitros framework. The class inherits from nitros::NitrosNode 
and provides methods for initializing the node, managing its lifecycle, and configuring the underlying 
processing graph. It encapsulates parameters related to image type, model input/output, ONNX and engine 
file paths, and processing thresholds. This node serves as the main entry point for integrating deep-learned 
stereo depth estimation into a ROS 2 robotic system.
*/

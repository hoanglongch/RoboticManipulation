// The purpose of this file: This file defines the ESSDisparityNode class, which is a ROS 2 node for processing disparity images using the NVIDIA Isaac SDK.
// It includes the necessary headers, defines the class and its methods, and provides a constructor for initializing the node with specific parameters.

#ifndef ISAAC_ROS_ESS__ESS_DISPARITY_NODE_HPP_
#define ISAAC_ROS_ESS__ESS_DISPARITY_NODE_HPP_

#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

using StringList = std::vector<std::string>;

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

class ESSDisparityNode : public nitros::NitrosNode
{
public:
  explicit ESSDisparityNode(const rclcpp::NodeOptions &);

  ~ESSDisparityNode();

  ESSDisparityNode(const ESSDisparityNode &) = delete;

  ESSDisparityNode & operator=(const ESSDisparityNode &) = delete;

  // The callback for submitting parameters to the node's graph
  void preLoadGraphCallback() override;
  void postLoadGraphCallback() override;

private:
  const std::string image_type_;
  const int input_layer_width_;
  const int input_layer_height_;
  const std::string model_input_type_;
  const std::string onnx_file_path_;
  const std::string engine_file_path_;
  const std::vector<std::string> input_layers_name_;
  const std::vector<std::string> output_layers_name_;
  const float threshold_;
  const int throttler_skip_;
};

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_ESS__ESS_DISPARITY_NODE_HPP_

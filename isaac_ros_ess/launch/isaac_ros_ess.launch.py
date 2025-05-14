

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='',
            description='The absolute path to the ESS engine plan.'),
        DeclareLaunchArgument(
            'threshold',
            default_value='0.4',
            description='Threshold value ranges between 0.0 and 1.0 '
            'for filtering disparity with confidence.'),
        DeclareLaunchArgument(
            'input_layer_width',
            default_value='960',
            description='Input layer width'),
        DeclareLaunchArgument(
            'input_layer_height',
            default_value='576',
            description='Input layer height'),
    ]
    engine_file_path = LaunchConfiguration('engine_file_path')
    threshold = LaunchConfiguration('threshold')
    input_layer_width = LaunchConfiguration('input_layer_width')
    input_layer_height = LaunchConfiguration('input_layer_height')

    disparity_node = ComposableNode(
        name='disparity',
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        parameters=[{'engine_file_path': engine_file_path,
                     'threshold': threshold,
                     'input_layer_width': input_layer_width,
                     'input_layer_height': input_layer_height}],
    )

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='disparity',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[disparity_node],
        output='screen'
    )

    return (launch.LaunchDescription(launch_args + [container]))

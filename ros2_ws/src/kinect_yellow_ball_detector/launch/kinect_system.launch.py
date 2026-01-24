from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    device_type_arg = DeclareLaunchArgument(
        'device_type',
        default_value='freenect1',
        description='Type of Kinect device: freenect1 (Xbox 360), freenect2 (Xbox One), or mock'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to custom YOLOv8 model'
    )

    driver_node = Node(
        package='kinect_yellow_ball_detector',
        executable='driver_node',
        name='kinect_driver_node',
        output='screen',
        parameters=[{
            'device_type': LaunchConfiguration('device_type'),
            'model_path': LaunchConfiguration('model_path')
        }]
    )

    viz_node = Node(
        package='kinect_yellow_ball_detector',
        executable='visualization_node',
        name='visualization_node',
        output='screen'
    )

    return LaunchDescription([
        device_type_arg,
        model_path_arg,
        driver_node,
        viz_node
    ])

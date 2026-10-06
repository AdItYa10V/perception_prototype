import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node

PKG = 'auv_perception_pixel'


def generate_launch_description():
    share = get_package_share_directory(PKG)
    cfg = lambda f: os.path.join(share, 'config', f)
    A = DeclareLaunchArgument
    args = [
        A('camera_topic', default_value='/camera/image_raw'),
        A('calib_file', default_value=cfg('camera_calib.yaml')),
        A('model_path', default_value=os.path.join(share, 'models', 'gate_model.engine')),
    ]
    nodes = [
        Node(package=PKG, executable='image_enhancement_node.py', name='image_enhancement',
             parameters=[cfg('enhancement.yaml'), {'input_topic': LC('camera_topic'),
                                                   'calib_file': LC('calib_file')}],
             output='screen'),
        Node(package=PKG, executable='gate_detector_node.py', name='gate_detector',
             parameters=[cfg('gate_detector.yaml'), {'model_path': LC('model_path'),
                                                     'use_keypoints': False}],
             output='screen'),
        Node(package=PKG, executable='gate_pose_node.py', name='gate_pose',
             parameters=[cfg('gate_pose.yaml')], output='screen'),
        Node(package=PKG, executable='gate_tracker_node.py', name='gate_tracker',
             parameters=[cfg('gate_pose.yaml')], output='screen'),
    ]
    return LaunchDescription(args + nodes)

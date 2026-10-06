import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node

PKG = 'auv_perception_tri'


def generate_launch_description():
    share = get_package_share_directory(PKG)
    cfg = lambda f: os.path.join(share, 'config', f)
    A = DeclareLaunchArgument
    args = [
        A('camera_topic', default_value='/camera/image_raw'),
        A('calib_file', default_value=cfg('camera_calib.yaml')),
        A('model_path', default_value=os.path.join(share, 'models', 'gate_model.engine')),
        # base_link (x fwd, y left, z up) -> camera optical frame. MEASURE these.
        A('cam_x', default_value='0.20'), A('cam_y', default_value='0.0'),
        A('cam_z', default_value='0.0'),
        A('cam_roll', default_value='-1.5708'), A('cam_pitch', default_value='0.0'),
        A('cam_yaw', default_value='-1.5708'),   # these two = camera looking straight ahead
    ]
    calib = {'calib_file': LC('calib_file')}
    nodes = [
        Node(package='tf2_ros', executable='static_transform_publisher', name='cam_tf',
             arguments=['--x', LC('cam_x'), '--y', LC('cam_y'), '--z', LC('cam_z'),
                        '--roll', LC('cam_roll'), '--pitch', LC('cam_pitch'),
                        '--yaw', LC('cam_yaw'),
                        '--frame-id', 'base_link', '--child-frame-id', 'camera_optical_frame']),
        Node(package=PKG, executable='image_enhancement_node.py', name='image_enhancement',
             parameters=[cfg('enhancement.yaml'), {'input_topic': LC('camera_topic')}, calib],
             output='screen'),
        Node(package=PKG, executable='gate_detector_node.py', name='gate_detector',
             parameters=[cfg('gate_detector.yaml'), {'model_path': LC('model_path'),
                                                     'use_keypoints': True}],
             output='screen'),
        Node(package=PKG, executable='gate_pose_node.py', name='gate_pose',
             parameters=[cfg('gate_pose.yaml'), calib], output='screen'),
        Node(package=PKG, executable='gate_tracker_node.py', name='gate_tracker',
             parameters=[cfg('gate_pose.yaml')], output='screen'),
    ]
    return LaunchDescription(args + nodes)

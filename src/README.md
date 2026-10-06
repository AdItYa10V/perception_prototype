# AUV gate perception: three packages, one per level

| package               | level | output (controller input)                                  |
|-----------------------|-------|------------------------------------------------------------|
| auv_perception_pixel  | 1     | gate/steering_filtered (Vector3Stamped: x,y err, size), gate/lock |
| auv_perception_pnp    | 2     | gate/pose_base (PoseStamped in base_link), gate/pose_odom, gate/lock |
| auv_perception_tri    | 3     | same topics as pnp, from multi-frame triangulation         |

Build:   colcon build --packages-select auv_perception_pixel   (etc.)
Run:     ros2 launch auv_perception_pnp perception.launch.py camera_topic:=/camera/image_raw
Needs:   pip ultralytics (JetPack 6 wheels), a TensorRT engine in models/, robot_localization EKF
         publishing odom->base_link (pnp/tri), real underwater camera calibration in config/camera_calib.yaml.

Differences from the requested layout
- Python module dir is named after the package (ament requires it), and utils/ lives inside it.
- Added config/camera_calib.yaml; tracker params live in gate_pose.yaml under `gate_tracker:`.
- Nodes are installed as scripts: `ros2 run <pkg> gate_pose_node.py`.

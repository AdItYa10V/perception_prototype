Put the TensorRT engine here as gate_model.engine. Build it ON the Orin (engines are device/TRT-version specific):

  yolo export model=gate.pt format=engine imgsz=384,640 half=True device=0 workspace=4
  mv gate.engine gate_model.engine

pixel package: detection model (bbox).  pnp/tri packages: YOLO11n-pose trained with the gate keypoints
in the SAME order as model_points in config/gate_pose.yaml.

cmake_minimum_required(VERSION 3.8)
project(auv_perception_pixel)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  ${PROJECT_NAME}/image_enhancement_node.py
  ${PROJECT_NAME}/gate_detector_node.py
  ${PROJECT_NAME}/gate_pose_node.py
  ${PROJECT_NAME}/gate_tracker_node.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config launch models DESTINATION share/${PROJECT_NAME})

ament_package()

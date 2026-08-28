cmake_minimum_required(VERSION 3.5)
project(qi)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MotorCmd.msg"
  "msg/MotorState.msg"
  "msg/IMUState.msg"
  "msg/HandCmd.msg"
  "msg/HandsCmd.msg"
  "msg/HandState.msg"
  "msg/HandsState.msg"
  "msg/LowCmd.msg"
  "msg/LowState.msg"
)

ament_package()

cmake_minimum_required(VERSION 3.8)
project(monoscale_tracker)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(OpenCV REQUIRED)

# cv_bridge is deliberately not used. It links against whichever OpenCV the ROS
# distribution was built with, and this node is meant to be pointed at a second
# OpenCV that has the CUDA modules -- two of them in one process is an ABI
# gamble that shows up as a crash a long way from its cause. Converting an
# Image to a Mat by hand is fifteen lines and removes the question.

# The optical flow has a GPU twin where OpenCV was built with cudaoptflow.
# Nothing is required: without it the node builds and runs exactly as before,
# and `use_cuda` reports that it cannot be honoured rather than failing.
set(HAVE_CUDA_OPTFLOW FALSE)
if(TARGET opencv_cudaoptflow OR "cudaoptflow" IN_LIST OpenCV_LIB_COMPONENTS)
  set(HAVE_CUDA_OPTFLOW TRUE)
endif()
if(NOT HAVE_CUDA_OPTFLOW)
  foreach(component ${OpenCV_LIBS})
    if(component MATCHES "cudaoptflow")
      set(HAVE_CUDA_OPTFLOW TRUE)
    endif()
  endforeach()
endif()

add_executable(feature_tracker src/feature_tracker.cpp)
target_include_directories(feature_tracker PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(feature_tracker ${OpenCV_LIBS})
ament_target_dependencies(feature_tracker rclcpp sensor_msgs std_msgs geometry_msgs rosbag2_cpp)
if(HAVE_CUDA_OPTFLOW)
  message(STATUS "monoscale_tracker: OpenCV has cudaoptflow; the GPU path is built")
  target_compile_definitions(feature_tracker PRIVATE MONOSCALE_TRACKER_HAS_CUDA)
else()
  message(STATUS "monoscale_tracker: no cudaoptflow in OpenCV; CPU only")
endif()
target_compile_options(feature_tracker PRIVATE -O3)

install(TARGETS feature_tracker DESTINATION lib/${PROJECT_NAME})
ament_package()

cmake_minimum_required(VERSION 3.16)
project(monoscale_odometry)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(monoscale_core REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosbag2_cpp REQUIRED)

add_library(monoscale_parameters src/parameters.cpp)
target_include_directories(monoscale_parameters PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(monoscale_parameters rclcpp monoscale_core)
target_compile_options(monoscale_parameters PRIVATE -O3)

# Scores a recorded drive without a graph, which is how a configuration is
# compared against another and how this port is compared against the Python.
add_executable(monoscale_replay src/replay.cpp)
target_link_libraries(monoscale_replay monoscale_parameters)
ament_target_dependencies(monoscale_replay
  rclcpp rosbag2_cpp sensor_msgs std_msgs geometry_msgs monoscale_core)
target_compile_options(monoscale_replay PRIVATE -O3)


# The deployed node.
add_executable(monoscale_odometry src/odometry_node.cpp)
target_link_libraries(monoscale_odometry monoscale_parameters)
ament_target_dependencies(monoscale_odometry
  rclcpp sensor_msgs std_msgs geometry_msgs nav_msgs tf2_ros monoscale_core)
target_compile_options(monoscale_odometry PRIVATE -O3)

install(TARGETS monoscale_parameters
  EXPORT export_monoscale_odometry
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS monoscale_replay monoscale_odometry DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config rviz DESTINATION share/${PROJECT_NAME})

ament_export_targets(export_monoscale_odometry HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp monoscale_core)

ament_package()

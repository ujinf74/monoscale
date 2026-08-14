cmake_minimum_required(VERSION 3.16)
project(monoscale_occupancy_grid_map)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(monoscale_core REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(monoscale_occupancy_grid_map src/occupancy_node.cpp)
ament_target_dependencies(monoscale_occupancy_grid_map
  rclcpp nav_msgs sensor_msgs monoscale_core)
target_compile_options(monoscale_occupancy_grid_map PRIVATE -O3)

install(TARGETS monoscale_occupancy_grid_map DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()

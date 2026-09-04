cmake_minimum_required(VERSION 3.16)
project(monoscale_sweep)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The CUDA path builds the same kernel monoscale_fast runs, from the same
# source, so the C++ node and the python reference share the arithmetic. If no
# CUDA compiler is present the Sweep falls back to the CPU path in sweep.cpp.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

set(SWEEP_SOURCES src/sweep.cpp src/publish.cpp)
# Overridable: the kernel source lives in monoscale_fast, which is not part of
# this repository. Without it (or without a CUDA compiler) the CPU path in
# sweep.cpp serves, bit-compatible per the parity log in the port session.
set(MONOSCALE_FAST_SRC
  /home/i/ros2_ws/hero-release/src/monoscale_fast/src/sweep_kernels.cu
  CACHE FILEPATH "monoscale_fast sweep_kernels.cu for the CUDA backend")
if(CMAKE_CUDA_COMPILER AND EXISTS ${MONOSCALE_FAST_SRC})
  list(APPEND SWEEP_SOURCES src/cuda_backend.cpp ${MONOSCALE_FAST_SRC})
  add_compile_definitions(MONOSCALE_SWEEP_HAS_CUDA)
endif()

add_library(monoscale_sweep_core ${SWEEP_SOURCES})
target_include_directories(monoscale_sweep_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(monoscale_sweep_core ${OpenCV_LIBS}
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
if(CMAKE_CUDA_COMPILER AND EXISTS ${MONOSCALE_FAST_SRC})
  set_target_properties(monoscale_sweep_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(monoscale_sweep_core CUDA::cudart)
endif()

# Static core: the CUDA runtime it pulls in must reach the executable too.
if(CMAKE_CUDA_COMPILER AND EXISTS ${MONOSCALE_FAST_SRC})
  set(SWEEP_CUDART CUDA::cudart)
endif()

# Offline runner: cache -> grid -> npy, for parity against plane_sweep.py.
add_executable(sweep_offline src/offline.cpp)
target_link_libraries(sweep_offline monoscale_sweep_core ${OpenCV_LIBS} ${SWEEP_CUDART})

# The online node.
add_executable(sweep_node src/sweep_node.cpp)
target_link_libraries(sweep_node monoscale_sweep_core ${OpenCV_LIBS} ${SWEEP_CUDART})
ament_target_dependencies(sweep_node
  rclcpp sensor_msgs nav_msgs geometry_msgs cv_bridge)

install(TARGETS monoscale_sweep_core sweep_offline sweep_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

ament_package()

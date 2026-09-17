cmake_minimum_required(VERSION 3.16)
project(monoscale_occupancy_grid_map)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The CUDA path builds the kernel in src/sweep_kernels.cu, which is the same
# source the python reference was checked against, so the two share the
# arithmetic. Without a CUDA compiler the Sweep falls back to sweep.cpp.
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
# The kernel lives here. It used to be pulled from a sibling workspace by an
# absolute path, which meant a fresh clone silently built the CPU path -- 28
# minutes for 674 keyframes against the CUDA path's 0.2 s each -- while the
# README advertised the GPU one. Vendored 2026-09-17 so the repository builds
# what it documents. Still overridable, for a tree that keeps its own copy.
set(MONOSCALE_SWEEP_KERNELS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/sweep_kernels.cu
  CACHE FILEPATH "CUDA kernels for the plane sweep")
if(CMAKE_CUDA_COMPILER AND EXISTS ${MONOSCALE_SWEEP_KERNELS})
  list(APPEND SWEEP_SOURCES src/cuda_backend.cpp ${MONOSCALE_SWEEP_KERNELS})
  add_compile_definitions(MONOSCALE_OCCUPANCY_HAS_CUDA)
endif()

add_library(monoscale_occupancy_core ${SWEEP_SOURCES})
target_include_directories(monoscale_occupancy_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(monoscale_occupancy_core ${OpenCV_LIBS}
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
if(CMAKE_CUDA_COMPILER AND EXISTS ${MONOSCALE_SWEEP_KERNELS})
  set_target_properties(monoscale_occupancy_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(monoscale_occupancy_core CUDA::cudart)
endif()

# Static core: the CUDA runtime it pulls in must reach the executable too.
if(CMAKE_CUDA_COMPILER AND EXISTS ${MONOSCALE_SWEEP_KERNELS})
  set(SWEEP_CUDART CUDA::cudart)
endif()

# Offline runner: cache -> grid -> npy, for parity against plane_sweep.py.
add_executable(sweep_offline src/offline.cpp)
target_link_libraries(sweep_offline monoscale_occupancy_core ${OpenCV_LIBS} ${SWEEP_CUDART})

# The online node.
add_executable(sweep_node src/sweep_node.cpp)
target_link_libraries(sweep_node monoscale_occupancy_core ${OpenCV_LIBS} ${SWEEP_CUDART})
ament_target_dependencies(sweep_node
  rclcpp sensor_msgs nav_msgs geometry_msgs cv_bridge)

install(TARGETS monoscale_occupancy_core sweep_offline sweep_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  # The dense sweep chain is unchanged and is covered by the offline parity
  # run; what is new is the tile storage the chain now reads its window from.
  ament_add_gtest(test_tiles test/test_tiles.cpp)
  target_link_libraries(test_tiles monoscale_occupancy_core ${OpenCV_LIBS} ${SWEEP_CUDART})
endif()

ament_package()

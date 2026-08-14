cmake_minimum_required(VERSION 3.16)
project(monoscale_core)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
# OpenCV is used for the parts that are genuinely its work -- the RANSAC
# similarity fit, triangulation, and the lens models with real coefficients.
# The hot loops are Eigen and plain C++ so that the CUDA path later replaces
# code we own rather than code we call.
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)

add_library(monoscale_core SHARED
  src/geometry.cpp
  src/anchors.cpp
  src/fusion.cpp
  src/strapdown.cpp
  src/inertial.cpp
  src/attitude.cpp
  src/estimator.cpp
  src/occupancy.cpp
)
target_include_directories(monoscale_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(monoscale_core PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
target_compile_options(monoscale_core PRIVATE -O3)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS monoscale_core
  EXPORT export_monoscale_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
ament_export_targets(export_monoscale_core HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 OpenCV)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_geometry test/test_geometry.cpp)
  target_link_libraries(test_geometry monoscale_core)
  ament_add_gtest(test_anchors test/test_anchors.cpp)
  target_link_libraries(test_anchors monoscale_core)
  ament_add_gtest(test_fusion test/test_fusion.cpp)
  target_link_libraries(test_fusion monoscale_core)
  ament_add_gtest(test_strapdown test/test_strapdown.cpp)
  target_link_libraries(test_strapdown monoscale_core)
  ament_add_gtest(test_inertial test/test_inertial.cpp)
  target_link_libraries(test_inertial monoscale_core)
  ament_add_gtest(test_estimator test/test_estimator.cpp)
  target_link_libraries(test_estimator monoscale_core)
  ament_add_gtest(test_occupancy test/test_occupancy.cpp)
  target_link_libraries(test_occupancy monoscale_core)
  ament_add_gtest(test_attitude test/test_attitude.cpp)
  target_link_libraries(test_attitude monoscale_core)
endif()

ament_package()

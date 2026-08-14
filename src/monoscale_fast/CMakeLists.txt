cmake_minimum_required(VERSION 3.16)
project(monoscale_fast)

# The plane sweep has a CUDA path. Blackwell is compute 12.0 and this toolkit is
# 12.8, which knows that number, so the kernels are built for it directly. An
# earlier machine was Ada on toolkit 11.5 and had to build 8.6 and let the
# driver raise them; if this moves again, the number follows the GPU.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
# pybind11's newer tooling builds on CMake's own Python module, and says so
# only as "Unknown CMake command python3_add_library" if it is missing.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
# The plane sweep is warps and box filters; both are OpenCV's.
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

if(CMAKE_CUDA_COMPILER)
  pybind11_add_module(monoscale_fast src/fast.cpp src/sweep_kernels.cu)
  set_target_properties(monoscale_fast PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
else()
  message(WARNING "no CUDA compiler; building the CPU sweep only")
  pybind11_add_module(monoscale_fast src/fast.cpp src/no_cuda.cpp)
endif()
target_link_libraries(monoscale_fast PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
target_include_directories(monoscale_fast PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_options(monoscale_fast PRIVATE -O3)

# Onto the path the estimator's interpreter already searches. ament sets
# PYTHON_INSTALL_DIR, but only for packages that declare Python of their own,
# and this one is a single extension module with no Python beside it.
if(NOT PYTHON_INSTALL_DIR)
  set(PYTHON_INSTALL_DIR
    "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
endif()
install(TARGETS monoscale_fast
  LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}")

ament_package()

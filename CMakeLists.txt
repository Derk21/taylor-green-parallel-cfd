cmake_minimum_required(VERSION 3.10)
project(TailorGreenVortex LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)


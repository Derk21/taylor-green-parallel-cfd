# Source files
file(GLOB SRC_FILES *.cu)
list(REMOVE_ITEM SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/main.cu)

# Find CUDA libraries
find_package(CUDA REQUIRED)
find_library(CUDA_cusolver_LIBRARY cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Main executable
add_executable(main main.cu ${SRC_FILES})
target_link_libraries(main ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} boost_iostreams boost_system boost_filesystem)

# Enable CUDA separable compilation
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Debug version of the main executable
add_executable(main_debug main.cu ${SRC_FILES})
target_compile_definitions(main_debug PRIVATE DEBUG)
target_compile_options(main_debug PRIVATE -g -O0)
target_link_libraries(main_debug ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} boost_iostreams boost_system boost_filesystem)
set_target_properties(main_debug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
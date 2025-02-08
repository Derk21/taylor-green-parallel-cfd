# Source files
file(GLOB TEST_FILES *.cu)
file(GLOB SRC_FILES ../src/*.cu)
list(REMOVE_ITEM SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../src/main.cu)

# Find CUDA libraries
find_package(CUDA REQUIRED)
find_library(CUDA_cusolver_LIBRARY cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Test executables
foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE} ${SRC_FILES})
    target_link_libraries(${TEST_NAME} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} boost_iostreams boost_system boost_filesystem)
    set_target_properties(${TEST_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Debug version 
    add_executable(${TEST_NAME}_debug ${TEST_FILE} ${SRC_FILES})
    target_compile_definitions(${TEST_NAME}_debug PRIVATE DEBUG)
    target_compile_options(${TEST_NAME}_debug PRIVATE -g -O0 -G -lineinfo  -rdynamic)
    target_link_libraries(${TEST_NAME}_debug ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} boost_iostreams boost_system boost_filesystem)
    set_target_properties(${TEST_NAME}_debug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()
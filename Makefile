# Compiler settings
CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc
CXX = clang
CXXFLAGS = -O2 -std=c++17

# Include directories
INCLUDE_DIRS = -I$(CUDA_HOME)/include -Iinclude

# Libraries
LIBS = -L$(CUDA_HOME)/lib64 -lcudart -lcuda -lboost_iostreams -lboost_system -lboost_filesystem -lcusolver -lcusparse

# Flags for compiling
NVCC_FLAGS = -lineinfo -Xcompiler "$(CXXFLAGS)"
DEBUG_FLAGS = -g -G -O0 -Xcompiler -std=c++17

# Source and build directories
SRC_DIR = src
BUILD_DIR = build
TEST_DIR = tests

# Source files
SRC = $(SRC_DIR)/main.cu $(SRC_DIR)/init.cu $(SRC_DIR)/pressure_correction.cu $(SRC_DIR)/solve.cu $(SRC_DIR)/utils.cu $(SRC_DIR)/plotting.cu $(SRC_DIR)/advect.cu $(SRC_DIR)/diffuse.cu 
TEST_SRC = $(TEST_DIR)/test_pressure_correction.cu $(TEST_DIR)/test_solve.cu

# Executable names
EXEC = $(BUILD_DIR)/main
DEBUG_EXEC = $(BUILD_DIR)/main_debug
TEST_EXEC = $(BUILD_DIR)/test_pressure_correction $(BUILD_DIR)/test_solve
DEBUG_TEST_EXEC = $(BUILD_DIR)/test_pressure_correction_debug $(BUILD_DIR)/test_solve_debug

# Target to build the executable
all: $(EXEC) $(DEBUG_EXEC) $(TEST_EXEC) $(DEBUG_TEST_EXEC)

# Rule for compiling the CUDA source files
$(EXEC): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $@ $(LIBS) $(INCLUDE_DIRS)

$(DEBUG_EXEC): $(SRC)
	$(NVCC) $(DEBUG_FLAGS) $(SRC) -o $@ $(LIBS) $(INCLUDE_DIRS)

# Rules for compiling the test files
$(BUILD_DIR)/test_pressure_correction: $(TEST_DIR)/test_pressure_correction.cu $(SRC_DIR)/pressure_correction.cu $(SRC_DIR)/utils.cu $(SRC_DIR)/solve.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS) $(INCLUDE_DIRS)

$(BUILD_DIR)/test_solve: $(TEST_DIR)/test_solve.cu $(SRC_DIR)/solve.cu $(SRC_DIR)/utils.cu 
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS) $(INCLUDE_DIRS)

# Rules for compiling the test files with debug flags
$(BUILD_DIR)/test_pressure_correction_debug: $(TEST_DIR)/test_pressure_correction.cu $(SRC_DIR)/pressure_correction.cu $(SRC_DIR)/utils.cu $(SRC_DIR)/solve.cu
	$(NVCC) $(DEBUG_FLAGS) $^ -o $@ $(LIBS) $(INCLUDE_DIRS)

$(BUILD_DIR)/test_solve_debug: $(TEST_DIR)/test_solve.cu $(SRC_DIR)/solve.cu $(SRC_DIR)/utils.cu
	$(NVCC) $(DEBUG_FLAGS) $^ -o $@ $(LIBS) $(INCLUDE_DIRS)

# Clean target to remove object files and executable
clean:
	rm -f $(BUILD_DIR)/*

# Run the program
run: $(EXEC)
	./$(EXEC)

# Run the tests
test: $(TEST_EXEC)
	./$(BUILD_DIR)/test_pressure_correction
	./$(BUILD_DIR)/test_solve

# Run the debug tests
debug_test: $(DEBUG_TEST_EXEC)
	./$(BUILD_DIR)/test_pressure_correction_debug
	./$(BUILD_DIR)/test_solve_debug

# Phony targets (not actual files)
.PHONY: all clean run test debug_test
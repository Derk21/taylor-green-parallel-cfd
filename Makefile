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
NVCC_FLAGS = -dc -lineinfo -Xcompiler "$(CXXFLAGS)"
DEBUG_FLAGS = -dc -g -G -O0 -Xcompiler -std=c++17

# Source and build directories
SRC_DIR = src
BUILD_DIR = build
TEST_DIR = tests

# Source files
SRC = $(filter-out $(SRC_DIR)/main.cu, $(wildcard $(SRC_DIR)/*.cu))
MAIN_SRC = $(SRC_DIR)/main.cu
TEST_SRC = $(wildcard $(TEST_DIR)/*.cu)

# Object files
OBJ = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRC))
MAIN_OBJ = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(MAIN_SRC))
TEST_OBJ = $(patsubst $(TEST_DIR)/%.cu, $(BUILD_DIR)/%.o, $(TEST_SRC))

# Executable names
EXEC = $(BUILD_DIR)/main
DEBUG_EXEC = $(BUILD_DIR)/main_debug
TEST_EXEC = $(patsubst $(TEST_DIR)/%.cu, $(BUILD_DIR)/%, $(TEST_SRC))
DEBUG_TEST_EXEC = $(patsubst $(TEST_DIR)/%.cu, $(BUILD_DIR)/%_debug, $(TEST_SRC))

# Target to build the executable
all: $(EXEC) $(DEBUG_EXEC)

# Rule for compiling CUDA source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Rule for linking the object files to create the executable
$(EXEC): $(MAIN_OBJ) $(OBJ)
	$(NVCC) $(MAIN_OBJ) $(OBJ) -o $@ $(LIBS) $(INCLUDE_DIRS)

$(DEBUG_EXEC): $(MAIN_OBJ) $(OBJ)
	$(NVCC) $(DEBUG_FLAGS) $(MAIN_OBJ) $(OBJ) -o $@ $(LIBS) $(INCLUDE_DIRS)

# Rules for compiling and linking the test files
$(BUILD_DIR)/%: $(BUILD_DIR)/%.o $(OBJ)
	$(NVCC) $(OBJ) $< -o $@ $(LIBS) $(INCLUDE_DIRS)

$(BUILD_DIR)/%_debug: $(BUILD_DIR)/%.o $(OBJ)
	$(NVCC) $(DEBUG_FLAGS) $(OBJ) $< -o $@ $(LIBS) $(INCLUDE_DIRS)

test: $(TEST_EXEC) $(DEBUG_TEST_EXEC)

clean:
	rm -f $(BUILD_DIR)/*

run: $(EXEC)
	./$(EXEC)

testrun: $(TEST_EXEC)
	@for test in $(TEST_EXEC); do ./$$test; done

# Phony targets (not actual files)
.PHONY: all clean run test testrun debug_test
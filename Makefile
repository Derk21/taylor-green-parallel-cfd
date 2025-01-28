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
SRC = $(filter-out $(SRC_DIR)/main.cu, $(wildcard $(SRC_DIR)/*.cu))
MAIN_SRC = $(SRC_DIR)/main.cu
TEST_SRC = $(wildcard $(TEST_DIR)/*.cu)

# Executable names
EXEC = $(BUILD_DIR)/main
DEBUG_EXEC = $(BUILD_DIR)/main_debug
TEST_EXEC = $(patsubst $(TEST_DIR)/%.cu, $(BUILD_DIR)/%, $(TEST_SRC))
DEBUG_TEST_EXEC = $(patsubst $(TEST_DIR)/%.cu, $(BUILD_DIR)/%_debug, $(TEST_SRC))

# Target to build the executable
all: $(EXEC) $(DEBUG_EXEC)

# Rule for compiling the CUDA source files
$(EXEC): $(MAIN_SRC) $(SRC)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS) $(INCLUDE_DIRS)

$(DEBUG_EXEC): $(MAIN_SRC) $(SRC)
	$(NVCC) $(DEBUG_FLAGS) $^ -o $@ $(LIBS) $(INCLUDE_DIRS)

# Rules for compiling the test files
$(BUILD_DIR)/%: $(TEST_DIR)/%.cu $(SRC)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS) $(INCLUDE_DIRS)

$(BUILD_DIR)/%_debug: $(TEST_DIR)/%.cu $(SRC)
	$(NVCC) $(DEBUG_FLAGS) $^ -o $@ $(LIBS) $(INCLUDE_DIRS)

test: $(TEST_EXEC) $(DEBUG_TEST_EXEC)

clean:
	rm -f $(BUILD_DIR)/*

run: $(EXEC)
	./$(EXEC)

testrun: $(TEST_EXEC)
	@for test in $(TEST_EXEC); do ./$$test; done

# Phony targets (not actual files)
.PHONY: all clean run test testrun debug_test
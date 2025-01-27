# Compiler settings
CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc
CXX = clang
CXXFLAGS = -O2 -std=c++17

# Include directories
INCLUDE_DIRS = -I$(CUDA_HOME)/include 

# Libraries
LIBS = -L$(CUDA_HOME)/lib64 -lcudart -lcuda -lboost_iostreams -lboost_system -lboost_filesystem -lcusolver -lcusparse

# Flags for compiling
# NVCC_FLAGS = -arch=sm_80 -lineinfo -Xcompiler $(CXXFLAGS)
NVCC_FLAGS = -lineinfo -Xcompiler $(CXXFLAGS)
DEBUG_FLAGS = -g -G -O0 -Xcompiler -std=c++17

# Target to build the executable
all: $(EXEC) clean main debug

# Rule for compiling the CUDA source file
$(EXEC): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(EXEC) $(LIBS)
main: main.cu
	$(NVCC) $(NVCC_FLAGS) main.cu -o main $(LIBS)

debug: main.cu
	$(NVCC) $(DEBUG_FLAGS) -g main.cu -o main_debug $(LIBS)

# Clean target to remove object files and executable
clean:
	rm -f $(EXEC) main

# Run the program
run: $(EXEC)
	./$(EXEC)

# Phony targets (not actual files)
.PHONY: all clean run
#ifndef UTILS_H
#define UTILS_H

#include "constants.cuh"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                               \
    if ((call) != cudaSuccess)                                         \
    {                                                                  \
        std::cerr << "CUDA error at " << __LINE__ << ":" << std::endl; \
        std::cerr << (cudaGetErrorString(call)) << std::endl;          \
        exit(EXIT_FAILURE);                                            \
    }

//copied from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/utils/cusolver_utils.h
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

//adapted from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/sparse2dense_csr/sparse2dense_csr_example.c
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(EXIT_FAILURE);                                                   \
    }                                                                          \
}

void print_matrix(const int &m, const int &n, const double *A, const int &lda);

void print_matrix_row_major(const int &m, const int &n, const double *A, const int &lda); 

void printCSR(const int* row_offsets, const int* col_indices, const double* values, int n);

__host__ __device__ size_t periodic_linear_Idx(const int &x, const int &y, const int bound_x = 2*NUM_N, const int bound_y = M);

__host__ __device__ void setClosestGridPointIdx(double x, double y, int n, int m, int &closest_x_i, int &closest_y_i);

__host__ __device__ bool is_close(const double &a, const double &b, const double &tolerance = 1e-6);

bool all_close(const double * a,const double* b,int n, int m);

__host__ void clip(double &v,const double min, const double max);

namespace gpu {
__device__ void clip(double &v,const double min, const double max);
}

void switchRowColMajor(double *A_rowMajor, const int &m, const int &n);

std::vector<double> readDataFile(const std::string& file_path);

double calculateRMSE(const std::vector<double>& reference, const std::vector<double>& simulation);

double calculateRelativeErr(const std::vector<double>& reference, double rmse);

void taylorGreenGroundTruth(double* periodic_grid,double *velocity_grid_next, int iteration, int n=NUM_N , int m=M);

#endif // UTILS_H
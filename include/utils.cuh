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

// copied from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/utils/cusolver_utils.h
#define CUSOLVER_CHECK(err)                                                   \
    do                                                                        \
    {                                                                         \
        cusolverStatus_t err_ = (err);                                        \
        if (err_ != CUSOLVER_STATUS_SUCCESS)                                  \
        {                                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cusolver error");                       \
        }                                                                     \
    } while (0)

// adapted from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/sparse2dense_csr/sparse2dense_csr_example.c
#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

/**
 * @brief print matrix col major 
 * // copied from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/utils/cusolver_utils.h
 * @param m display rows
 * @param n display cols
 * @param A matrix
 * @param lda number of rows of matrix
 */
void print_matrix(const int &m, const int &n, const double *A, const int &lda);

/**
 * @brief print matrix row major 
 * 
 * @param m display rows
 * @param n display cols
 * @param A matrix
 * @param lda number of cols of matrix
 */
void print_matrix_row_major(const int &m, const int &n, const double *A, const int &lda);

/**
 * @brief print sparse matrix 
 * 
 * @param row_offsets 
 * @param col_indices 
 * @param values 
 * @param n 
 */
void printCSR(const int *row_offsets, const int *col_indices, const double *values, int n);

/**
 * @brief returns linear index of given 2D index, while also ensuring periodic boundary
 * 
 * ensures periodic boundary condition
 * 
 * @param x 
 * @param y 
 * @param bound_x 
 * @param bound_y 
 * @return __host__ 
 */
__host__ __device__ size_t periodic_linear_Idx(const int &x, const int &y, const int bound_x = 2 * NUM_N, const int bound_y = M);

/**
 * @brief Set the Closest Grid Point Idx of a given point
 * 
 * @param x input x
 * @param y input y
 * @param n grid dim x
 * @param m grid dim y 
 * @param closest_x_i output x index
 * @param closest_y_i output y index
 */
__host__ __device__ void setClosestGridPointIdx(double x, double y, int n, int m, int &closest_x_i, int &closest_y_i);

/**
 * @brief returns if absolute difference between given values is within tolerance
 * 
 * @param a value
 * @param b value
 * @param tolerance 
 * @return bool
 */
__host__ __device__ bool is_close(const double &a, const double &b, const double &tolerance = 1e-6);

/**
 * @brief returns if all matrix values are similiar
 * 
 * @param a matrix 
 * @param b matrix
 * @param n grid dim x
 * @param m grid dim y
 * @return bool
 */
bool all_close(const double *a, const double *b, int n, int m);

/**
 * @brief caps values in interval 
 * 
 * @param v input and output val
 * @param min 
 * @param max 
 */
__host__ void clip(double &v, const double min, const double max);

namespace gpu
{
    /**
     * @brief caps values in interval
     * 
     * @param v input and output value
     * @param min 
     * @param max 
     */
    __device__ void clip(double &v, const double min, const double max);
}

/**
 * @brief switches matrix between row and col major
 * 
 * transposes matrix
 * 
 * @param A 
 * @param m 
 * @param n 
 */
void switchRowColMajor(double *A, const int &m, const int &n);

/**
 * @brief calculate Root mean square error of simulation
 * 
 * @param reference reference vector of values
 * @param simulation simulation vector of values
 * @return double 
 */
double calculateRMSE(const std::vector<double> &reference, const std::vector<double> &simulation);

/**
 * @brief get taylor green ground truth velocity field at iteration
 * 
 * @param periodic_grid 
 * @param velocity_grid_next 
 * @param iteration 
 * @param n 
 * @param m 
 */
void taylorGreenGroundTruth(double *periodic_grid, double *velocity_grid_next, int iteration, int n = NUM_N, int m = M);

#endif // UTILS_H
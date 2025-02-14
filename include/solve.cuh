#pragma once
#ifndef SOLVE_H
#define SOLVE_H

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <vector>
#include "constants.cuh"
#include "utils.cuh"

/**
 * @brief Solves a dense linear system A * X = B.
 *
 * @param A Pointer to the matrix A. (cpu memory)
 * @param B Pointer to the matrix B. (cpu memory)
 * @param X Pointer to the solution matrix X. (cpu memory)
 * @param m Number of rows/columns in the matrices
 */
void solveDense(const double *A, const double *B, double *X, size_t m = 2 * NUM_N);

namespace gpu
{
    /**
     * @brief Solves a dense linear system on the GPU. d_A * X = d_B, output X is stored ind d_B
     *
     * @param d_A Pointer to the matrix A on the device.
     * @param d_B Pointer to the matrix B on the device. (also output)
     * @param m Number of rows/columns in the matrices.
     */
    void solveDense(double *d_A, double *d_B, size_t m);

    /**
     * @brief Solves a sparse linear system with cholesky solver on the GPU.
     *
     * @param dA_values Pointer to the non-zero values of the sparse matrix A on the device.
     * @param dA_columns Pointer to the column indices of the sparse matrix A on the device.
     * @param dA_csrOffsets Pointer to the row offsets of the sparse matrix A in CSR format on the device.
     * @param d_divergence Pointer to the divergence vector on the device.
     * @param d_pressure Pointer to the pressure vector on the device.
     * @param A_nnz Number of non-zero elements in the sparse matrix A (default is 5 * NUM_N * NUM_N).
     * @param m Number of rows/columns in the matrices (default is 2 * NUM_N).
     */
    void solveSparse(double *dA_values, int *dA_columns, int *dA_csrOffsets, double *d_divergence, double *d_pressure, int A_nnz = 5 * NUM_N * NUM_N, const int m = 2 * NUM_N);
}
#endif // SOLVE_H
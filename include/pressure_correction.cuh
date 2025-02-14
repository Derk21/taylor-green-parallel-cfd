#pragma once
#ifndef PRESSURE_CORRECTION_H
#define PRESSURE_CORRECTION_H

#include "constants.cuh"
#include "utils.cuh"
#include "solve.cuh"

/**
 * @brief Corrects the velocity by solving poission equation with dense cuSolver  
 * 
 * This function performs pressure correction on a given velocity grid to ensure
 * incompressibility by solving the Poisson equation for pressure. The correction
 * is based on the divergence of the velocity field and the discrete Laplace matrix.
 * 
 * @param velocity_grid The velocity grid to be corrected.
 * @param divergence The divergence of the velocity field.
 * @param pressure The pressure field to be updated.
 * @param laplace The discrete Laplace matrix 
 * @param n Number of grid points in the x-direction.
 * @param m Number of grid points in the y-direction.
 * @param dx The grid spacing.
 */
void makeIncompressible(double *velocity_grid, double *divergence, double *pressure, double *laplace, int n = NUM_N, int m = M, const double dx = DX);

/**
 * @brief Corrects the velocity field based on the pressure field.
 * 
 * @param velocity_grid Pointer to the velocity grid.
 * @param pressure Pointer to the pressure field.
 * @param n Number of grid points in the x-direction.
 * @param m Number of grid points in the y-direction.
 * @param dx Grid spacing.
 */
void correct_velocity(double *velocity_grid, double *pressure, int n, int m, double dx);

/**
 * @brief Calculates the divergence of the velocity field 
 * 
 * @param velocity_grid Pointer to the velocity grid.
 * @param divergence Pointer to the divergence field.
 * @param n Number of grid points in the x-direction  
 * @param m Number of grid points in the y-direction 
 * @param dx Grid spacing 
 */
void calculateDivergence(const double *velocity_grid, double *divergence, int n = NUM_N, int m = M, const double dx = DX);

/**
 * @brief Constructs the discretized Laplacian matrix.
 * 
 * @param laplace_discrete Pointer to the discretized Laplacian matrix.
 * @param n Number of grid points in the x-direction 
 * @param dx Grid spacing 
 */
void constructDiscretizedLaplacian(double *laplace_discrete, int n = NUM_N, const double dx = DX);

/**
 * @brief Constructs the sparse CSR representation of the Laplacian matrix.
 * 
 * @param values Pointer to the values array of the CSR matrix.
 * @param row_offsets Pointer to the row offsets array of the CSR matrix.
 * @param col_indices Pointer to the column indices array of the CSR matrix.
 * @param n Number of grid points in the x-direction
 * @param dx Grid spacing
 */
void constructLaplaceSparseCSR(double *values, int *row_offsets, int *col_indices, const int n = NUM_N, const double dx = DX);

namespace gpu
{
    /**
     * @brief CUDA kernel to construct the sparse CSR representation of the Laplacian matrix.
     * 
     * @param values Pointer to the values array of the CSR matrix.
     * @param row_offsets Pointer to the row offsets array of the CSR matrix.
     * @param col_indices Pointer to the column indices array of the CSR matrix.
     * @param n Number of grid points in the x-direction
     * @param dx Grid spacing
     */
    __global__ void constructLaplaceSparseCSR(double *values, int *row_offsets, int *col_indices, const int n = NUM_N, const double dx = DX);

    /**
     * @brief Corrects the velocity by solving poission equation with sparse cholesky solver 
     * 
     * @param velocity_grid Pointer to the velocity grid.
     * @param divergence Pointer to the divergence field.
     * @param pressure Pointer to the pressure field.
     * @param lp_values Pointer to the values array of the Laplacian CSR matrix.
     * @param lp_columns Pointer to the column indices array of the Laplacian CSR matrix.
     * @param lp_row_offsets Pointer to the row offsets array of the Laplacian CSR matrix.
     * @param n Number of grid points in the x-direction 
     * @param m Number of grid points in the y-direction 
     * @param dx Grid spacing
     */
    void makeIncompressibleSparse(double *velocity_grid, double *divergence, double *pressure, double *lp_values, int *lp_columns, int *lp_row_offsets, int n = NUM_N, int m = M, double dx = DX);

    /**
     * @brief Corrects velocity by solving poisson equation with dense solver
     * 
     * @param velocity_grid Pointer to the velocity grid.
     * @param d_B Pointer to the right-hand side vector.
     * @param laplace Pointer to the Laplacian matrix.
     * @param n Number of grid points in the x-direction 
     * @param m Number of grid points in the y-direction 
     * @param dx Grid spacing
     */
    void makeIncompressible(double *velocity_grid, double *d_B, double *laplace, int n = NUM_N, int m = M, const double dx = DX);

    /**
     * @brief CUDA kernel to correct the velocity field based on the pressure field.
     * 
     * @param velocity_grid Pointer to the velocity grid.
     * @param pressure Pointer to the pressure field.
     * @param n Number of grid points in the x-direction.
     * @param m Number of grid points in the y-direction.
     * @param dx Grid spacing
     */
    __global__ void correct_velocity(double *velocity_grid, double *pressure, int n, int m, const double dx = DX);

    /**
     * @brief CUDA kernel to fill the values of the discretized Laplacian matrix.
     * 
     * @param laplace_discrete Pointer to the discretized Laplacian matrix.
     * @param n Number of grid points in the x-direction 
     * @param dx Grid spacing 
     */
    __global__ void fillLaplaceValues(double *laplace_discrete, int n = NUM_N, const double dx = DX);

    /**
     * @brief Constructs the discretized Laplacian matrix.
     * 
     * @param laplace_discrete Pointer to the discretized Laplacian matrix.
     * @param n Number of grid points in the x-direction 
     * @param dx Grid spacing 
     */
    void constructDiscretizedLaplacian(double *laplace_discrete, int n = NUM_N, const double dx = DX);

    /**
     * @brief CUDA kernel to calculate the divergence of the velocity field.
     * 
     * @param velocity_grid Pointer to the velocity grid.
     * @param divergence Pointer to the divergence field.
     * @param n Number of grid points in the x-direction 
     * @param m Number of grid points in the y-direction 
     * @param dx Grid spacing 
     */
    __global__ void calculateDivergence(const double *velocity_grid, double *divergence, int n = NUM_N, int m = M, const double dx = DX);
}

#endif // PRESSURE_CORRECTION_H
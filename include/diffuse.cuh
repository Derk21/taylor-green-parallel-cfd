#pragma once
#ifndef DIFFUSE_H
#define DIFFUSE_H
#include "constants.cuh"
#include "utils.cuh"

/**
 * @brief performs single diffusion step
 * 
 * @param velocity_grid  diffusion input
 * @param velocity_grid_next diffusion output
 * @param amount stepsize
 * @param n grid dim
 * @param m grid dim
 * @param dx grid spacing 
 */
void diffuseExplicitStep(const double *velocity_grid, double *velocity_grid_next, const double amount, const int n = NUM_N, const int m = M, const double dx = DX);

/**
 * @brief performs explicit diffusion (iterative method)
 * 
 * @param velocity_grid  diffusion input
 * @param velocity_grid_next diffusion output
 * @param n grid dim
 * @param m grid dim
 * @param dx grid spacing 
 */
void diffuseExplicit(double *velocity_grid, double *velocity_grid_next, const int n = NUM_N, const int m = M, const double dx = DX);

namespace gpu
{
    /**
     * @brief Explicit diffusion kernel
     * 
     * @param velocity_grid diffusion target
     * @param amount stepsize
     * @param n grid dim
     * @param m grid dim
     * @param dx grid spacing
     * @return __global__ 
     */
    __global__ void diffuseExplicitStep(double *velocity_grid, const double amount, const int n = NUM_N, const int m = M, const double dx = DX);

    /**
    * @brief performs explicit diffusion (iterative method) on gpu
    * 
    * @param velocity_grid  diffusion target
    * @param n grid dim
    * @param m grid dim
    * @param dx grid spacing 
    */
    void diffuseExplicit(double *velocity_grid, const int n = NUM_N, const int m = M, const double dx = DX);

}
#endif // DIFFUSE_H
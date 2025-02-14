#pragma once
#ifndef ADVECT_H
#define ADVECT_H
#include "constants.cuh"
#include "utils.cuh"
#include <cmath>

/**
 * @brief bilinear interpolates single velocity vector
 * 
 * @param u 
 * @param v 
 * @param x_d integrated destination point
 * @param y_d integrated destination point
 * @param periodic_grid 
 * @param velocity_grid 
 * @param n 
 * @param m 
 * @param dx 
 * @return __host__ 
 */
__host__ __device__ void interpolateVelocity(double &u, double &v, double x_d, double y_d, const double *periodic_grid, const double *velocity_grid, const int n = NUM_N, int m = M, const double dx = DX);

/**
 * @brief Get the interpolated object
 * 
 * @param i_closest closest grid point x index 
 * @param y_i_closest closest grid point y index
 * @param x_diff distance from closest 
 * @param y_diff distance form closest
 * @param velocity_grid 
 * @param n 
 * @param m 
 * @return __host__ 
 */
__host__ __device__ double get_interpolated(const int &i_closest, const int &y_i_closest, const double &x_diff, const double &y_diff, const double *velocity_grid, int n = NUM_N, int m = M);

/**
 * @brief euler integrate single velocity vector
 * 
 * @param velocity_grid 
 * @param y_i input index
 * @param u_i input index 
 * @param v_i input index
 * @param periodic_grid 
 * @param x_d integrated destination
 * @param y_d integrated destination
 * @param dt timestep 
 * @param n 
 * @param m 
 * @return __host__ 
 */
__host__ __device__ void integrateEuler(const double *velocity_grid, int &y_i, int &u_i, int &v_i, const double *periodic_grid, double &x_d, double &y_d, const double dt, int n = NUM_N, int m = M);

/**
 * @brief advects velocity with semi-lagrange method, result is written to velocity_grid
 * 
 * @param velocity_grid input (and output)
 * @param velocity_grid_next temporary output
 * @param periodic_grid 
 * @param dt timestep
 * @param n 
 * @param m 
 * @param dx 
 */
void advectSemiLagrange(double *velocity_grid, double *velocity_grid_next, const double *periodic_grid, const double dt, int n = NUM_N, int m = M, const double dx = DX);

/**
 * @brief calculates extrema in 4 neighboring points of stencil of that velocity component
 * 
 * @param min output 
 * @param max output
 * @param idx 
 * @param y_i 
 * @param velocity_grid 
 * @param n 
 * @param m 
 */
void min_max_neighbors(double &min, double &max, const int idx, const int y_i, const double *velocity_grid, const int n = NUM_N, const int m = M);

/**
 * @brief corrects the prediction of backward advected single velocity component with advection in the original direction 
 * 
 * @param idx_x x idx of velocity component
 * @param y_i y idx of velocity component
 * @param velocity_grid 
 * @param velocity_grid_bw backward advected velocity
 * @param velocity_grid_fw backward-forward advected velocity
 * @param n 
 * @param m 
 * @return double 
 */
double mac_cormack_correction(const int idx_x, const int y_i, const double *velocity_grid, const double *velocity_grid_bw, const double *velocity_grid_fw, int n = NUM_N, int m = M);

/**
 * @brief advection with maccormack method, involving a prediction and correction step
 * 
 * @param velocity_grid input and output of advected velocity
 * @param velocity_grid_next temporary output velocity
 * @param periodic_grid 
 * @param dt timestep
 * @param n 
 * @param m 
 * @param dx grid spacing
 */
void advectMacCormack(double *velocity_grid, double *velocity_grid_next, const double *periodic_grid, const double dt, const int n = NUM_N, const int m = M, const double dx = DX);

namespace gpu
{
    /**
     * @brief advects velocity with semi-lagrange method on the GPU, result is written to velocity_grid
     * 
     * @param velocity_grid input (and output)
     * @param velocity_grid_backward temporary output for backward integration
     * @param periodic_grid grid with periodic boundary conditions
     * @param dt timestep
     * @param n 
     * @param m 
     * @param dx grid spacing
     */
    void advectSemiLagrange(
        double *velocity_grid,
        double *velocity_grid_backward,
        const double *periodic_grid,
        const double dt = TIMESTEP, int n = NUM_N, int m = M, double dx = DX);

    /**
     * @brief advects velocity with semi-lagrange method on the GPU, result is written to velocity_grid
     *
     * separate integration and interpolation kernel for debugging, is slower than combined kernel
     * 
     * @param velocity_grid input (and output)
     * @param velocity_grid_next temporary output for next velocity grid
     * @param integrated_backward temporary output for backward integration
     * @param periodic_grid grid with periodic boundary conditions
     * @param dt timestep
     * @param n number of grid points in x direction
     * @param m number of grid points in y direction
     * @param dx grid spacing
     */
    void advectSemiLagrangeSeparate(
        double *velocity_grid,
        double *velocity_grid_next,
        double *integrated_backward,
        const double *periodic_grid,
        const double dt = TIMESTEP, int n = NUM_N, int m = M, double dx = DX);
    /**
    /**
     * @brief advects velocity with MacCormack method on the GPU, involving a prediction and correction step
     * 
     * @param velocity_grid input and output of advected velocity
     * @param velocity_grid_backward temporary output for backward integration
     * @param velocity_grid_forward temporary output for forward integration
     * @param integrated_fw temporary output for forward integrated velocity
     * @param integrated_bw temporary output for backward integrated velocity
     * @param periodic_grid 
     * @param dt timestep
     * @param n 
     * @param m 
     * @param dx grid spacing
     */
    void advectMacCormack(
        double *velocity_grid,
        double *velocity_grid_backward,
        double *velocity_grid_forward,
        double *integrated_fw,
        double *integrated_bw,
        const double *periodic_grid,
        const double dt = TIMESTEP, int n = NUM_N, int m = M, const double dx = DX);

    /**
     * @brief euler integrates whole velocity grid on gpu
     * 
     * @param periodic_grid 
     * @param velocity_grid input velocities
     * @param integrated output destinations
     * @param dt timestep
     * @param n 
     * @param m 
     * @return __global__ 
     */
    __global__ void integrateKernel(const double *periodic_grid, const double *velocity_grid, double *integrated, const double dt, const int n = NUM_N, const int m = M);

    /**
    * @brief bilinear interpolates whole velocity grid on gpu
    * 
    * @param periodic_grid grid with periodic boundary conditions
    * @param velocity_grid input velocities
    * @param velocity_grid_next output velocities after interpolation
    * @param integrated integrated positions
    * @param n number of grid points in x direction
    * @param m number of grid points in y direction
    * @param dx grid spacing
    * @return __global__
     */
    __global__ void interpolateKernel(const double *periodic_grid, const double *velocity_grid, double *velocity_grid_next, double *integrated, const int n = NUM_N, const int m = M, const double dx = DX);

    /**
     * @brief combined euler integration and interpolation kernel
     * 
     * less memory demanding and faster than separate integation and interpolation
     * 
     * @param periodic_grid 
     * @param velocity_grid 
     * @param velocity_grid_next 
     * @param dt 
     * @param n 
     * @param m 
     * @param dx 
     * @return __global__ 
     */
    __global__ void integrateAndInterpolateKernel(const double *periodic_grid, const double *velocity_grid, double *velocity_grid_next, const double dt, const int n = NUM_N, const int m = M, const double dx = DX);

    /**
    * @brief kernel for MacCormack correction step on the GPU
    * 
    * @param velocity_grid input and output of corrected velocity
    * @param velocity_grid_bw backward integrated velocity
    * @param velocity_grid_fw forward integrated velocity
    * @param n number of grid points in x direction
    * @param m number of grid points in y direction
    * @return __global__ 
     */
    __global__ void macCormackCorrectionKernel(double *velocity_grid, const double *velocity_grid_bw, const double *velocity_grid_fw, int n = NUM_N, int m = M);

    /**
     * @brief device function for correction of single velocity prediction
     * 
     * @param idx_x 
     * @param y_i 
     * @param velocity_grid 
     * @param velocity_grid_bw 
     * @param velocity_grid_fw 
     * @param n 
     * @param m 
     * @return __device__ 
     */
    __device__ double mac_cormack_correction(const int idx_x, const int y_i, const double *velocity_grid, const double *velocity_grid_bw, const double *velocity_grid_fw, int n = NUM_N, int m = M);

    /**
    * @brief calculates extrema in 4 neighboring points of stencil of that velocity component 
    * 
    * TODO: do some reduction on gpu instead
    * 
    * @param min output minimum value
    * @param max output maximum value
    * @param idx x index of the velocity component
    * @param y_i y index of the velocity component
    * @param velocity_grid input velocity grid
    * @param n number of grid points in x direction
    * @param m number of grid points in y direction
    * @return __device__ 
     */
    __device__ void min_max_neighbors(double &min, double &max, const int idx, const int y_i, const double *velocity_grid, const int n = NUM_N, const int m = M);

}

#endif // ADVECT_H
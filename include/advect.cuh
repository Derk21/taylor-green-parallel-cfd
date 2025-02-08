#pragma once
#ifndef ADVECT_H
#define ADVECT_H
#include "constants.cuh"
#include "utils.cuh"
#include <cmath>

__host__ __device__ void interpolateVelocity(double &u, double &v,double x_d, double y_d,  const double *periodic_grid, const double *velocity_grid, const int n = NUM_N, int m=M, const double dx = DX);

__host__ __device__ double get_interpolated(const int &i_closest, const int & y_i_closest,const double &x_diff, const double &y_diff,const double * velocity_grid,int n=NUM_N, int m=M);

__host__ __device__ void integrateEuler(const double *velocity_grid, int &y_i, int &u_i, int &v_i, const double *periodic_grid, double &x_d,  double &y_d, const double dt, int n=NUM_N, int m=M);

void advectSemiLagrange(double *velocity_grid, double *velocity_grid_next, const double *periodic_grid, const double dt, int n=NUM_N, int m=M, const double dx=DX);

void min_max_neighbors(double &min, double &max, const int idx,const int y_i, const double * velocity_grid,const int n=NUM_N, const int m=M);

double mac_cormack_correction(const int idx_x,const int y_i,const double * velocity_grid, const double * velocity_grid_bw, const double* velocity_grid_fw, int n=NUM_N, int m=M);

void advectMacCormack(double *velocity_grid, double *velocity_grid_next, const double *periodic_grid, const double dt, const int n=NUM_N, const int m=M, const double dx=DX);

namespace gpu
{

void advectSemiLagrange(
    double *velocity_grid,
    double *velocity_grid_backward, 
    const double *periodic_grid, 
    const double dt=TIMESTEP, int n=NUM_N, int m=M,double dx=DX);

void advectMacCormack(
    double *velocity_grid,
    double *velocity_grid_backward, 
    double *velocity_grid_forward, 
    const double *periodic_grid, 
    const double dt=TIMESTEP, int n=NUM_N, int m=M,const double dx=DX);

__global__ void integrateAndInterpolateKernel(const double *periodic_grid, const double *velocity_grid, double * velocity_grid_next,const double dt,const int n=NUM_N, const int m=M,const double dx=DX);

__global__ void macCormackCorrectionKernel(double * velocity_grid, const double * velocity_grid_bw, const double* velocity_grid_fw,  int n=NUM_N, int m=M);

__device__ double mac_cormack_correction(const int idx_x,const int y_i,const double * velocity_grid, const double * velocity_grid_bw, const double* velocity_grid_fw,  int n=NUM_N, int m=M);

__device__ void min_max_neighbors(double &min, double &max, const int idx,const int y_i, const double * velocity_grid,const int n=NUM_N, const int m=M);

}

#endif // ADVECT_H
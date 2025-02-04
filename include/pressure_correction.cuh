#pragma once
#ifndef PRESSURE_CORRECTION_H
#define PRESSURE_CORRECTION_H

#include "constants.cuh"
#include "utils.cuh"
#include "solve.cuh"

void makeIncompressible(double* velocity_grid, double* divergence, double* pressure, int n = NUM_N, int m = M, const double dx = DX);

void correct_velocity(double * velocity_grid,double * pressure,int n, int m, double dx);

void calculateDivergence(const double* velocity_grid, double* divergence, int n = NUM_N, int m = M,const double dx = DX);

void constructDiscretizedLaplacian(double* laplace_discrete, int n = NUM_N, const double dx = DX);

namespace gpu{

void makeIncompressible(double* velocity_grid, double* d_B, double* laplace, int n=NUM_N, int m=M);

__global__ void correct_velocity(double * velocity_grid,double * pressure,int n, int m, const double dx=DX);

__global__ void fillLaplaceValues(double* laplace_discrete,int n=NUM_N,const double dx=DX);

void constructDiscretizedLaplacian(double* laplace_discrete,int n, const double dx);

__global__ void calculateDivergence(const double* velocity_grid,double*divergence,int n=NUM_N, int m=M, const double dx=DX);
}
    
#endif // PRESSURE_CORRECTION_H
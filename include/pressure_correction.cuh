#pragma once
#ifndef PRESSURE_CORRECTION_H
#define PRESSURE_CORRECTION_H

#include "constants.cuh"
#include "utils.cuh"
#include "solve.cuh"

void makeIncompressible(double* velocity_grid, double* divergence, double* pressure,double* laplace, int n = NUM_N, int m = M, const double dx = DX);

void correct_velocity(double * velocity_grid,double * pressure,int n, int m, double dx);

void calculateDivergence(const double* velocity_grid, double* divergence, int n = NUM_N, int m = M,const double dx = DX);

void constructDiscretizedLaplacian(double* laplace_discrete, int n = NUM_N, const double dx = DX);

void constructLaplaceSparseCSR(double* values, int* row_offsets,int* col_indices,const int n=NUM_N, const double dx=DX);

namespace gpu{
__global__ void constructLaplaceSparseCSR(double* values, int* row_offsets,int* col_indices,const int n=NUM_N, const double dx=DX);

void makeIncompressibleSparse(double* velocity_grid, double* divergence, double* pressure, double* lp_values,int* lp_columns,int* lp_row_offsets, int n=NUM_N, int m=M,double dx=DX);

void makeIncompressible(double* velocity_grid, double* d_B, double* laplace, int n=NUM_N, int m=M,const double dx = DX);

__global__ void correct_velocity(double * velocity_grid,double * pressure,int n, int m, const double dx=DX);

__global__ void fillLaplaceValues(double* laplace_discrete,int n=NUM_N,const double dx=DX);

void constructDiscretizedLaplacian(double* laplace_discrete,int n=NUM_N, const double dx=DX);

__global__ void calculateDivergence(const double* velocity_grid,double*divergence,int n=NUM_N, int m=M, const double dx=DX);
}
    
#endif // PRESSURE_CORRECTION_H
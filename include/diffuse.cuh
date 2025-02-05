#pragma once
#ifndef DIFFUSE_H
#define DIFFUSE_H
#include "constants.cuh"
#include "utils.cuh"

void diffuseExplicitStep(const double *velocity_grid, double *velocity_grid_next, const double amount,const int n=NUM_N, const int m=M,const double dx = DX);

void diffuseExplicit(double *velocity_grid, double *velocity_grid_next, const int n=NUM_N, const int m=M, const double dx=DX);

namespace gpu 
{
__global__ void diffuseExplicitStep(double *velocity_grid, const double amount,const int n=NUM_N, const int m=M,const double dx = DX);

void diffuseExplicit(double *velocity_grid, const int n=NUM_N, const int m=M,const double dx = DX);

}
#endif // DIFFUSE_H
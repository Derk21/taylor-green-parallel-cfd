#pragma once
#ifndef DIFFUSE_H
#define DIFFUSE_H
#include "constants.cuh"
#include "utils.cuh"

void diffuseExplicitStep(const double *velocity_grid, double *velocity_grid_next, double amount,int n=NUM_N, int m=M);
void diffuseExplicit(double *velocity_grid, double *velocity_grid_next, int n=NUM_N, int m=M);
void diffuseImplicit(double *velocity_grid, double *velocity_grid_next, int n=NUM_N, int m=M);
#endif // DIFFUSE_H
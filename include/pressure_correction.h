#pragma once
#ifndef PRESSURE_CORRECTION_H
#define PRESSURE_CORRECTION_H

#include "constants.h"
#include "utils.h"
#include "solve.h"
void make_incompressible(double* velocity_grid, double* divergence, double* pressure, int n = NUM_N, int m = M);
void calculateDivergence(const double* velocity_grid, double* divergence, int n = NUM_N, int m = M);
void constructDiscretizedLaplacian(double* laplace_discrete, int n = NUM_N);
void initilizePressure(double *pressure_grid, int n, int m);
void initilizeVelocityGrid(double *velocity_grid,double *periodic_grid,int n ,int m);

#endif // PRESSURE_CORRECTION_H
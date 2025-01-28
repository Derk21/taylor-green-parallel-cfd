#pragma once
#ifndef ADVECT_H
#define ADVECT_H
#include "constants.h"
#include "utils.h"
#include <cmath>

void interpolateVelocity(double x_d, double y_d, int n, int m, const double *periodic_grid, double *velocity_grid, double * velocity_grid_next);

void integrateEuler(double *velocity_grid, int &u_i, int &y_i, int &v_i, const double *periodic_grid, double &x_d,  double &y_d, const double dt, int n, int m);

void advectSemiLagrange(double *velocity_grid, double *velocity_grid_next, const double *periodic_grid, const double dt, int n=NUM_N, int m=M);

#endif // ADVECT_H
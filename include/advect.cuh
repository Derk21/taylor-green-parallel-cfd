#pragma once
#ifndef ADVECT_H
#define ADVECT_H
#include "constants.cuh"
#include "utils.cuh"
#include <cmath>

void interpolateVelocity(double x_d, double y_d, int n, int m, double periodic_start,double periodic_end, const double *periodic_grid, const double *velocity_grid, double * velocity_grid_next);

double get_interpolated(const int &i_closest, const int & y_i_closest,const double &x_diff, const double &y_diff,const double * velocity_grid,int n, int m);

void integrateEuler(const double *velocity_grid, int &y_i, int &u_i, int &v_i, const double *periodic_grid, double &x_d,  double &y_d, const double dt, int n, int m);

void advectSemiLagrange(double *velocity_grid, double *velocity_grid_next, const double *periodic_grid, const double dt, int n=NUM_N, int m=M);

void min_max_neighbors(double &min, double &max, const int idx,const int y_i, const double * velocity_grid,const int n, const int m);

double mac_cormack_correction(const int idx_x,const int y_i,const double * velocity_grid, const double * velocity_grid_bw, const double* velocity_grid_fw, int n=NUM_N, int m=M);

void advectMacCormack(double *velocity_grid, double *velocity_grid_next, const double *periodic_grid, const double dt, int n=NUM_N, int m=M);

#endif // ADVECT_H
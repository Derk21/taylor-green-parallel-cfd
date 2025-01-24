#pragma once
#include "constants.h"
#include "utils.h"
#include <cmath>

void interpolateVelocity(double x_d, double y_d, int n, int m, const double *periodic_grid, double *velocity_grid)
{
    // get grid location
    int u_i_closest, v_i_closest, y_i_closest;
    setClosestGridPointIdx(x_d, y_d, n, m, v_i_closest, y_i_closest);
    u_i_closest = v_i_closest - 1;

    // interpolation weights
    double x_closest = periodic_grid[periodic_linear_Idx(u_i_closest, y_i_closest,n,m)];
    double y_closest = periodic_grid[periodic_linear_Idx(v_i_closest, y_i_closest,n,m)];
    // normalized grid distances
    double x_diff = (x_d - x_closest) / (PERIODIC_END - PERIODIC_START);
    double y_diff = (y_d - y_closest) / (PERIODIC_END - PERIODIC_START);

    // forward bilinear interpolation
    // containing grid cell
    double u = (1.0 - x_diff) * (1.0 - y_diff) * velocity_grid[periodic_linear_Idx(u_i_closest, y_i_closest)];
    double v = (1.0 - y_diff) * (1.0 - y_diff) * velocity_grid[periodic_linear_Idx(v_i_closest, y_i_closest)];
    // x_direction next grid cell
    u += x_diff * (1.0 - y_diff) * velocity_grid[periodic_linear_Idx(u_i_closest + 2, y_i_closest)];
    v += x_diff * (1.0 - y_diff) * velocity_grid[periodic_linear_Idx(v_i_closest + 2, y_i_closest)];
    // y_direction next grid cell
    u += (1.0 - x_diff) * y_diff * velocity_grid[periodic_linear_Idx(u_i_closest, y_i_closest + 1)];
    v += (1.0 - x_diff) * y_diff * velocity_grid[periodic_linear_Idx(v_i_closest, y_i_closest + 1)];
    // next grid cell in diagonal direction 
    u += (1.0 - x_diff) * (1.0 - y_diff) * velocity_grid[periodic_linear_Idx(u_i_closest + 2, y_i_closest + 1)];
    v += (1.0 - x_diff) * (1.0 - y_diff) * velocity_grid[periodic_linear_Idx(v_i_closest + 2, y_i_closest + 1)];

    // assign to closest grid point
    velocity_grid[periodic_linear_Idx(u_i_closest, y_i_closest)] = u;
    velocity_grid[periodic_linear_Idx(v_i_closest, y_i_closest)] = v;
}

void integrateEuler(double *velocity_grid, int &u_i, int &y_i, int &v_i, const double *periodic_grid, double &x_d, const double dt, double &y_d,int n=N, int m=M)
{
    double u_old = velocity_grid[periodic_linear_Idx(u_i, y_i)];
    double v_old = velocity_grid[periodic_linear_Idx(v_i, y_i)];

    double x = periodic_grid[periodic_linear_Idx(u_i, y_i,n,m)];
    double y = periodic_grid[periodic_linear_Idx(v_i, y_i,n,m)];

    x_d = fmod(x + dt * u_old, PERIODIC_END);
    y_d = fmod(y + dt * v_old, PERIODIC_END);
} 

void advectSemiLagrange(double *velocity_grid, const double *periodic_grid, const double dt, int n, int m)
{
    int nn = 2 * n;
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < nn; i+=2)
        {   
            int u_i = i-1;
            int v_i = i;
            double x_d, y_d;
            integrateEuler(velocity_grid, u_i, y_i, v_i, periodic_grid, x_d, -dt, y_d);
            interpolateVelocity(x_d, y_d, n, m, periodic_grid, velocity_grid);
        }
    }
}
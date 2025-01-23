#pragma once
#include "constants.h"
#include <cmath>


int periodic_linear_Idx(const int &x, const int &y, const int bound_x = 2*N,const int bound_y = M)
{
    //return y * bound_x + x;
    return (y % bound_y) * bound_x + (x % bound_x);
}

void setClosestGridPointIdx(float x, float y, int n, int m, int &closest_x_i, int &closest_y_i)
{
    //sets index to y-value,(v_i) right?
    float dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    float dy = (PERIODIC_END - PERIODIC_START) / (m - 1);

    closest_x_i = round((x - PERIODIC_START) / dx);
    closest_y_i = round((y - PERIODIC_START) / dy);

    //Boundary 
    if (closest_x_i < 0) closest_x_i = 0;
    else if (closest_x_i >= n) closest_x_i = n - 1;
    if (closest_y_i < 0) closest_y_i = 0;
    else if (closest_y_i >= m) closest_y_i = m - 1;
}

void taylorGreenGroundTruth(float* periodic_grid,float *velocity_grid_next, int iteration, int n , int m){
    float dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    float dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    int nn = 2 * n;
    float t = iteration * TIMESTEP;
    float F = exp(-2.0f * DIFFUSIVITY * t);
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < nn; i+=2)
        {   
            int u_i = i-1;
            int v_i = i;

            float x = periodic_grid[periodic_linear_Idx(u_i,y_i,n,m)];
            float y = periodic_grid[periodic_linear_Idx(v_i,y_i,n,m)];

            velocity_grid_next[periodic_linear_Idx(u_i,y_i)] =  sin(x) * cos(y) * F;
            velocity_grid_next[periodic_linear_Idx(v_i,y_i)] = -1.0f * cos(x) * sin(y) * F;
        }
    }
}
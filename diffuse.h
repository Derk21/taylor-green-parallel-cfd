#pragma once
#include "constants.h"
#include "utils.h"

void diffuseExplicit(double *velocity_grid,double *velocity_grid_next, int n , int m){
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    int nn = 2 * n;
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < nn; i+=2)
        {   
            int u_i = i-1;
            int v_i = i;

            double u = velocity_grid[periodic_linear_Idx(u_i,y_i)];
            double v = velocity_grid[periodic_linear_Idx(v_i,y_i)];

            double u_left = velocity_grid[periodic_linear_Idx(u_i - 2,y_i)];
            double u_right = velocity_grid[periodic_linear_Idx(u_i + 2,y_i)];
            double u_up = velocity_grid[periodic_linear_Idx(u_i,y_i+1)];
            double u_down = velocity_grid[periodic_linear_Idx(u_i,y_i-1)];

            double v_left = velocity_grid[periodic_linear_Idx(v_i - 2,y_i)];
            double v_right = velocity_grid[periodic_linear_Idx(v_i + 2,y_i)];
            double v_up = velocity_grid[periodic_linear_Idx(v_i,y_i+1)];
            double v_down = velocity_grid[periodic_linear_Idx(v_i,y_i-1)];

            double u_diffusion = DIFFUSIVITY * (u_right - 2 * u + u_left) / (dx * dx) + DIFFUSIVITY * (u_up - 2 * u + u_down) / (dy * dy);
            double v_diffusion = DIFFUSIVITY * (v_right - 2 * v + v_left) / (dx * dx) + DIFFUSIVITY * (v_up - 2 * v + v_down) / (dy * dy);

            velocity_grid_next[periodic_linear_Idx(u_i,y_i)] = u + TIMESTEP * u_diffusion;
            velocity_grid_next[periodic_linear_Idx(v_i,y_i)] = v + TIMESTEP * v_diffusion;
        }
    }
}
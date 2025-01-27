#pragma once
#include "constants.h"
#include "utils.h"

void initializePeriodicGrid(double *periodic_grid, int n, int m)
{
    //TODO: y doesn't change in y_direction, but in x direction
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < 2*n; i+=2)
        {
            int x_i = i / 2;
            periodic_grid[y_i * (2*n) + i - 1] = PERIODIC_START + x_i * dx; //x component 
            periodic_grid[y_i * (2*n) + i] = PERIODIC_START + y_i * dy; //y component 
        }
    }
}


void initilizeVelocityGrid(double *velocity_grid,double *periodic_grid,int n ,int m)
{
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < 2*n; i+=2)
        {
            double x = periodic_grid[y_i * (2*n) + i - 1];
            double y = periodic_grid[y_i * (2*n) + i];

            velocity_grid[y_i * (2*n) + i - 1] = sin(x) * cos(y); //u component 
            velocity_grid[y_i * (2*n) + i] = -1.0 * cos(x) * sin(y); //v component 
        }
    }
}

void initilizePressure(double *pressure_grid,double * periodic_grid, int n ,int m)
{
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    //int nn = 2 * n;
    //double t = iteration * TIMESTEP;
    //double F = exp(-2.0 * DIFFUSIVITY * 1.0);
    double F = 1.0;
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < n; i++)
        {

            int u_i = 2 * (i-1);
            int v_i = 2 * i;

            double x = periodic_grid[periodic_linear_Idx(u_i,y_i,n,m)];
            double y = periodic_grid[periodic_linear_Idx(v_i,y_i,n,m)];
            pressure_grid[y_i * n + i] = (1.0 / 4 )* (cos(2*x)+cos(2*y))*pow(F,2); 
        }
    }
}
#pragma once
#include "constants.h"
#include "utils.h"

void initializePeriodicGrid(float *periodic_grid, int n, int m)
{
    //TODO: y doesn't change in y_direction, but in x direction
    float dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    float dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
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


void initilizeVelocityGrid(float *velocity_grid,float *periodic_grid,int n ,int m)
{
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < 2*n; i+=2)
        {
            float x = periodic_grid[y_i * (2*n) + i - 1];
            float y = periodic_grid[y_i * (2*n) + i];

            velocity_grid[y_i * (2*n) + i - 1] = sin(x) * cos(y); //u component 
            velocity_grid[y_i * (2*n) + i] = -1.0f * cos(x) * sin(y); //v component 
        }
    }
}

void initilizePressure(float *pressure_grid,float * periodic_grid, int n ,int m)
{
    float dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    float dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    //int nn = 2 * n;
    //float t = iteration * TIMESTEP;
    //float F = exp(-2.0f * DIFFUSIVITY * 1.0f);
    float F = 1.0f;
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < n; i++)
        {

            int u_i = 2 * (i-1);
            int v_i = 2 * i;

            float x = periodic_grid[periodic_linear_Idx(u_i,y_i,n,m)];
            float y = periodic_grid[periodic_linear_Idx(v_i,y_i,n,m)];
            pressure_grid[y_i * n + i] = (1.0f / 4 )* (cos(2*x)+cos(2*y))*pow(F,2); 
        }
    }
}
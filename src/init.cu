#include "init.h"

void initializePeriodicGrid(double *periodic_grid, int n, int m,const double periodic_start,const double periodic_end)
{
    //TODO: y doesn't change in y_direction, but in x direction
    double dx = (periodic_end - periodic_start) / (n - 1);
    double dy = (periodic_end - periodic_start) / (m - 1);
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

void initilizePressure(double *pressure_grid, int n, int m)
{
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < n; i++)
        {
            pressure_grid[y_i * n + i] = 0.0; 
        }
    }
}

void initializeGaussianBlob(double *velocity_grid, double *periodic_grid, int n, int m, 
                            double sigma, double amplitude) 
{
    int y_mid_idx = m; 
    int x_mid_idx = 2*n/2; 
    double x0 = periodic_grid[y_mid_idx * n + x_mid_idx];
    double y0 = periodic_grid[y_mid_idx * n + x_mid_idx + 1];
    for (int y_i = 0; y_i < m; y_i++) 
    {
        for (int i = 1; i < 2 * n; i += 2) 
        {
            double x = periodic_grid[y_i * (2 * n) + i - 1];
            double y = periodic_grid[y_i * (2 * n) + i];

            // Compute Gaussian weight
            double r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
            double gaussian = amplitude * exp(-r2 / (2.0 * sigma * sigma));

            velocity_grid[y_i * (2 * n) + i - 1] = (gaussian * x) / 3; // u component
            velocity_grid[y_i * (2 * n) + i] = 0.0; // v component
        }
    }
}
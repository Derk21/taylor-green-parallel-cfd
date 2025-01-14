#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
#include "gnuplot-iostream.h"
#include "plotting.h"

#define N 128               // Grid size X
#define M 128              // Grid size Y
#define ITERATIONS 100000    // Number of iterations
#define PERIODIC_START 0.0f
#define PERIODIC_END 2 * M_PI
#define DIFFUSIVITY 0.1f
#define TIMESTEP 0.5f

#define CHECK_CUDA(call)                                               \
    if ((call) != cudaSuccess)                                         \
    {                                                                  \
        std::cerr << "CUDA error at " << __LINE__ << ":" << std::endl; \
        std::cerr << (cudaGetErrorString(call)) << std::endl;          \
        exit(EXIT_FAILURE);                                            \
    }



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

void initilizeVelocityGrid(float *velocity_grid,float *periodic_grid,int n ,int m){
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


int main()
{
    float *periodic_grid = (float *)malloc(N * M * 2 * sizeof(float));
    float *velocity_grid = (float *)malloc(N * M * 2 * sizeof(float));
    //float *curr = (float *)malloc(N * M * 2 * sizeof(float));
    //float *next = (float *)malloc(N * M * 2 * sizeof(float));


    // Check for allocation failures
    //if (curr == NULL || next == NULL)
    if (periodic_grid == NULL || velocity_grid == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize the grids
    //initializePeriodicGrid(curr, N, M);
    //initializePeriodicGrid(next, N, M);
    initializePeriodicGrid(periodic_grid,N,M);
    initilizeVelocityGrid(velocity_grid,periodic_grid,N,M);

    //float *d_curr;
    //allocate memory on device    
    //CHECK_CUDA(cudaMalloc(&d_curr, N * M * sizeof(float)));

    //copy data to device
    //CHECK_CUDA(cudaMemcpy(d_curr, curr, N * M * sizeof(float), cudaMemcpyHostToDevice));
    std::cout << "first of periodic grid:" << std::endl;
    for (int y_i = 0; y_i < 5; ++y_i)
    {
        for (int x_i = 1; x_i < 10; x_i+=2)
        {
            std::cout << periodic_grid[y_i * (2*N) + x_i-1] << "," << periodic_grid[y_i * (2*N) + x_i] <<" ";
        }
        std::cout << std::endl;
    }
    plotPeriodicGrid(periodic_grid, N, M);
    plotVelocityGrid(periodic_grid, velocity_grid, N, M, PERIODIC_START, PERIODIC_END);
    free(periodic_grid);
}
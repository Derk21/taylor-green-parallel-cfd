#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
#include "gnuplot-iostream.h"
#define N 128               // Grid size X
#define M 128              // Grid size Y
#define ITERATIONS 100000    // Number of iterations
#define PERIODIC_START 0.0f
#define PERIODIC_END 2 * M_PI

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
            int x_i = i - 1;
            periodic_grid[y_i * n + i] = PERIODIC_START + y_i * dy; //y component 
            periodic_grid[y_i * n + i - 1] = PERIODIC_START + x_i * dx; //x component 
        }
    }
}


int main()
{
    float *periodic_grid = (float *)malloc(N * M * 2 * sizeof(float));
    //float *curr = (float *)malloc(N * M * 2 * sizeof(float));
    //float *next = (float *)malloc(N * M * 2 * sizeof(float));


    // Check for allocation failures
    //if (curr == NULL || next == NULL)
    if (periodic_grid == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize the grids
    //initializePeriodicGrid(curr, N, M);
    //initializePeriodicGrid(next, N, M);
    initializePeriodicGrid(periodic_grid,N,M);

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
            std::cout << periodic_grid[y_i * N + x_i-1] << "," << periodic_grid[y_i * N + x_i] <<" ";
        }
        std::cout << std::endl;
    }
    //plot periodic grid 
    //std::vector<std::vector<std::pair<float, float>>> periodic_grid_plot;
    //for (int y_i = 0; y_i < M; ++y_i)
    //{
        //std::vector<std::pair<float,float>> row;
        //for (int i = 1; i < 2*N; i+=2)
        //{
            //std::pair p = std::make_pair(periodic_grid[y_i * N + i-1], periodic_grid[y_i * N + i]);
            //row.emplace_back(p);
        //}
        //periodic_grid_plot.push_back(row);
    //}
    std::ofstream data_file("periodic_grid_data.dat");
    for (int y_i = 0; y_i < M; ++y_i)
    {
        for (int i = 0; i < 2*N; i+=2)
        {
            float x = periodic_grid[y_i * N + i-1];
            float y = periodic_grid[y_i * N + i];
            data_file << x << " " << y << " " << 1.0 << "\n";
        } 
        data_file << "\n";
    }
    data_file.close();
    Gnuplot gp;
    gp << "set terminal png size 800,600\n"; // Use PNG terminal with specified size
    gp << "set output 'periodic_grid_plot.png'\n"; // Output file
    gp << "set view map\n"; // 2D projection
    gp << "set pm3d at b\n"; // Enable color mapping
    gp << "splot 'periodic_grid_data.dat' using 1:2:3 with pm3d\n"; // Plot with color mapping
    //gp.send2d(periodic_grid_plot);          // Send data
    free(periodic_grid);
}
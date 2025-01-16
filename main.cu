#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
#include "gnuplot-iostream.h"
#include "plotting.h"

#define N 64               // Grid size X
#define M 64              // Grid size Y
#define ITERATIONS 100    // Number of iterations
#define PERIODIC_START 0.0f
#define PERIODIC_END 2 * M_PI
#define DIFFUSIVITY 0.1f
#define TIMESTEP 1e-3f

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

int periodic_linear_Idx(const int &x, const int &y, const int bound_x = 2*N,const int bound_y = M)
{
    //return y * bound_x + x;
    return (y % bound_y) * bound_x + (x % bound_x);
}
void diffuseExplicit(float *velocity_grid,float *velocity_grid_next, int n , int m){
    float dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    float dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    int nn = 2 * n;
    //TODO: Boundary 
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < nn; i+=2)
        {   
            int u_i = i-1;
            int v_i = i;

            float u = velocity_grid[periodic_linear_Idx(u_i,y_i)];
            float v = velocity_grid[periodic_linear_Idx(v_i,y_i)];

            float u_left = velocity_grid[periodic_linear_Idx(u_i - 2,y_i)];
            float u_right = velocity_grid[periodic_linear_Idx(u_i + 2,y_i)];
            float u_up = velocity_grid[periodic_linear_Idx(u_i,y_i+1)];
            float u_down = velocity_grid[periodic_linear_Idx(u_i,y_i-1)];

            float v_left = velocity_grid[periodic_linear_Idx(v_i - 2,y_i)];
            float v_right = velocity_grid[periodic_linear_Idx(v_i + 2,y_i)];
            float v_up = velocity_grid[periodic_linear_Idx(v_i,y_i+1)];
            float v_down = velocity_grid[periodic_linear_Idx(v_i,y_i-1)];

            float u_diffusion = DIFFUSIVITY * (u_right - 2 * u + u_left) / (dx * dx) + DIFFUSIVITY * (u_up - 2 * u + u_down) / (dy * dy);
            float v_diffusion = DIFFUSIVITY * (v_right - 2 * v + v_left) / (dx * dx) + DIFFUSIVITY * (v_up - 2 * v + v_down) / (dy * dy);

            velocity_grid_next[periodic_linear_Idx(u_i,y_i)] = u + TIMESTEP * u_diffusion;
            velocity_grid_next[periodic_linear_Idx(v_i,y_i)] = v + TIMESTEP * v_diffusion;
        }
    }
}


int main()
{
    float *periodic_grid = (float *)malloc(N * M * 2 * sizeof(float));
    float *velocity_grid = (float *)malloc(N * M * 2 * sizeof(float));
    float *velocity_grid_next = (float *)malloc(N * M * 2 * sizeof(float));


    // Check for allocation failures
    if (periodic_grid == NULL || velocity_grid == NULL || velocity_grid_next == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    initializePeriodicGrid(periodic_grid,N,M);
    initilizeVelocityGrid(velocity_grid,periodic_grid,N,M);
    //float *d_curr;
    //allocate memory on device    
    //CHECK_CUDA(cudaMalloc(&d_curr, N * M * sizeof(float)));
    //copy data to device
    //CHECK_CUDA(cudaMemcpy(d_curr, curr, N * M * sizeof(float), cudaMemcpyHostToDevice));
    std::string dirName = createTimestampedDirectory();
    plotPeriodicGrid(periodic_grid, N, M);
    std::string plot_name("velocity_0");
    plotVelocityGrid(periodic_grid, velocity_grid, N, M, PERIODIC_START, PERIODIC_END,plot_name, dirName);
    for (int i = 1; i < ITERATIONS+1; i++){
        diffuseExplicit(velocity_grid,velocity_grid_next,N,M);
        std::swap(velocity_grid,velocity_grid_next);
        plot_name = "velocity_" + std::to_string(i);
        plotVelocityGrid(periodic_grid, velocity_grid, N, M, PERIODIC_START, PERIODIC_END,plot_name, dirName);
    }
    std::cout << "Creating velocity animation" << std::endl;
    createGifFromPngs(dirName,"animation_velocity.gif",PERIODIC_START,PERIODIC_END);

    //for (int y_i = 0; y_i < 5; ++y_i)
    //{
        //for (int x_i = 1; x_i < 10; x_i+=2)
        //{
            //std::cout << periodic_grid[y_i * (2*N) + x_i-1] << "," << periodic_grid[y_i * (2*N) + x_i] <<" ";
        //}
        //std::cout << std::endl;
    //}

    free(periodic_grid);
    free(velocity_grid);
    free(velocity_grid_next);
}
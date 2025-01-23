#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
#include <iomanip>
#include "gnuplot-iostream.h"
#include "plotting.h"

#define N 64               // Grid size X
#define M 64              // Grid size Y
#define ITERATIONS 100    // Number of iterations
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

void eulerianIntegrator(float *departure_points, const float *periodic_grid, const float *velocity_grid,float dt, int n, int m)
{
    int nn = 2 * n;
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < nn; i+=2)
        {   
            int u_i = i-1;
            int v_i = i;

            float u = velocity_grid[periodic_linear_Idx(u_i,y_i)];
            float v = velocity_grid[periodic_linear_Idx(v_i,y_i)];

            float x = periodic_grid[periodic_linear_Idx(u_i,y_i)];
            float y = periodic_grid[periodic_linear_Idx(v_i,y_i)];

            departure_points[periodic_linear_Idx(u_i,y_i)] = fmod(x + dt * u, PERIODIC_END);
            departure_points[periodic_linear_Idx(v_i,y_i)] = fmod(y + dt * v, PERIODIC_END);
        }
    }
}

void interpolateVelocity(float *velocity_grid, const float *departure_points, const float *periodic_grid, int n, int m)
{
    //bilinear interpolation of gridpoints
    int nn = 2 * n;
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < nn; i+=2)
        {   
            int u_i = i-1;
            int v_i = i;
            //get grid location
            float x_d = departure_points[periodic_linear_Idx(u_i,y_i)];
            float y_d = departure_points[periodic_linear_Idx(v_i,y_i)];
            int u_i_closest, v_i_closest, y_i_closest;
            setClosestGridPointIdx(x_d, y_d, n, m, v_i_closest, y_i_closest);
            u_i_closest = v_i_closest - 1;
            
            //interpolation weights
            float x_closest = periodic_grid[periodic_linear_Idx(u_i_closest,y_i_closest)];
            float y_closest = periodic_grid[periodic_linear_Idx(v_i_closest,y_i_closest)];
            //normalized grid distances
            float x_diff = (x_d - x_closest) / (PERIODIC_END - PERIODIC_START);
            float y_diff = (y_d - y_closest) / (PERIODIC_END - PERIODIC_START);

            //forward bilinear interpolation
            //containing grid cell
            float u = (1.0f - x_diff) * (1.0f - y_diff) * velocity_grid[periodic_linear_Idx(u_i_closest,y_i_closest)];
            float v = (1.0f - y_diff) * (1.0f - y_diff) * velocity_grid[periodic_linear_Idx(v_i_closest,y_i_closest)];
            //x_direction next grid cell
            u += x_diff * (1.0f - y_diff) * velocity_grid[periodic_linear_Idx(u_i_closest+2,y_i_closest)];
            v += x_diff * (1.0f - y_diff) * velocity_grid[periodic_linear_Idx(v_i_closest+2,y_i_closest)];
            //y_direction next grid cell
            u += (1.0f - x_diff) * y_diff * velocity_grid[periodic_linear_Idx(u_i_closest,y_i_closest+1)];
            v += (1.0f - x_diff) * y_diff * velocity_grid[periodic_linear_Idx(v_i_closest,y_i_closest+1)];
            //next grid cell in both directions
            u += (1.0f - x_diff) * (1.0f - y_diff) * velocity_grid[periodic_linear_Idx(u_i_closest+2,y_i_closest+1)];
            v += (1.0f - x_diff) * (1.0f - y_diff) * velocity_grid[periodic_linear_Idx(v_i_closest+2,y_i_closest+1)];

            //assign to closest grid point
            velocity_grid[periodic_linear_Idx(u_i_closest,y_i_closest)] = u;
            velocity_grid[periodic_linear_Idx(v_i_closest,y_i_closest)] = v;
        }
    }
}

//void advectSemiLagrange(float *velocity_grid,float *velocity_grid_next, int n , int m){
    //float dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    //float dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    //int nn = 2 * n;
    
    ////look up velocity at previous timestep
//}

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

            float x = periodic_grid[periodic_linear_Idx(u_i,y_i)];
            float y = periodic_grid[periodic_linear_Idx(v_i,y_i)];

            velocity_grid_next[periodic_linear_Idx(u_i,y_i)] =  sin(x) * cos(y) * F;
            velocity_grid_next[periodic_linear_Idx(v_i,y_i)] = -1.0f * cos(x) * sin(y) * F;
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
    std::string plot_name("velocity_0000");
    plotVelocityGrid(periodic_grid, velocity_grid, N, M, PERIODIC_START, PERIODIC_END,plot_name, dirName);
    for (int i = 1; i < ITERATIONS+1; i++){
        //diffuseExplicit(velocity_grid,velocity_grid_next,N,M);
        taylorGreenGroundTruth(periodic_grid,velocity_grid_next,i,N,M);
        std::swap(velocity_grid,velocity_grid_next);
        std::stringstream plot_name;
        plot_name << "velocity_"<< std::setw(4) << std::setfill('0') << i;
        plotVelocityGrid(periodic_grid, velocity_grid, N, M, PERIODIC_START, PERIODIC_END,plot_name.str(), dirName);
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
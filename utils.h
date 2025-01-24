#pragma once
#include "constants.h"
#include <cmath>

#define CHECK_CUDA(call)                                               \
    if ((call) != cudaSuccess)                                         \
    {                                                                  \
        std::cerr << "CUDA error at " << __LINE__ << ":" << std::endl; \
        std::cerr << (cudaGetErrorString(call)) << std::endl;          \
        exit(EXIT_FAILURE);                                            \
    }

//copied from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/utils/cusolver_utils.h
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

int periodic_linear_Idx(const int &x, const int &y, const int bound_x = 2*N,const int bound_y = M)
{   
    int mod_x = ((x % bound_x) + bound_x) % bound_x; // ensures non-negative result
    int mod_y = ((y % bound_y) + bound_y) % bound_y;
    return mod_y * bound_x + mod_x;
}

void setClosestGridPointIdx(double x, double y, int n, int m, int &closest_x_i, int &closest_y_i)
{
    //sets index to y-value,(v_i) right?
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);

    closest_x_i = round((x - PERIODIC_START) / dx);
    closest_y_i = round((y - PERIODIC_START) / dy);

    //Boundary 
    if (closest_x_i < 0) closest_x_i = 0;
    else if (closest_x_i >= n) closest_x_i = n - 1;
    if (closest_y_i < 0) closest_y_i = 0;
    else if (closest_y_i >= m) closest_y_i = m - 1;
}

void taylorGreenGroundTruth(double* periodic_grid,double *velocity_grid_next, int iteration, int n , int m){
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    int nn = 2 * n;
    double t = iteration * TIMESTEP;
    double F = exp(-2.0 * DIFFUSIVITY * t);
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < nn; i+=2)
        {   
            int u_i = i-1;
            int v_i = i;

            double x = periodic_grid[periodic_linear_Idx(u_i,y_i,n,m)];
            double y = periodic_grid[periodic_linear_Idx(v_i,y_i,n,m)];

            velocity_grid_next[periodic_linear_Idx(u_i,y_i)] =  sin(x) * cos(y) * F;
            velocity_grid_next[periodic_linear_Idx(v_i,y_i)] = -1.0 * cos(x) * sin(y) * F;
        }
    }
}

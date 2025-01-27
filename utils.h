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

void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    //copied from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/utils/cusolver_utils.h
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

int periodic_linear_Idx(const int &x, const int &y, const int bound_x = 2*N,const int bound_y = M)
{   
    int mod_x = (x + bound_x) % bound_x; // ensures non-negative result
    int mod_y = (y + bound_y) % bound_y;
    return mod_y * bound_x + mod_x;
}

void setClosestGridPointIdx(double x, double y, int n, int m, int &closest_x_i, int &closest_y_i)
{
    //sets index to y-value,(v_i) 
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);

    closest_x_i = round((x - PERIODIC_START) / dx);
    closest_y_i = round((y - PERIODIC_START) / dy);

    //periodic boundary conditions
    closest_x_i = (closest_x_i + n) % n;
    closest_y_i = (closest_y_i + m) % m;

    //convert to v_i coordinate
    closest_x_i = closest_x_i * 2 + 1;
}

void test_setClosestGridPointIdx()
{
    int closest_x_i, closest_y_i;
    int n = N;
    int m = M;

    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    // Test case 1: Point within bounds
    setClosestGridPointIdx(dx, dx, n, m, closest_x_i, closest_y_i);
    assert(closest_x_i == 3);
    assert(closest_y_i == 1);

    // Test case 2: Point at the boundary
    setClosestGridPointIdx(PERIODIC_END, PERIODIC_END, n, m, closest_x_i, closest_y_i);
    assert(closest_x_i == 2*n - 1);
    assert(closest_y_i == m - 1);

    // Test case 3: Point outside the boundary (positive)
    setClosestGridPointIdx(PERIODIC_END + dx/2, PERIODIC_END + dy/2, n, m, closest_x_i, closest_y_i);
    assert(closest_x_i == 1);
    assert(closest_y_i == 0);

    // Test case 4: Point outside the boundary (negative)
    setClosestGridPointIdx(PERIODIC_START - dx/2, PERIODIC_START - dy/2, n, m, closest_x_i, closest_y_i);
    assert(closest_x_i == 2*n - 1);
    assert(closest_y_i == m - 1);

    // Test case 5: Point exactly at the start
    setClosestGridPointIdx(PERIODIC_START, PERIODIC_START, n, m, closest_x_i, closest_y_i);
    assert(closest_x_i == 1);
    assert(closest_y_i == 0);

    // Test case 6: Point exactly at the middle

    setClosestGridPointIdx((PERIODIC_START + PERIODIC_END) / 2 + 1e-10, (PERIODIC_START + PERIODIC_END) / 2 +1e-10, n, m, closest_x_i, closest_y_i);
    assert(closest_x_i == 2*(n / 2)+1);
    assert(closest_y_i == m / 2);

    std::cout << "All test cases passed!" << std::endl;
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

            double x = periodic_grid[periodic_linear_Idx(u_i,y_i)];
            double y = periodic_grid[periodic_linear_Idx(v_i,y_i)];

            velocity_grid_next[periodic_linear_Idx(u_i,y_i)] =  sin(x) * cos(y) * F;
            velocity_grid_next[periodic_linear_Idx(v_i,y_i)] = -1.0 * cos(x) * sin(y) * F;
        }
    }
}

void setPressureGroundTruth(double *pressure_grid,double * periodic_grid,int iteration, int n ,int m)
{
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    double t = iteration * TIMESTEP;
    double F = exp(-2.0 * DIFFUSIVITY * t);
    double rho = 0.1;
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < n; i++)
        {

            int u_i = 2 * (i-1);
            int v_i = 2 * i;

            double x = periodic_grid[periodic_linear_Idx(u_i,y_i,n,m)];
            double y = periodic_grid[periodic_linear_Idx(v_i,y_i,n,m)];
            pressure_grid[y_i * n + i] = (rho / 4 )* (cos(2*x)+cos(2*y))*pow(F,2); 
        }
    }
}

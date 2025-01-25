#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
//#include <cusolverSp.h>
#include <cusolverDn.h>
#include <iomanip>
#include "gnuplot-iostream.h"
#include "plotting.h"
#include "constants.h"
#include "utils.h"
#include "init.h"
#include "advect.h"
#include "diffuse.h"
#include <cassert>


void calculateDivergence(double* velocity_grid,double*divergence,int n=N, int m=M){

    //divergence by central differences
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 0; i < n; i++)
        {   
            int u_i = 2 * i;
            int v_i = 2 * i + 1;

            double u_left = velocity_grid[periodic_linear_Idx(u_i - 2,y_i)];
            double u_right = velocity_grid[periodic_linear_Idx(u_i + 2,y_i)];

            double v_up = velocity_grid[periodic_linear_Idx(v_i,y_i+1)];
            double v_down = velocity_grid[periodic_linear_Idx(v_i,y_i-1)];

            double div = (u_right - u_left) / (2 * dx) + (v_up - v_down) / (2 * dy);
            divergence[periodic_linear_Idx(i,y_i,n,m)] = div;
        }
    }
}

void poissonSolve(double*pressure,double * pressure_next, double* divergence,int n=N, int m=M)
{
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    //double* pressure_next = (double *)malloc(n * m * sizeof(double));

    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < n; i++)
        {   
            double b = divergence[periodic_linear_Idx(i,y_i,n,m)];
            double p = pressure[periodic_linear_Idx(i,y_i,n,m)];
            double p_left = pressure[periodic_linear_Idx(i-1,y_i,n,m)];
            double p_right = pressure[periodic_linear_Idx(i+1,y_i,n,m)];
            double p_up = pressure[periodic_linear_Idx(i,y_i+1,n,m)];
            double p_down = pressure[periodic_linear_Idx(i,y_i-1,n,m)];

            double p_next = (p_right + p_left)*dx*dx + (p_up + p_down)*dy*dy - b*dx*dx*dy*dy;
            p_next = p_next / (2*(dx*dx + dy*dy));
            pressure_next[periodic_linear_Idx(i,y_i,n,m)] = p_next;
        }
    } 
    memcpy(pressure,pressure_next,n*m*sizeof(double));
}

void constructDiscretizedLaplacian(double* laplace_discrete,int n=N){
    //discretized laplacian is always same for grid -> unit laplacian is sufficient
    //order 2
    for (int i = 0; i < n; i++)
    {
        laplace_discrete[periodic_linear_Idx(i,i,n,n)] = -4.0;
        laplace_discrete[periodic_linear_Idx(i,i-1,n,n)] = 1.0;
        laplace_discrete[periodic_linear_Idx(i,i+1,n,n)] = 1.0;
        laplace_discrete[periodic_linear_Idx(i-1,i,n,n)] = 1.0;
        laplace_discrete[periodic_linear_Idx(i+1,i,n,n)] = 1.0;
    }
}

//void solveDense(double* A, double* b, double* x, int n=N){
    ////A is discretized laplacian
    ////b is divergence (flat)
    ////x is pressure (flat)

    ////LU decomposition with partial pivoting
    ////needs to be double for cusolver
    ////probably many are 0 -> TODO: sparse solver

    ////adapted from https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSOLVER/getrf

    //cusolverDnHandle_t cusolverH = NULL;
    //cudaStream_t stream = NULL;
    //const int64_t lda = n;
    //const int64_t ldb = n;
    //std::vector<int64_t> Ipiv(n, 0);
    ////1. put into cublas format
    //double *d_A, *d_b, *d_x;
    //CHECK_CUDA(cudaMalloc(&d_A, n * n * sizeof(double)));
    //CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(double)));
    //CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(double)));

    //int64_t *d_Ipiv = nullptr; /* pivoting sequence */

    ////copy to device
    ////CHECK_CUDA(cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice));
    ////CHECK_CUDA(cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice));
    //CHECK_CUDA(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));

    //CHECK_CUDA(cudaMemcpyAsync(d_A, A, sizeof(double) * n * n, cudaMemcpyHostToDevice,
                               //stream));
    //CHECK_CUDA(cudaMemcpyAsync(d_b, b, sizeof(double) * n * n, cudaMemcpyHostToDevice,
                               //stream));
    //CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * Ipiv.size()));
    ////CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));


    //const int pivot_on = 1;

    //if (pivot_on) {
        //std::printf("pivot is on : compute P*A = L*U \n");
    //} else {
        //std::printf("pivot is off: compute A = L*U (not numerically stable)\n");
    //}

    //// solver handle
    //cusolverDnHandle_t cusolverH;
    //CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    //CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    //CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    //cusolverDnParams_t params;
    //CUSOLVER_CHECK(cusolverDnCreateParams(&params));
        
    //CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));

    //CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * n * n));
    //CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * n * n));


    ////workspace 
    //int workspace_size = 0;
    //CHECK_CUSOLVE(cusolverDnSgetrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_FULL, n, d_A, n, &workspace_size));

    //double* d_workspace;
    //CHECK_CUDA(cudaMalloc(&d_workspace, sizeof(double) * workspace_size));

    //int* dev_info;
    //CHECK_CUDA(cudaMalloc(&dev_info, sizeof(int)));

    ////Cholesky factorization
    //CHECK_CUSOLVE(
        //cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_FULL, n, d_A, n, d_workspace, workspace_size, dev_info)
    //);

    //// Solve for x (forward and backward substitution)
    //CHECK_CUSOLVE(
        //cusolverDnDpotrs(cusolver_handle, CUBLAS_FILL_MODE_FULL, n, 1, d_A, n, d_b, n, dev_info)
    //);

    //// Copy the solution back to the host
    //CHECK_CUDA(cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    //// Check for errors
    //int info;
    //CHECK_CUDA(cudaMemcpy(&info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    //if (info != 0) {
        //std::cerr << "cuSolver failed with info = " << info << std::endl;
    //} else {
        //std::cout << "Solution vector x:" << std::endl;
        //for (int i = 0; i < n; ++i) {
            //std::cout << x[i] << std::endl;
        //}
    //}

    //// Free resources
    //CHECK_CUDA(cudaFree(d_A));
    //CHECK_CUDA(cudaFree(d_b));
    //CHECK_CUDA(cudaFree(d_x));
    //CHECK_CUDA(cudaFree(d_workspace));
    //CHECK_CUDA(cudaFree(dev_info));
    //CHECK_CUSOLVE(cusolverDnDestroy(cusolver_handle));
//}

double l1_norm(double* a, double* b, int n){
    //mean reduction
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += abs(a[i] - b[i]);
    }
    return sum;
}

void make_incompressible(double* velocity_grid, double* divergence, double*pressure, int n=N, int m=M){
    calculateDivergence(velocity_grid,divergence);

    //explicit solve for now 
    const double TOLERANCE = 1.0;
    double *pressure_next = (double *)malloc(n * m * sizeof(double));
    memcpy(pressure_next,pressure,n*m*sizeof(double));
    for(int i = 0; i < 100; i++){
        poissonSolve(pressure,pressure_next,divergence);
        double loss = l1_norm(pressure,divergence,n*m);
        std::cout << "Loss: " << loss << std::endl;
        if (loss < TOLERANCE){
            break;
        }
        //TODO: l1 norm of pressure - divergence
    }
    free(pressure_next);
    //TODO: implicit solver 

    //constructDiscretizedLaplacian(laplace);
    //solveDense(laplace,divergence,pressure,n);
    //calculate gradient
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < n; i++)
        {   
            int u_i = 2 * (i - 1);
            int v_i = 2 * i;
            //TODO: check which finite difference to use
            double p = pressure[periodic_linear_Idx(i,y_i,n,m)];
            double p_right = pressure[periodic_linear_Idx(i+1,y_i,n,m)];
            double p_down = pressure[periodic_linear_Idx(i,y_i+1,n,m)];
            double p_dx = (p_right - p) / dx;
            double p_dy = (p_right - p) / dy;

            //TODO:scale somehow with dt?
            velocity_grid[periodic_linear_Idx(u_i,y_i)] -= p_dx;
            velocity_grid[periodic_linear_Idx(v_i,y_i)] -= p_dy;
        }
    }
    

}


int main()
{   
    test_setClosestGridPointIdx();
    //vortex decays exponentially -> use double to stabilize
    double *periodic_grid = (double *)malloc(N * M * 2 * sizeof(double));
    double *velocity_grid = (double *)malloc(N * M * 2 * sizeof(double));
    double *velocity_grid_next = (double *)malloc(N * M * 2 * sizeof(double));
    //doubles to make compatible with cuSolver
    double *divergence = (double *)malloc(N * M * sizeof(double));
    double *pressure = (double *)malloc(N * M * sizeof(double));


    // Check for allocation failures
    if (periodic_grid == NULL || velocity_grid == NULL || velocity_grid_next == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    initializePeriodicGrid(periodic_grid,N,M);
    initilizeVelocityGrid(velocity_grid,periodic_grid,N,M);
    memcpy(velocity_grid_next,velocity_grid,N*M*2*sizeof(double));
    //double *d_curr;
    //allocate memory on device    
    //CHECK_CUDA(cudaMalloc(&d_curr, N * M * sizeof(double)));
    //copy data to device
    //CHECK_CUDA(cudaMemcpy(d_curr, curr, N * M * sizeof(double), cudaMemcpyHostToDevice));
    std::string dirName = createTimestampedDirectory();
    //plotPeriodicGrid(periodic_grid, N, M);
    std::string plot_name("velocity_0000");
    plotVelocityGrid(periodic_grid, velocity_grid, N, M, PERIODIC_START, PERIODIC_END,plot_name, dirName);
    for (int i = 1; i < ITERATIONS+1; i++){
        diffuseExplicit(velocity_grid,velocity_grid_next,N,M);
        std::stringstream plot_name;
        //plot_name << "velocity_"<< std::setw(4) << std::setfill('0') << i << "_diffused";
        //plotVelocityGrid(periodic_grid, velocity_grid, N, M, PERIODIC_START, PERIODIC_END,plot_name.str(), dirName);
        //plot_name.str("");
        advectSemiLagrange(velocity_grid,velocity_grid_next,periodic_grid,TIMESTEP,N,M);
        make_incompressible(velocity_grid,divergence,pressure);
        //taylorGreenGroundTruth(periodic_grid,velocity_grid_next,i,N,M);
        //std::swap(velocity_grid,velocity_grid_next);
        plot_name << "velocity_"<< std::setw(4) << std::setfill('0') << i;
        plotVelocityGrid(periodic_grid, velocity_grid, N, M, PERIODIC_START, PERIODIC_END,plot_name.str(), dirName);
        plot_name.str("");
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
    free(divergence);
    free(pressure);
}
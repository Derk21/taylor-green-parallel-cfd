#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
#include <iomanip>
#include "gnuplot-iostream.h"
#include "plotting.cuh"
#include "constants.cuh"
#include "utils.cuh"
#include "init.cuh"
#include "advect.cuh"
#include "diffuse.cuh"
#include "pressure_correction.cuh"


//void solveDense(const std::vector<double> &A, const std::vector<double>& B, std::vector<double> & X, size_t m=2*NUM_N){





int main()
{   
    //testSolveDense();
    //vortex decays exponentially -> use double to stabilize
    double *periodic_grid = (double *)malloc(NUM_N * M * 2 * sizeof(double));
    double *velocity_grid = (double *)malloc(NUM_N * M * 2 * sizeof(double));
    double *velocity_grid_next = (double *)malloc(NUM_N * M * 2 * sizeof(double));
    //doubles to make compatible with cuSolver
    double *divergence = (double *)malloc(NUM_N * M * sizeof(double));
    double *pressure = (double *)malloc(NUM_N * M * sizeof(double));

    //setPressureGroundTruth(pressure,periodic_grid,1,NUM_N,M);
    initializePressure(pressure,NUM_N,M);
    initializePeriodicGrid(periodic_grid,NUM_N,M);
    initializeVelocityGrid(velocity_grid,periodic_grid,NUM_N,M);
    //initializeGaussianBlob(velocity_grid,periodic_grid,NUM_N,M,0.5,1);
    memcpy(velocity_grid_next,velocity_grid,NUM_N*M*2*sizeof(double));

    // Check for allocation failures
    if (periodic_grid == NULL || velocity_grid == NULL || velocity_grid_next == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    //if (GPU){
        //double * d_periodic_grid,* d_velocity_grid, * d_velocity_grid_next, *d_laplacian_discrete;
        //CHECK_CUDA(cudaMalloc(&d_periodic_grid, NUM_N * M * 2 * sizeof(double)));
        //CHECK_CUDA(cudaMalloc(&d_velocity_grid, NUM_N * M * 2 * sizeof(double)));
        //CHECK_CUDA(cudaMalloc(&d_velocity_grid_next, NUM_N * M * 2 * sizeof(double)));
        //CHECK_CUDA(cudaMalloc(&d_laplacian_discrete, NUM_N * M * NUM_N * M * sizeof(double)));

        //const int PADDED_SIZE = TILE_SIZE + 2; //might be better to do more depending on velocities
        //const int dx = (PERIODIC_END-PERIODIC_START) / (NUM_N - 1);
        //dim3 blockDim(TILE_SIZE);
        //dim3 gridDimLaplace((NUM_N*NUM_N + TILE_SIZE-1)/TILE_SIZE); 
        //gpu::fillLaplaceValues<<<gridDimLaplace,blockDim>>>(d_laplacian_discrete,NUM_N,dx);

        
    //}

    
    //double *d_curr;
    //allocate memory on device    
    //CHECK_CUDA(cudaMalloc(&d_curr, NUM_N * M * sizeof(double)));
    //copy data to device
    //CHECK_CUDA(cudaMemcpy(d_curr, curr, NUM_N * M * sizeof(double), cudaMemcpyHostToDevice));
    std::string dirName = createTimestampedDirectory();
    //plotPeriodicGrid(periodic_grid, NUM_N, M);
    std::string plot_name("velocity_0000");
    plotVelocityGrid(periodic_grid, velocity_grid, NUM_N, M, PERIODIC_START, PERIODIC_END,plot_name, dirName);
    for (int i = 1; i < ITERATIONS+1; i++){
        std::stringstream plot_name;
        diffuseExplicit(velocity_grid,velocity_grid_next);
        advectSemiLagrange(velocity_grid,velocity_grid_next,periodic_grid,TIMESTEP);
        //advectMacCormack(velocity_grid,velocity_grid_next,periodic_grid,TIMESTEP);
        makeIncompressible(velocity_grid,divergence,pressure);

        //taylorGreenGroundTruth(periodic_grid,velocity_grid_next,i,NUM_N,M);
        //std::swap(velocity_grid,velocity_grid_next);
        
        plot_name << "velocity_"<< std::setw(4) << std::setfill('0') << i;
        plotVelocityGrid(periodic_grid, velocity_grid, NUM_N, M, PERIODIC_START, PERIODIC_END,plot_name.str(), dirName);
        plot_name.str("");
    }
    std::cout << "Creating velocity animation" << std::endl;
    createGifFromPngs(dirName,"animation_velocity.gif",PERIODIC_START,PERIODIC_END);
    plotErrors("plots/_ground_truth",dirName);

    free(periodic_grid);
    free(velocity_grid);
    free(velocity_grid_next);
    free(divergence);
    free(pressure);
}
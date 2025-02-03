#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
#include <iomanip>
#include "gnuplot-iostream.h"
#include "plotting.h"
#include "constants.h"
#include "utils.h"
#include "init.h"
#include "advect.h"
#include "diffuse.h"
#include "pressure_correction.h"


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


    // Check for allocation failures
    if (periodic_grid == NULL || velocity_grid == NULL || velocity_grid_next == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    //setPressureGroundTruth(pressure,periodic_grid,1,NUM_N,M);
    initilizePressure(pressure,NUM_N,M);
    initializePeriodicGrid(periodic_grid,NUM_N,M);

    initilizeVelocityGrid(velocity_grid,periodic_grid,NUM_N,M);

    //initializeGaussianBlob(velocity_grid,periodic_grid,NUM_N,M,0.2,1);

    memcpy(velocity_grid_next,velocity_grid,NUM_N*M*2*sizeof(double));
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
        //advectSemiLagrange(velocity_grid,velocity_grid_next,periodic_grid,TIMESTEP);
        advectMacCormack(velocity_grid,velocity_grid_next,periodic_grid,TIMESTEP);
        make_incompressible(velocity_grid,divergence,pressure);

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
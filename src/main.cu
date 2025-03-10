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

int main()
{
    // vortex decays exponentially -> use double to stabilize
    double *periodic_grid = (double *)malloc(NUM_N * M * 2 * sizeof(double));
    double *velocity_grid = (double *)malloc(NUM_N * M * 2 * sizeof(double));
    double *velocity_grid_next = (double *)malloc(NUM_N * M * 2 * sizeof(double));
    // doubles to make compatible with cuSolver
    double *divergence = (double *)malloc(NUM_N * M * sizeof(double));
    double *pressure = (double *)malloc(NUM_N * M * sizeof(double));

    // setPressureGroundTruth(pressure,periodic_grid,1,NUM_N,M);
    initializePressure(pressure, NUM_N, M);
    initializePeriodicGrid(periodic_grid, NUM_N, M);
    initializeVelocityGrid(velocity_grid, periodic_grid, NUM_N, M);
    // initializeGaussianBlob(velocity_grid,periodic_grid,NUM_N,M,0.5,1);
    memcpy(velocity_grid_next, velocity_grid, NUM_N * M * 2 * sizeof(double));

    // Check for allocation failures
    if (periodic_grid == NULL || velocity_grid == NULL || velocity_grid_next == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string dirName = createTimestampedDirectory();
    // plotPeriodicGrid(periodic_grid, NUM_N, M);
    std::string plot_name("velocity_0000");
    plotVelocityGrid(periodic_grid, velocity_grid, NUM_N, M, PERIODIC_START, PERIODIC_END, plot_name, dirName);

    if (GPU)
    {
        double *d_periodic_grid, *d_vel, *d_vel_A, *d_vel_B, *d_divergence, *d_integrated_bw, *d_integrated_fw;
        double *h_velocity_buffer = (double *)malloc(BUFFER_SIZE * NUM_N * M * 2 * sizeof(double));
        double *d_velocity_buffer;
        CHECK_CUDA(cudaMalloc(&d_velocity_buffer, BUFFER_SIZE * NUM_N * M * 2 * sizeof(double)));

        CHECK_CUDA(cudaMalloc(&d_periodic_grid, NUM_N * M * 2 * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_vel, NUM_N * M * 2 * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_vel_A, NUM_N * M * 2 * sizeof(double)));
        if (MACCORMACK)
            CHECK_CUDA(cudaMalloc(&d_vel_B, NUM_N * M * 2 * sizeof(double)));

        // for maccormack
        CHECK_CUDA(cudaMalloc(&d_integrated_bw, NUM_N * M * 2 * sizeof(double)));
        if (MACCORMACK)
        {
            CHECK_CUDA(cudaMalloc(&d_integrated_fw, NUM_N * M * 2 * sizeof(double)));
        }
        // pressure correction
        double *d_laplace, *d_pressure;
        int *d_lp_cols, *d_lp_row_offsets;
        int nnz = 5 * NUM_N * M;
        CHECK_CUDA(cudaMalloc(&d_divergence, NUM_N * M * sizeof(double)));
        if (SPARSE)
        {
            CHECK_CUDA(cudaMalloc(&d_pressure, NUM_N * M * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_laplace, nnz * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_lp_cols, nnz * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_lp_row_offsets, (NUM_N * M + 1) * sizeof(int)));

            CHECK_CUDA(cudaMemcpy(d_pressure, pressure, NUM_N * M * sizeof(double), cudaMemcpyHostToDevice));
        }
        else
        {
            CHECK_CUDA(cudaMalloc(&d_laplace, NUM_N * M * NUM_N * M * sizeof(double)));
        }

        CHECK_CUDA(cudaMemcpy(d_periodic_grid, periodic_grid, NUM_N * M * 2 * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_vel, velocity_grid, NUM_N * M * 2 * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_vel_A, velocity_grid, NUM_N * M * 2 * sizeof(double), cudaMemcpyHostToDevice));
        if (MACCORMACK)
            CHECK_CUDA(cudaMemcpy(d_vel_B, velocity_grid, NUM_N * M * 2 * sizeof(double), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMemset(d_divergence, 0, NUM_N * M * sizeof(double)))

        auto start_gpu = std::chrono::system_clock::now();
        if (SPARSE)
        {
            int blockDim = TILE_SIZE * TILE_SIZE;
            int gridDim = (nnz + blockDim - 1) / blockDim;
            gpu::constructLaplaceSparseCSR<<<gridDim, blockDim>>>(d_laplace, d_lp_row_offsets, d_lp_cols);
        }
        else
        {
            gpu::constructDiscretizedLaplacian(d_laplace);
        }

        for (int i = 1; i < ITERATIONS + 1; i++)
        {
            gpu::diffuseExplicit(d_vel);

            MACCORMACK ? gpu::advectMacCormack(d_vel, d_vel_A, d_vel_B, d_integrated_fw, d_integrated_bw, d_periodic_grid)
                       : gpu::advectSemiLagrange(d_vel, d_vel_A, d_periodic_grid);
                       //: gpu::advectSemiLagrangeSeparate(d_vel, d_vel_A,d_integrated_bw,d_periodic_grid); //slower

            SPARSE ? gpu::makeIncompressibleSparse(d_vel, d_divergence, d_pressure, d_laplace, d_lp_cols, d_lp_row_offsets)
                   : gpu::makeIncompressible(d_vel, d_divergence, d_laplace);

            // copy result to buffer
            int buffer_index = (i - 1) % BUFFER_SIZE;
            double *d_buffer_ptr = d_velocity_buffer + buffer_index * NUM_N * M * 2;
            CHECK_CUDA(cudaMemcpy(d_buffer_ptr, d_vel, NUM_N * M * 2 * sizeof(double), cudaMemcpyDeviceToDevice));

            // save results when buffer full
            if (i % BUFFER_SIZE == 0)
            {
                CHECK_CUDA(cudaMemcpy(h_velocity_buffer, d_velocity_buffer, BUFFER_SIZE * NUM_N * M * 2 * sizeof(double), cudaMemcpyDeviceToHost));

                for (int j = 0; j < BUFFER_SIZE; j++)
                {
                    std::stringstream plot_name;
                    plot_name << "velocity_" << std::setw(4) << std::setfill('0') << (i - BUFFER_SIZE + j + 1);
                    plotVelocityGrid(periodic_grid, h_velocity_buffer + j * NUM_N * M * 2, NUM_N, M, PERIODIC_START, PERIODIC_END, plot_name.str(), dirName);
                }
            }
        }

        // Copy only the remaining velocities from device to host
        int remaining_iterations = ITERATIONS % BUFFER_SIZE;
        if (remaining_iterations > 0)
        {
            int last_full_batch_iteration = (ITERATIONS / BUFFER_SIZE) * BUFFER_SIZE;
            CHECK_CUDA(cudaMemcpy(h_velocity_buffer, d_velocity_buffer, remaining_iterations * NUM_N * M * 2 * sizeof(double), cudaMemcpyDeviceToHost));

            for (int j = 0; j < remaining_iterations; j++)
            {
                std::stringstream plot_name;
                plot_name << "velocity_" << std::setw(4) << std::setfill('0') << (last_full_batch_iteration + j + 1);
                plotVelocityGrid(periodic_grid, h_velocity_buffer + j * NUM_N * M * 2, NUM_N, M, PERIODIC_START, PERIODIC_END, plot_name.str(), dirName);
            }
        }

        auto end_gpu = std::chrono::system_clock::now();
        std::chrono::duration<double> gpu_seconds = end_gpu - start_gpu;
        std::cout << "gpu time: " << gpu_seconds.count() << "s" << std::endl;
        free(h_velocity_buffer);
        CHECK_CUDA(cudaFree(d_velocity_buffer));
        CHECK_CUDA(cudaFree(d_periodic_grid));
        CHECK_CUDA(cudaFree(d_vel));
        CHECK_CUDA(cudaFree(d_vel_A));
        CHECK_CUDA(cudaFree(d_integrated_bw));
        if (MACCORMACK)
        {
            CHECK_CUDA(cudaFree(d_vel_B));
            CHECK_CUDA(cudaFree(d_integrated_fw));
        }
        CHECK_CUDA(cudaFree(d_laplace));
        CHECK_CUDA(cudaFree(d_divergence));
        if (SPARSE)
        {
            CHECK_CUDA(cudaFree(d_lp_row_offsets));
            CHECK_CUDA(cudaFree(d_lp_cols));
            CHECK_CUDA(cudaFree(d_pressure));
        }
    }
    else
    {

        double *laplace = (double *)malloc(NUM_N * M * NUM_N * M * sizeof(double));
        auto start_cpu = std::chrono::system_clock::now();
        constructDiscretizedLaplacian(laplace);
        auto advect = MACCORMACK ? advectMacCormack : advectSemiLagrange;
        for (int i = 1; i < ITERATIONS + 1; i++)
        {
            diffuseExplicit(velocity_grid, velocity_grid_next);
            advect(velocity_grid, velocity_grid_next, periodic_grid, TIMESTEP, NUM_N, M, DX);
            makeIncompressible(velocity_grid, divergence, pressure, laplace);

            //for ground truth plotting
            // taylorGreenGroundTruth(periodic_grid,velocity_grid_next,i,NUM_N,M);
            // std::swap(velocity_grid,velocity_grid_next);

            std::stringstream plot_name;
            plot_name << "velocity_" << std::setw(4) << std::setfill('0') << i;
            plotVelocityGrid(periodic_grid, velocity_grid, NUM_N, M, PERIODIC_START, PERIODIC_END, plot_name.str(), dirName);
            plot_name.str("");
        }
        auto end_cpu = std::chrono::system_clock::now();
        std::chrono::duration<double> gpu_seconds = end_cpu - start_cpu;
        std::cout << "cpu time: " << gpu_seconds.count() << "s" << std::endl;
        free(laplace);
    }
    std::cout << "Creating velocity animation" << std::endl;
    createGifFromPngs(dirName, "animation_velocity.gif", PERIODIC_START, PERIODIC_END);
    plotErrors("plots/_ground_truth", dirName);

    free(periodic_grid);
    free(velocity_grid);
    free(velocity_grid_next);
    free(divergence);
    free(pressure);
}
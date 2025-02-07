#include "pressure_correction.cuh"
#include <cassert>

void test_divergence()
{
    //int n = NUM_N;
    int n = 4;
    double *velocity_grid = (double *)malloc(n * n * 2 * sizeof(double));
    double *divergence = (double *)malloc(n * n * sizeof(double));
    for (int y_i = 0; y_i < n; y_i++)
    {
        for (int i = 1; i < 2*n; i+=2)
        {
            velocity_grid[y_i * (2*n) + i - 1] = 1.0; //u component 
            velocity_grid[y_i * (2*n) + i] = 1.0;
        }
    }
    calculateDivergence(velocity_grid,divergence,n,n);
    for (int i = 0; i < n*n; i++)
    {
        assert(divergence[i] == 0.0);
    }
    std::cout << "CPU Divergence is correct" << std::endl;
    double *d_div, *d_vel;
    double *h_div = (double*) malloc(n*n * sizeof(double));
    CHECK_CUDA(cudaMalloc(&d_div, n*n* sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_vel, n*n*2* sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_vel, velocity_grid,n*n*2* sizeof(double) , cudaMemcpyHostToDevice));
    //gpu test
    dim3 blockDim(TILE_SIZE,TILE_SIZE);
    dim3 gridDim((n+ TILE_SIZE-1)/TILE_SIZE,(n+ TILE_SIZE-1)/TILE_SIZE); 
    gpu::calculateDivergence<<<gridDim,blockDim>>>(d_vel,d_div,n,n);

    CHECK_CUDA(cudaMemcpy(h_div, d_div,n*n* sizeof(double) , cudaMemcpyDeviceToHost));
    assert(all_close(h_div,divergence,n,n));
    std::cout << "CPU Divergence is identical to GPU divergence" << std::endl;

    CHECK_CUDA(cudaFree(d_div));
    CHECK_CUDA(cudaFree(d_vel));
    free(velocity_grid);
    free(divergence);
}

void test_make_incompressible()
{
    int n = 8;
    int dx = 1.0;
    double *velocity_grid = (double *)malloc(n * n * 2 * sizeof(double));
    double *velocity_grid_copy = (double *)malloc(n * n * 2 * sizeof(double));
    double *divergence = (double *)malloc(n * n * sizeof(double));
    double *pressure = (double *)malloc(n * n * sizeof(double));
    for (int y_i = 0; y_i < n; y_i++)
    {
        for (int i = 1; i < 2*n; i+=2)
        {
            velocity_grid[y_i * (2*n) + i - 1] = 1.0; //u component 
            velocity_grid[y_i * (2*n) + i] = 1.0;
        }
    }
    double testval = velocity_grid[periodic_linear_Idx(-2,0,2*n,n)];

    std::cout << "test_val" << testval << std::endl;
    for (int y_i = 0; y_i < n; y_i++)
    {
        for (int i = 0; i < n; i++)
        {
            divergence[y_i * n + i] = 0.0;
            pressure[y_i * n + i] = 0.0;
        }
    }
    int u_i = 4;
    int v_i = u_i + 1;
    int y_i = 4;
    // middle

    velocity_grid[periodic_linear_Idx(u_i,y_i,2*n,n)] = 0.0; 
    velocity_grid[periodic_linear_Idx(v_i,y_i,2*n,n)] = 0.0; 
    //right
    velocity_grid[periodic_linear_Idx(u_i+2,y_i,2*n,n)] = 3.0; 
    velocity_grid[periodic_linear_Idx(v_i+2,y_i,2*n,n)] = 0.0; 
    //left
    velocity_grid[periodic_linear_Idx(u_i-2,y_i,2*n,n)] = -3.0; 
    velocity_grid[periodic_linear_Idx(v_i-2,y_i,2*n,n)] = 0.0; 
    //up
    velocity_grid[periodic_linear_Idx(u_i,y_i+1,2*n,n)] = 0.0; 
    velocity_grid[periodic_linear_Idx(v_i,y_i+1,2*n,n)] = -3.0; 
    //down
    velocity_grid[periodic_linear_Idx(u_i,y_i-1,2*n,n)] = 0.0; 
    velocity_grid[periodic_linear_Idx(v_i,y_i-1,2*n,n)] = 3.0; 

    memcpy(velocity_grid_copy,velocity_grid,n*n*2*sizeof(double));

    std::cout << "velocity" << std::endl;
    print_matrix_row_major(n,2*n,velocity_grid,2*n);

    calculateDivergence(velocity_grid,divergence,n,n,dx);
    std::cout << "divergence" <<std::endl;
    print_matrix_row_major(n,n,divergence,n);

    makeIncompressible(velocity_grid,divergence,pressure,n,n,dx);
    std::cout << "CPU velocity after correction" << std::endl;
    print_matrix_row_major(n,2*n,velocity_grid,2*n);
    assert(!all_close(velocity_grid,velocity_grid_copy,2*n,n));
    //GPU
    
    double *d_div, *d_vel, *d_lp;
    double *h_vel= (double*) malloc(n*n * 2 * sizeof(double));
    CHECK_CUDA(cudaMalloc(&d_div, n*n* sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_vel, n*n*2* sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_lp, n*n*n*n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_vel, velocity_grid_copy,n*n*2* sizeof(double) , cudaMemcpyHostToDevice));

    //GPU
    gpu::constructDiscretizedLaplacian(d_lp,n,dx);
    //calculateDivergence(velocity_grid,divergence,n,n,dx);
    //std::cout << "divergence" << std::endl;
    //print_matrix_row_major(n,n,divergence,n);

    //gpu test
    CHECK_CUDA(cudaMemset(d_div,0,n*n*sizeof(double)));
    gpu::makeIncompressible(d_vel,d_div,d_lp,n,n);
    CHECK_CUDA(cudaMemcpy(h_vel, d_vel,n*n*2* sizeof(double) , cudaMemcpyDeviceToHost));
    std::cout << "GPU velocity after correction" << std::endl;
    print_matrix_row_major(n,2*n,h_vel,2*n);
    assert(!all_close(h_vel,velocity_grid_copy,2*n,n));
    assert(all_close(h_vel,velocity_grid,2*n,n));
    //std::cout << "CPU corrected velocity is identical to GPU corrected velocity" << std::endl;
    

    CHECK_CUDA(cudaFree(d_div));
    CHECK_CUDA(cudaFree(d_vel));
    CHECK_CUDA(cudaFree(d_lp));
    free(h_vel);
    
    free(divergence);
    free(pressure);
    free(velocity_grid);
}

void test_correct_velocity()
{

    int n = 2;
    double dx = 1.0;

    double velocity_grid_gpu[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; 
    double velocity_grid_cpu[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; 
    double velocity_grid_correct[8] = {0.0, -1.0, 2.0, -1.0, 0.0, 3.0, 2.0, 3.0}; 
    double pressure[4] = {1.0, 2.0, 3.0, 4.0}; 

    std::cout << "corrected velocity cpu"<<std::endl;
    correct_velocity(velocity_grid_cpu,pressure,n,n,dx);
    print_matrix_row_major(n,2*n,velocity_grid_cpu,2*n);
    assert(all_close(velocity_grid_cpu,velocity_grid_correct,2*n,n));
    // Allocate device memory
    double *d_velocity_grid, *d_pressure;
    CHECK_CUDA(cudaMalloc(&d_velocity_grid, n*n* 2* sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_pressure, n*n * sizeof(double)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_velocity_grid, velocity_grid_gpu, n*n*2 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pressure, pressure, n*n * sizeof(double), cudaMemcpyHostToDevice));

    // Launch the kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(1, 1);
    gpu::correct_velocity<<<gridDim, blockDim>>>(d_velocity_grid, d_pressure, n, n, dx);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(velocity_grid_gpu, d_velocity_grid, 8 * sizeof(double), cudaMemcpyDeviceToHost));

    // Verify the results
    std::cout<< "corrected velocity gpu: " << std::endl;
    print_matrix_row_major(n,2*n,velocity_grid_gpu,2*n);
    assert(all_close(velocity_grid_gpu,velocity_grid_correct,2*n,n));
    std::cout << "velocity correction test passed!" << std::endl;

    // Free memory
    CHECK_CUDA(cudaFree(d_velocity_grid));
    CHECK_CUDA(cudaFree(d_pressure));
}

void test_laplace()
{   
    double dx = 1.0; 
    int n = 4;
    double * lp = (double *)malloc(n*n*n*n * sizeof(double));
    if (lp == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
    constructDiscretizedLaplacian(lp,n,dx);

    for (int i = 0; i < n*n; i++)
    {
        double row_sum = 0.0;
        for (int j = 0; j < n*n; j++)
        {
            row_sum += lp[i*n*n+j];
        }
        assert(is_close(row_sum,0.0));
    }
    std::cout << "Laplacian has correct row sums" << std::endl;
    std::cout << "Laplacian" << std::endl;
    print_matrix_row_major(n*n, n*n, lp, n*n);

    //gpu
    double * d_lp;
    double *h_lp = (double*) malloc(n*n*n*n * sizeof(double));
    CHECK_CUDA(cudaMalloc(&d_lp, n*n*n*n * sizeof(double)));
    gpu::constructDiscretizedLaplacian(d_lp,n,dx);
    CHECK_CUDA(cudaMemcpy(h_lp, d_lp,n*n*n*n * sizeof(double) , cudaMemcpyDeviceToHost));
    std::cout << "gpu laplacian:" << std::endl;
    print_matrix_row_major(n*n,n*n,h_lp,n*n);

    assert(all_close(lp,h_lp,n*n,n*n));
    

    CHECK_CUDA(cudaFree(d_lp));
    std::cout << "GPU Laplacian is identical to CPU Laplacian" << std::endl;
    free(lp);
    free(h_lp);
}




int main()
{
    test_divergence();
    test_laplace();
    test_correct_velocity();
    test_make_incompressible();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
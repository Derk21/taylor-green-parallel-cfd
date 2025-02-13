#include "pressure_correction.cuh"
#include <cassert>

void test_divergence()
{
    //int n = NUM_N;
    int n = 8;
    double dx = 1.0;
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
    dim3 blockDim(4,4);
    dim3 gridDim((n+ blockDim.x-1)/blockDim.x,(n+ blockDim.y-1)/blockDim.y); 
    gpu::calculateDivergence<<<gridDim,blockDim>>>(d_vel,d_div,n,n,dx);
    CHECK_CUDA(cudaMemcpy(h_div, d_div,n*n* sizeof(double) , cudaMemcpyDeviceToHost));
    std::cout << "CPU Divergence" << std::endl;
    print_matrix_row_major(n,n,divergence,n);
    std::cout << "GPU Divergence" << std::endl;
    print_matrix_row_major(n,n,h_div,n);
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
    double dx = 1.0;
    double *velocity_grid = (double *)malloc(n * n * 2 * sizeof(double));
    double *velocity_grid_copy = (double *)malloc(n * n * 2 * sizeof(double));
    double *divergence = (double *)malloc(n * n * sizeof(double));
    double *pressure = (double *)malloc(n * n * sizeof(double));
    for (int y_i = 0; y_i < n; y_i++)
    {
        for (int i = 0; i < n; i++)
        {
            velocity_grid[y_i * (2*n) + (2*i)] = 1.0; //u component 
            velocity_grid[y_i * (2*n) + (2*i+1)] = 1.0;
            divergence[y_i * n + i] = 0.0;
            pressure[y_i * n + i] = 0.0;
        }
    }
    //double testval = velocity_grid[periodic_linear_Idx(-2,0,2*n,n)];
    //std::cout << "test_val " << testval << std::endl;
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


    //GPU
    
    double *d_div, *d_vel, *d_lp;
    double *h_vel= (double*) malloc(n*n*2*sizeof(double));
    double *h_div= (double*) malloc(n*n*sizeof(double));
    CHECK_CUDA(cudaMalloc(&d_div, n*n* sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_vel, n*n*2* sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_lp, n*n*n*n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_vel, velocity_grid,n*n*2* sizeof(double) , cudaMemcpyHostToDevice));

    
    //gpu test
    gpu::constructDiscretizedLaplacian(d_lp,n,dx);
    CHECK_CUDA(cudaMemcpy(d_div,divergence,n*n*sizeof(double),cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaDeviceSynchronize());

    gpu::makeIncompressible(d_vel,d_div,d_lp,n,n,dx);

    CHECK_CUDA(cudaMemcpy(h_vel, d_vel,n*n*2*sizeof(double) , cudaMemcpyDeviceToHost));
    std::cout << "GPU velocity after correction" << std::endl;
    print_matrix_row_major(n,2*n,h_vel,2*n);
    //assert(!all_close(h_vel,velocity_grid_copy,2*n,n));
    //assert(all_close(h_vel,velocity_grid,2*n,n));
    //std::cout << "CPU corrected velocity is identical to GPU corrected velocity" << std::endl;
    std::cout << "CPU velocity after correction" << std::endl;

    CHECK_CUDA(cudaFree(d_div));
    CHECK_CUDA(cudaFree(d_vel));
    CHECK_CUDA(cudaFree(d_lp));

    makeIncompressible(velocity_grid_copy,divergence,pressure,n,n,dx);
    print_matrix_row_major(n,2*n,velocity_grid_copy,2*n);
    assert(all_close(h_vel,velocity_grid_copy,2*n,n));

    free(h_vel);
    free(h_div);
    free(divergence);
    free(pressure);
    free(velocity_grid);
    free(velocity_grid_copy);
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
    free(h_lp);

    //SPARSE
    double *values = (double*)malloc(5*n*n*sizeof(double));
    int *col_indices = (int*)malloc(5*n*n*sizeof(int));
    int *row_offsets = (int*)malloc((n*n+1)*sizeof(int));
    std::cout << "sparse Laplacian cpu" << std::endl;
    constructLaplaceSparseCSR(values,row_offsets,col_indices,n,dx);

    printCSR(row_offsets,col_indices,values,n*n);

    //SPARSE GPU
    double *h_values= (double*)malloc(5*n*n*sizeof(double));
    int *h_columns= (int*)malloc(5*n*n*sizeof(int));
    int *h_row_offsets = (int*)malloc((n*n+1)*sizeof(int));

    double * d_values;
    int* d_columns, *d_row_offsets;
    int nnz= 5*n*n;
    CHECK_CUDA(cudaMalloc(&d_values,nnz*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_columns,nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_row_offsets,(n*n+1)*sizeof(int)));

    int numThreads = 16; 
    int numBlocks = (nnz+ numThreads - 1) / numThreads;
    gpu::constructLaplaceSparseCSR<<<numThreads,numBlocks>>>(d_values,d_row_offsets,d_columns,n,dx); 

    CHECK_CUDA(cudaMemcpy(h_values, d_values, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_columns, d_columns, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_row_offsets, d_row_offsets, (n*n+1) * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "sparse Laplacian gpu" << std::endl;
    printCSR(h_row_offsets,h_columns,h_values,n*n);

    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_columns));
    CHECK_CUDA(cudaFree(d_row_offsets));

    free(h_values);
    free(h_columns);
    free(h_row_offsets);
    
    free(values);
    free(col_indices);
    free(row_offsets);
    free(lp);
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
#include "pressure_correction.cuh"
#include <cassert>

void test_divergence()
{
    int n = NUM_N;
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
    dim3 gridDim((NUM_N+ TILE_SIZE-1)/TILE_SIZE,(NUM_N + TILE_SIZE-1)/TILE_SIZE); 
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
    int n = 5;
    int dx = 1.0;
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

    for (int y_i = 0; y_i < n; y_i++)
    {
        for (int i = 0; i < n; i++)
        {
            divergence[y_i * n + i] = 0.0;
        }
    }

    velocity_grid[2 * (2*n) + 2 - 1] = -1.0; //u component 
    velocity_grid[2 * (2*n) + 2] = -1.0;

    calculateDivergence(velocity_grid,divergence,n,n);
    std::cout << "divergence" << std::endl;
    print_matrix_row_major(n,n,divergence,n);
    

    free(velocity_grid);
    free(divergence);

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
    test_make_incompressible();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
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
    print_matrix(2*n, n, velocity_grid, 2*n);
    calculateDivergence(velocity_grid,divergence,n,n);
    print_matrix(n, n, divergence, n);
    for (int i = 0; i < n*n; i++)
    {
        assert(divergence[i] == 0.0);
    }
    free(velocity_grid);
    free(divergence);
    std::cout << "CPU Divergence is correct" << std::endl;
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
    print_matrix(n*n, n*n, lp, n*n);

    //gpu
    double * d_lp;
    double *h_lp = (double*) malloc(n*n*n*n * sizeof(double));
    CHECK_CUDA(cudaMalloc(&d_lp, n*n*n*n * sizeof(double)));
    dim3 blockDim(4);
    dim3 gridDim((n*n + 4 -1) / 4);
    gpu::constructDiscretizedLaplacian<<<gridDim,blockDim>>>(d_lp,n*n,dx);
    CHECK_CUDA(cudaMemcpy(h_lp, d_lp,n*n*n*n * sizeof(double) , cudaMemcpyDeviceToHost));

    for (int i = 0; i < n*n; i++)
    {
        for (int j = 0; j < n*n; j++)
        {
            assert(is_close(lp[i*n*n+j],h_lp[i*n*n+j]));
        }
    }

    CHECK_CUDA(cudaFree(d_lp));
    std::cout << "GPU Laplacian is identical to CPU Laplacian" << std::endl;
    free(lp);
    free(h_lp);
}



int main()
{
    test_divergence();
    test_laplace();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
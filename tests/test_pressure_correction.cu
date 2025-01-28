#include "pressure_correction.h"
#include <cassert>

void test_divergence(){
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
    std::cout << "Divergence is correct" << std::endl;
}

void test_laplace(){ 
    int n = 4;
    double * lp = (double *)malloc(n*n*n*n * sizeof(double));
    if (lp == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
    constructDiscretizedLaplacian(lp,n);

    for (int i = 0; i < n*n; i++)
    {
        double row_sum = 0.0;
        for (int j = 0; j < n*n; j++)
        {
            row_sum += lp[i*n*n+j];
        }
        assert(round(row_sum) == 0.0);
    }
    std::cout << "Laplacian has correct row sums" << std::endl;
    std::cout << "Laplacian" << std::endl;
    print_matrix(n*n, n*n, lp, n*n);
    free(lp);
}

int main() {
    test_divergence();
    test_laplace();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
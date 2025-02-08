#include "constants.cuh"
#include "diffuse.cuh"
//#include "plotting.cuh"
#include "utils.cuh"
#include "init.cuh"
#include <string>
#include <iostream>
#include <cassert>

void test_diffuseExplicitStep()
{
    int n = 8;
    int m = 8;
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    double *periodic_grid = (double*)malloc(sizeof(double) * 2 * n * m);
    double *velocity_grid = (double*)malloc(sizeof(double) * 2 * n * m);
    double *velocity_grid_next = (double*)malloc(sizeof(double) * 2 * n * m);
    initializePeriodicGrid(periodic_grid,n,m);
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < 2 * n; i += 2)
        {
            int u_i = i - 1;
            int v_i = i;
            velocity_grid[y_i * (2*n) + u_i] = 0.0;
            velocity_grid[y_i * (2*n) + v_i] = 0.0;
            //velocity_grid[periodic_linear_Idx(u_i, y_i,2*n,m)] = 1.0;
            //velocity_grid[periodic_linear_Idx(v_i, y_i,2*n,m)] = 0.0;

            velocity_grid_next[periodic_linear_Idx(u_i, y_i,2*n,m)] = 0.0;
            velocity_grid_next[periodic_linear_Idx(v_i, y_i,2*n,m)] = 0.0;
        }
    }
    std::string dirname = "test_output/";
    //int y_i_half = m / 2;
    //int i_half = 2*n / 2;
    //velocity_grid[y_i_half * (2*n) + i_half] = 1.0;
    velocity_grid[(m-1) * (2*n) + (2*(n-1))]= 1.0;

    std::cout << "before diffusion:" << std::endl;
    print_matrix_row_major(m,2*n, velocity_grid,2*n);
    std::string filename = "velocity_grid_init.png";
    //plotVelocityGrid(periodic_grid,velocity_grid, n, m,PERIODIC_START,PERIODIC_END, filename,dirname);

    //cuda inits
    double *d_vel=NULL;
    double *h_vel= (double*) malloc(n*n * 2 * sizeof(double));

    CHECK_CUDA(cudaMalloc((void**)&d_vel, n*n*2* sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_vel, velocity_grid,n*n*2* sizeof(double) , cudaMemcpyHostToDevice));

    diffuseExplicit(velocity_grid, velocity_grid_next, n, m,dx);
    filename = "velocity_grid_diffused_test.png";
    //plotVelocityGrid(periodic_grid,velocity_grid, n, m,PERIODIC_START,PERIODIC_END, filename,dirname);
    std::cout << "after diffusion:" << std::endl;
    print_matrix_row_major(m,2*n, velocity_grid_next,2*n);

    //gpu


    gpu::diffuseExplicit(d_vel,n,m,dx);
    CHECK_CUDA(cudaMemcpy(h_vel, d_vel,n*n*2* sizeof(double) , cudaMemcpyDeviceToHost));

    std::cout << "after diffusion gpu:" << std::endl;
    print_matrix_row_major(m,2*n, h_vel,2*n);

    assert(all_close(h_vel,velocity_grid,2*n,m));
    std::cout<< "gpu cpu diffusion identical"<< std::endl;

    CHECK_CUDA(cudaFree(d_vel));
    free(h_vel);
    free(periodic_grid);
    free(velocity_grid);
    free(velocity_grid_next);
    std::cout << "diffusion plotted!" << std::endl;
}

int main(){
    test_diffuseExplicitStep();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
#include "diffuse.cuh"

void diffuseExplicit(double *velocity_grid, double *velocity_grid_next, const int n, const int m,const double dx)
{
    const double amount = DIFFUSIVITY * (TIMESTEP / SUBSTEPS_EXPLICIT);
    //TODO: add guard for breaking cfl condition
    for(int i = 0; i < SUBSTEPS_EXPLICIT; i++)
    {
        diffuseExplicitStep(velocity_grid, velocity_grid_next, amount,n,m,dx);
        memcpy(velocity_grid, velocity_grid_next, sizeof(double) * 2 * n * m);
        //std::cout << "diffusion step:" << i << std::endl;
        //print_matrix_row_major(2*n,m, velocity_grid,2*n);
    }
}

void diffuseExplicitStep(const double *velocity_grid, double *velocity_grid_next, double amount, const int n, const int m,const double dx)
{
    //double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    //double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 0; i < n; i++)
        {
            int u_i = 2*i;
            int v_i = u_i + 1;

            double u = velocity_grid[periodic_linear_Idx(u_i, y_i,2*n,m)];
            double v = velocity_grid[periodic_linear_Idx(v_i, y_i,2*n,m)];

            double u_left = velocity_grid[periodic_linear_Idx(u_i - 2, y_i,2*n,m)];
            double u_right = velocity_grid[periodic_linear_Idx(u_i + 2, y_i,2*n,m)];
            double u_up = velocity_grid[periodic_linear_Idx(u_i, y_i + 1,2*n,m)];
            double u_down = velocity_grid[periodic_linear_Idx(u_i, y_i - 1,2*n,m)];

            double v_left = velocity_grid[periodic_linear_Idx(v_i - 2, y_i,2*n,m)];
            double v_right = velocity_grid[periodic_linear_Idx(v_i + 2, y_i,2*n,m)];
            double v_up = velocity_grid[periodic_linear_Idx(v_i, y_i + 1,2*n,m)];
            double v_down = velocity_grid[periodic_linear_Idx(v_i, y_i - 1,2*n,m)];

            double u_diffusion = (u_right - 2 * u + u_left) / (dx * dx) + (u_up - 2 * u + u_down) / (dx * dx);
            double v_diffusion = (v_right - 2 * v + v_left) / (dx * dx) + (v_up - 2 * v + v_down) / (dx * dx);

            velocity_grid_next[periodic_linear_Idx(u_i, y_i,2*n,m)] = u + amount * u_diffusion;
            velocity_grid_next[periodic_linear_Idx(v_i, y_i,2*n,m)] = v + amount * v_diffusion;
        }
    }
}

namespace gpu
{

void diffuseExplicit(double *velocity_grid, const int n, const int m,const double dx)
{
    const double amount = DIFFUSIVITY * (TIMESTEP / SUBSTEPS_EXPLICIT);
    //TODO: add guard for breaking cfl condition

    dim3 blockDim(TILE_SIZE,TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE-1)/TILE_SIZE,(n+ TILE_SIZE-1)/TILE_SIZE); 

    for(int i = 0; i < SUBSTEPS_EXPLICIT; i++)
    {
        gpu::diffuseExplicitStep<<<gridDim,blockDim>>>(velocity_grid, amount,n,m,dx);
        //std::cout << "diffusion step:" << i << std::endl;
        //print_matrix_row_major(2*n,m, velocity_grid,2*n);
    }
}
    
__global__ void diffuseExplicitStep(double *velocity_grid,  const double amount,const int n, const int m,const double dx)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x; 
    int row = threadIdx.y + blockIdx.y * blockDim.y; 
    double tmp_u, tmp_v = 0.0;
    int u_i = col * 2;
    int v_i = (col * 2) + 1;
    if (row < m && col < n)
    {

        double u = velocity_grid[periodic_linear_Idx(u_i, row,2*n,m)];
        double v = velocity_grid[periodic_linear_Idx(v_i, row)];

        double u_left = velocity_grid[periodic_linear_Idx(u_i - 2, row,2*n,m)];
        double u_right = velocity_grid[periodic_linear_Idx(u_i + 2, row,2*n,m)];
        double u_up = velocity_grid[periodic_linear_Idx(u_i, row + 1,2*n,m)];
        double u_down = velocity_grid[periodic_linear_Idx(u_i, row - 1,2*n,m)];

        double v_left = velocity_grid[periodic_linear_Idx(v_i - 2, row,2*n,m)];
        double v_right = velocity_grid[periodic_linear_Idx(v_i + 2, row,2*n,m)];
        double v_up = velocity_grid[periodic_linear_Idx(v_i, row + 1,2*n,m)];
        double v_down = velocity_grid[periodic_linear_Idx(v_i, row - 1,2*n,m)];

        double u_diffusion = (u_right - 2 * u + u_left) / (dx * dx) + (u_up - 2 * u + u_down) / (dx * dx);
        double v_diffusion = (v_right - 2 * v + v_left) / (dx * dx) + (v_up - 2 * v + v_down) / (dx * dx);

        tmp_u = u + amount * u_diffusion;
        tmp_v = v + amount * v_diffusion;
    }
    __syncthreads();
    if (row < m && col < n)
    {
        velocity_grid[periodic_linear_Idx(u_i, row,2*n,m)] = tmp_u;
        velocity_grid[periodic_linear_Idx(v_i, row,2*n,m)] = tmp_v;
    }
}
}



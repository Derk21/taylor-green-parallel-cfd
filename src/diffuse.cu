#include "diffuse.cuh"

void diffuseExplicitStep(const double *velocity_grid, double *velocity_grid_next, double amount,int n, int m)
{
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    int nn = 2 * n;
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < nn; i += 2)
        {
            int u_i = i - 1;
            int v_i = i;

            double u = velocity_grid[periodic_linear_Idx(u_i, y_i)];
            double v = velocity_grid[periodic_linear_Idx(v_i, y_i)];

            double u_left = velocity_grid[periodic_linear_Idx(u_i - 2, y_i)];
            double u_right = velocity_grid[periodic_linear_Idx(u_i + 2, y_i)];
            double u_up = velocity_grid[periodic_linear_Idx(u_i, y_i + 1)];
            double u_down = velocity_grid[periodic_linear_Idx(u_i, y_i - 1)];

            double v_left = velocity_grid[periodic_linear_Idx(v_i - 2, y_i)];
            double v_right = velocity_grid[periodic_linear_Idx(v_i + 2, y_i)];
            double v_up = velocity_grid[periodic_linear_Idx(v_i, y_i + 1)];
            double v_down = velocity_grid[periodic_linear_Idx(v_i, y_i - 1)];

            double u_diffusion = (u_right - 2 * u + u_left) / (dx * dx) + (u_up - 2 * u + u_down) / (dy * dy);
            double v_diffusion = (v_right - 2 * v + v_left) / (dx * dx) + (v_up - 2 * v + v_down) / (dy * dy);

            velocity_grid_next[periodic_linear_Idx(u_i, y_i)] = u + amount * u_diffusion;
            velocity_grid_next[periodic_linear_Idx(v_i, y_i)] = v + amount * v_diffusion;
        }
    }
}

void diffuseExplicit(double *velocity_grid, double *velocity_grid_next, int n, int m)
{
    double amount = DIFFUSIVITY * (TIMESTEP / SUBSTEPS_EXPLICIT);
    //TODO: add guard for breaking cfl condition
    for(int i = 0; i < SUBSTEPS_EXPLICIT; i++)
    {
        diffuseExplicitStep(velocity_grid, velocity_grid_next, amount);
        memcpy(velocity_grid, velocity_grid_next, sizeof(double) * 2 * n * m);
        //std::cout << "diffusion step:" << i << std::endl;
        //print_matrix_row_major(2*n,m, velocity_grid,2*n);
    }
}

void diffuseImplicit(double *velocity_grid, double *velocity_grid_next, int n, int m){
   int nn = 2 * n; 
   std::cout << "Diffuse Implicit" << nn << std::endl;
}
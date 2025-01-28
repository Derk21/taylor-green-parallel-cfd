#include "advect.h"

void interpolateVelocity(double x_d, double y_d, int n, int m, const double *periodic_grid, double *velocity_grid, double * velocity_grid_next)
{
    int u_i_closest, v_i_closest, y_i_closest;
    setClosestGridPointIdx(x_d, y_d, n, m, v_i_closest, y_i_closest);
    u_i_closest = v_i_closest - 1;

    // interpolation weights
    double x_closest = periodic_grid[periodic_linear_Idx(u_i_closest, y_i_closest)];
    double y_closest = periodic_grid[periodic_linear_Idx(v_i_closest, y_i_closest)];
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    double x_diff = (x_d - x_closest) / dx;
    double y_diff = (y_d - y_closest) / dy;

    // forward bilinear interpolation
    // containing grid cell
    double u = (1.0 - x_diff) * (1.0 - y_diff) * velocity_grid[periodic_linear_Idx(u_i_closest, y_i_closest)];
    double v = (1.0 - x_diff) * (1.0 - y_diff) * velocity_grid[periodic_linear_Idx(v_i_closest, y_i_closest)];
    // x_direction next grid cell
    u += (1.0 - x_diff) * y_diff * velocity_grid[periodic_linear_Idx(u_i_closest, y_i_closest + 1)];
    v += (1.0 - x_diff) * y_diff * velocity_grid[periodic_linear_Idx(v_i_closest, y_i_closest + 1)];
    // y_direction next grid cell
    u += x_diff * (1.0 - y_diff) * velocity_grid[periodic_linear_Idx(u_i_closest + 2, y_i_closest)];
    v += x_diff * (1.0 - y_diff) * velocity_grid[periodic_linear_Idx(v_i_closest + 2, y_i_closest)];
    // next grid cell in diagonal direction 
    u +=  x_diff *  y_diff * velocity_grid[periodic_linear_Idx(u_i_closest + 2, y_i_closest + 1)];
    v +=  x_diff *  y_diff * velocity_grid[periodic_linear_Idx(v_i_closest + 2, y_i_closest + 1)];

    // assign to closest grid point
    velocity_grid_next[periodic_linear_Idx(u_i_closest, y_i_closest)] = u;
    velocity_grid_next[periodic_linear_Idx(v_i_closest, y_i_closest)] = v;
}

void integrateEuler(double *velocity_grid, int &u_i, int &y_i, int &v_i, const double *periodic_grid, double &x_d,  double &y_d,const double dt,int n=NUM_N, int m=M)
{
    double u_old = velocity_grid[periodic_linear_Idx(u_i, y_i)];
    double v_old = velocity_grid[periodic_linear_Idx(v_i, y_i)];

    double x = periodic_grid[periodic_linear_Idx(u_i, y_i)];
    double y = periodic_grid[periodic_linear_Idx(v_i, y_i)];

    x_d = fmod(x + dt * u_old+PERIODIC_END,PERIODIC_END);
    y_d = fmod(y + dt * v_old+PERIODIC_END,PERIODIC_END);
} 

void advectSemiLagrange(double *velocity_grid, double *velocity_grid_next, const double *periodic_grid, const double dt, int n, int m)
{
    int nn = 2 * n;
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < nn; i+=2)
        {   
            int u_i = i-1;
            int v_i = i;
            double x_d, y_d;
            //backward euler -dt
            integrateEuler(velocity_grid, u_i, y_i, v_i, periodic_grid, x_d, y_d,-dt);
            //interpolate 
            interpolateVelocity(x_d, y_d, n, m, periodic_grid, velocity_grid,velocity_grid_next);
        }
    } 
    memcpy(velocity_grid, velocity_grid_next, 2 * n * m * sizeof(double));
}
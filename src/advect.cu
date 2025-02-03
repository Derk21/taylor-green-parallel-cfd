#include "advect.h"
#include <cassert>

void interpolateVelocity(double x_d, double y_d, int n, int m, double periodic_start,double periodic_end, const double *periodic_grid, const double *velocity_grid, double * velocity_grid_next)
{
    int u_i_closest, v_i_closest, y_i_closest;
    setClosestGridPointIdx(x_d, y_d, n, m, v_i_closest, y_i_closest);
    u_i_closest = v_i_closest - 1;

    // interpolation weights
    double x_closest = periodic_grid[periodic_linear_Idx(u_i_closest, y_i_closest,2*n,m)];
    double y_closest = periodic_grid[periodic_linear_Idx(v_i_closest, y_i_closest,2*n,m)];
    double dx = (periodic_end - periodic_start) / (n - 1);
    double dy = (periodic_end - periodic_start ) / (m - 1);
    double x_diff = (x_d - x_closest) / dx;
    double y_diff = (y_d - y_closest) / dy;

    // forward bilinear interpolation
    double u = get_interpolated(u_i_closest,y_i_closest,x_diff,y_diff,velocity_grid,n,m);
    double v = get_interpolated(v_i_closest,y_i_closest,x_diff,y_diff,velocity_grid,n,m);

    // assign to closest grid point
    velocity_grid_next[periodic_linear_Idx(u_i_closest, y_i_closest,2*n,m)] = u;
    velocity_grid_next[periodic_linear_Idx(v_i_closest, y_i_closest,2*n,m)] = v;
}

double get_interpolated(const int &i_closest, const int & y_i_closest,const double &x_diff, const double &y_diff,const double * velocity_grid,int n, int m)
{
    double weight_contained = (1.0 - x_diff) * (1.0 - y_diff);
    double weight_next_x = x_diff * (1.0 - y_diff);
    double weight_next_x_y = x_diff * y_diff;
    double weight_next_y = (1.0 - x_diff) * y_diff;

    double sum_weights = weight_contained + weight_next_x + weight_next_y + weight_next_x_y;
    assert(is_close(sum_weights,1.0));
    double val = weight_contained * velocity_grid[periodic_linear_Idx(i_closest, y_i_closest,2*n,m)];
    // y_direction next grid cell
    val += weight_next_y * velocity_grid[periodic_linear_Idx(i_closest, y_i_closest + 1,2*n,m)];
    // x_direction next grid cell
    val += weight_next_x * velocity_grid[periodic_linear_Idx(i_closest + 2, y_i_closest,2*n,m)];
    // next grid cell in diagonal direction 
    val += weight_next_x_y * velocity_grid[periodic_linear_Idx(i_closest + 2, y_i_closest + 1,2*n,m)];
    return val;
}

void integrateEuler(const double *velocity_grid, int &y_i, int &u_i,  int &v_i, const double *periodic_grid, double &x_d,  double &y_d,const double dt,int n=NUM_N, int m=M)
{
    double u_old = velocity_grid[periodic_linear_Idx(u_i, y_i)];
    double v_old = velocity_grid[periodic_linear_Idx(v_i, y_i)];

    double x = periodic_grid[periodic_linear_Idx(u_i, y_i)];
    double y = periodic_grid[periodic_linear_Idx(v_i, y_i)];

    x_d = fmod(x + dt * u_old+PERIODIC_END,PERIODIC_END)+PERIODIC_START;
    y_d = fmod(y + dt * v_old+PERIODIC_END,PERIODIC_END)+PERIODIC_START;
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
            //backward euler
            integrateEuler(velocity_grid,y_i, u_i,  v_i, periodic_grid, x_d, y_d,-dt);
            interpolateVelocity(x_d, y_d, n, m,PERIODIC_START,PERIODIC_END, periodic_grid, velocity_grid,velocity_grid_next);
        }
    } 
    memcpy(velocity_grid, velocity_grid_next, 2 * n * m * sizeof(double));
}

void advectMacCormack(double *velocity_grid, double *velocity_grid_next, const double *periodic_grid, const double dt, int n, int m)
{
    double *velocity_grid_next_2 = (double *)malloc(n * m * 2 * sizeof(double));
    memcpy(velocity_grid_next_2,velocity_grid, 2*n*m*sizeof(double));

    int nn = 2 * n;
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < nn; i+=2)
        {   
            int u_i = i-1;
            int v_i = i;
            double x_backward_d, y_backward_d, x_forward_d, y_forward_d;
            //backward euler -dt
            integrateEuler(velocity_grid,y_i, u_i,  v_i, periodic_grid, x_backward_d, y_backward_d,-dt);
            //forward euler +dt
            integrateEuler(velocity_grid,y_i, u_i,  v_i, periodic_grid, x_forward_d, y_forward_d,dt);
            //interpolate backward 
            interpolateVelocity(x_backward_d, y_backward_d, n, m,PERIODIC_START,PERIODIC_END, periodic_grid, velocity_grid,velocity_grid_next);
            //interpolate forward
            interpolateVelocity(x_forward_d, y_forward_d, n, m,PERIODIC_START,PERIODIC_END, periodic_grid, velocity_grid,velocity_grid_next_2);

            double u = mac_cormack_correction(u_i,y_i,velocity_grid,velocity_grid_next,velocity_grid_next_2,n,m);
            velocity_grid_next[periodic_linear_Idx(u_i,y_i,2*n,m)] = u;

            double v = mac_cormack_correction(v_i,y_i,velocity_grid,velocity_grid_next,velocity_grid_next_2,n,m);
            velocity_grid_next[periodic_linear_Idx(v_i,y_i,2*n,m)] = v;
        }
    } 
    memcpy(velocity_grid, velocity_grid_next, 2 * n * m * sizeof(double));
    free(velocity_grid_next_2);
}

double mac_cormack_correction(const int idx_x,const int y_i,const double * velocity_grid, const double * velocity_grid_bw, const double* velocity_grid_fw,  int n, int m)
{
            double bw = velocity_grid_bw[periodic_linear_Idx(idx_x,y_i,2*n,m)];
            double fw = velocity_grid_fw[periodic_linear_Idx(idx_x,y_i,2*n,m)];
            double field = velocity_grid[periodic_linear_Idx(idx_x,y_i,2*n,m)];
            //double out_val = 0.5 * (bw + fw); // temporal average
            double out_val = fw + MACCORMACK_CORRECTION * 0.5 * (field - bw); //like in PHIflow, but clashes with wikipedia-definition
            double min_,max_;
            //clipping
            min_max_neighbors(min_,max_,idx_x,y_i,velocity_grid,n,m);
            clip(out_val,min_,max_);
            return out_val;
}

void min_max_neighbors(double &min, double &max, const int idx,const int y_i, const double * velocity_grid,const int n, const int m)
{
    double neighbors[4];
    neighbors[0] = velocity_grid[periodic_linear_Idx(idx-2,y_i,2*n,m)];
    neighbors[1] = velocity_grid[periodic_linear_Idx(idx+2,y_i,2*n,m)];
    neighbors[2] = velocity_grid[periodic_linear_Idx(idx,y_i-1,2*n,m)];
    neighbors[3] = velocity_grid[periodic_linear_Idx(idx,y_i+1,2*n,m)];

    min = neighbors[0];
    max = neighbors[0];
    for (int i = 1; i < 4; i++) {
        if (neighbors[i] < min) {
            min = neighbors[i];
        }
        if (neighbors[i] > max) {
            max = neighbors[i];
        }
    }
}
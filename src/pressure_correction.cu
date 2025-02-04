#include "pressure_correction.h"

void make_incompressible(double* velocity_grid, double* divergence, double*pressure, int n, int m){
    calculateDivergence(velocity_grid,divergence);
    //cuBlas is column major
    //switchRowColMajor(divergence,m,n); //not needed, solver interprets this as 1D-vector anyways

    double *laplace = (double *)malloc(n * m * n * m * sizeof(double));
    //std::cout << "Divergence" << std::endl;
    //print_matrix(m, n, divergence, n);
    //std::cout << "Laplacian" << std::endl;
    constructDiscretizedLaplacian(laplace); // LP^T = LP -> no need to transpose

    //print_matrix(n*m, n*m, laplace, n*m);
    size_t pressure_size = n*m;
    solveDense(laplace,divergence,pressure,pressure_size);
    //std::cout << "Pressure" << std::endl;
    //switchRowColMajor(pressure,n,m);
    //print_matrix_row_major(m, n, pressure, n);
    //free(laplace);
    //calculate gradient
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < n; i++)
        {   
            int u_i = 2 * (i - 1);
            int v_i = 2 * i;
            //TODO: check which finite difference to use
            double p = pressure[periodic_linear_Idx(i,y_i,n,m)];
            double p_right = pressure[periodic_linear_Idx(i+1,y_i,n,m)];
            double p_down = pressure[periodic_linear_Idx(i,y_i+1,n,m)];
            double p_dx = (p_right - p) / dx;
            double p_dy = (p_down - p) / dy;

            //central differences
            //double p = pressure[periodic_linear_Idx(i,y_i,n,m)];
            //double p_left = pressure[periodic_linear_Idx(i-1,y_i,n,m)];
            //double p_right = pressure[periodic_linear_Idx(i+1,y_i,n,m)];
            //double p_up = pressure[periodic_linear_Idx(i,y_i-1,n,m)];
            //double p_down = pressure[periodic_linear_Idx(i,y_i+1,n,m)];
            //double p_dx = (p_right - p_left) / dx;
            //double p_dy = (p_down - p_up) / dy;

            //TODO:scale somehow with dt?
            velocity_grid[periodic_linear_Idx(u_i,y_i)] -= p_dx;
            velocity_grid[periodic_linear_Idx(v_i,y_i)] -= p_dy;
            //velocity_grid[periodic_linear_Idx(u_i,y_i)] += p_dx;
            //velocity_grid[periodic_linear_Idx(v_i,y_i)] += p_dy;
        }
    }
    

}
void calculateDivergence(const double* velocity_grid,double*divergence,int n, int m){

    //divergence by central differences
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 0; i < n; i++)
        {   
            int u_i = 2 * i;
            int v_i = 2 * i + 1;
            double u = velocity_grid[periodic_linear_Idx(u_i,y_i)];
            double v = velocity_grid[periodic_linear_Idx(v_i,y_i)];

            double u_left = velocity_grid[periodic_linear_Idx(u_i - 2,y_i)];
            double u_right = velocity_grid[periodic_linear_Idx(u_i + 2,y_i)];

            double v_down= velocity_grid[periodic_linear_Idx(v_i,y_i+1)];
            double v_up= velocity_grid[periodic_linear_Idx(v_i,y_i-1)];
            //central differences
            //double div = (u_right - u_left) / (2 * dx) + (v_up - v_down) / (2 * dy);
            double div = (u_right - u_left) / (2 * dx) + (v_down - v_up) / (2 * dy);
            //backward differences
            //double div = (u - u_left) / dx + (v - v_down) / dy;
            divergence[periodic_linear_Idx(i,y_i,n,m)] = div;
        }
    }
}

/*
void jacobiStep(double*pressure,double * pressure_next, double* divergence,int n=NUM_N, int m=M)
{
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    //double* pressure_next = (double *)malloc(n * m * sizeof(double));
    double alpha = 1;
    double beta = 1;
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < n; i++)
        {   
            double b = divergence[periodic_linear_Idx(i,y_i,n,m)];
            double p = pressure[periodic_linear_Idx(i,y_i,n,m)];
            double p_left = pressure[periodic_linear_Idx(i-1,y_i,n,m)];
            double p_right = pressure[periodic_linear_Idx(i+1,y_i,n,m)];
            double p_up = pressure[periodic_linear_Idx(i,y_i+1,n,m)];
            double p_down = pressure[periodic_linear_Idx(i,y_i-1,n,m)];
            double p_next = (p_right + p_left)*dx*dx + (p_up + p_down)*dy*dy - alpha *b*dx*dx*dy*dy;
            p_next = p_next / (2*(dx*dx + dy*dy));
            p_next *= beta;
            pressure_next[periodic_linear_Idx(i,y_i,n,m)] = p_next;
        }
    } 
    memcpy(pressure,pressure_next,n*m*sizeof(double));
}

*/
void constructDiscretizedLaplacian(double* laplace_discrete,int n){
    //discretized laplacian is always same for grid -> unit laplacian is sufficient
    //order 2
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    std::vector<double> lp_(n*n*n*n, 0);
    std::copy(lp_.begin(), lp_.end(), laplace_discrete);
    for (int lp_row = 0; lp_row < n*n; lp_row++)
    {
        //one lp_row has one entry for all entries in source martrix
        int src_y = lp_row / n; 
        int src_x = lp_row % n; 

        int lp_center= src_y * n + src_x;
        laplace_discrete[lp_row * n*n + lp_center] = 4.0 / (dx*dx);
        //neighbors
        int up_src = periodic_linear_Idx(src_x, src_y - 1, n, n);
        int down_src = periodic_linear_Idx(src_x, src_y + 1, n, n);
        int left_src = periodic_linear_Idx(src_x - 1, src_y, n, n);
        int right_src = periodic_linear_Idx(src_x + 1, src_y, n, n);
        laplace_discrete[lp_row * n * n + up_src] = -1.0 *(dx*dx);
        laplace_discrete[lp_row * n * n + down_src] = -1.0 *(dx*dx);
        laplace_discrete[lp_row * n * n + left_src] = -1.0 * (dx*dx);
        laplace_discrete[lp_row * n * n + right_src] = -1.0 * (dx*dx);
    }
}
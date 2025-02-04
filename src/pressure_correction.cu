#include "pressure_correction.cuh"

void makeIncompressible(double* velocity_grid, double* divergence, double*pressure, int n, int m, const double dx){
    calculateDivergence(velocity_grid,divergence,n,n,dx);
    //cuBlas is column major
    //switchRowColMajor(divergence,m,n); //not needed, solver interprets this as 1D-vector anyways

    double *laplace = (double *)malloc(n * m * n * m * sizeof(double));
    //std::cout << "Divergence" << std::endl;
    //print_matrix(m, n, divergence, n);
    //std::cout << "Laplacian" << std::endl;
    constructDiscretizedLaplacian(laplace,n,dx); // LP^T = LP -> no need to transpose

    //print_matrix(n*m, n*m, laplace, n*m);
    size_t pressure_size = n*m;
    solveDense(laplace,divergence,pressure,pressure_size);
    //std::cout << "Pressure" << std::endl;
    //switchRowColMajor(pressure,n,m);
    //print_matrix_row_major(m, n, pressure, n);
    free(laplace);
    correct_velocity(velocity_grid,pressure,n,m,dx);
}
void correct_velocity(double * velocity_grid,double * pressure,int n, int m, double dx)
{
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 0; i < n; i++)
        {   
            int u_i = 2 * i;
            int v_i = 2 * i + 1;
            //TODO: check which finite difference to use
            double p = pressure[periodic_linear_Idx(i,y_i,n,m)];
            double p_right = pressure[periodic_linear_Idx(i+1,y_i,n,m)];
            double p_down = pressure[periodic_linear_Idx(i,y_i+1,n,m)];
            double p_dx = (p_right - p) / dx;
            double p_dy = (p_down - p) / dx;

            //central differences
            //double p = pressure[periodic_linear_Idx(i,y_i,n,m)];
            //double p_left = pressure[periodic_linear_Idx(i-1,y_i,n,m)];
            //double p_right = pressure[periodic_linear_Idx(i+1,y_i,n,m)];
            //double p_up = pressure[periodic_linear_Idx(i,y_i-1,n,m)];
            //double p_down = pressure[periodic_linear_Idx(i,y_i+1,n,m)];
            //double p_dx = (p_right - p_left) / dx;
            //double p_dy = (p_down - p_up) / dy;

            //TODO:scale somehow with dt?
            velocity_grid[periodic_linear_Idx(u_i,y_i,2*n,m)] -= p_dx;
            velocity_grid[periodic_linear_Idx(v_i,y_i,2*n,m)] -= p_dy;
            //velocity_grid[periodic_linear_Idx(u_i,y_i)] += p_dx;
            //velocity_grid[periodic_linear_Idx(v_i,y_i)] += p_dy;
        }
    }
}

namespace gpu 
{

void makeIncompressible(double* velocity_grid, double* d_B, double* laplace, int n, int m)
{
    /*d_B is used for divergence and pressure data*/

    dim3 blockDim(TILE_SIZE,TILE_SIZE);
    dim3 gridDimDiv((n + TILE_SIZE-1)/TILE_SIZE,(+ TILE_SIZE-1)/TILE_SIZE); 
    gpu::calculateDivergence<<<gridDimDiv,blockDim>>>(velocity_grid,d_B,n,m,DX);
    CHECK_CUDA(cudaDeviceSynchronize());
    gpu::solveDense(laplace,d_B,n*m);

    //TODO: parallelize u and v correction? -> don't coalesce?
    dim3 gridDimVel((NUM_N + TILE_SIZE-1)/TILE_SIZE,(NUM_N + TILE_SIZE-1)/TILE_SIZE); 
    gpu::correct_velocity<<<gridDimVel,blockDim>>>(velocity_grid,d_B,n,m,DX);
}

__global__ void correct_velocity(double * velocity_grid,double * pressure,int n, int m, double dx)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x; 
    int row = threadIdx.y + blockIdx.y * blockDim.y; 
    if (row < m && col < n)
    {
            int u_i = 2 * col;
            int v_i = 2 * col + 1;
            double p = pressure[periodic_linear_Idx(col,row,n,m)];
            double p_right = pressure[periodic_linear_Idx(col+1,row,n,m)];
            double p_down = pressure[periodic_linear_Idx(col,row+1,n,m)];
            double p_dx = (p_right - p) / dx;
            double p_dy = (p_down - p) / dx;

            velocity_grid[periodic_linear_Idx(u_i,row,2*n,m)] -= p_dx;
            velocity_grid[periodic_linear_Idx(v_i,row,2*n,m)] -= p_dy;
    }
}
}


void calculateDivergence(const double* velocity_grid,double*divergence,int n, int m, const double dx){

    //divergence by central differences
    //double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    //double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 0; i < n; i++)
        {   
            int u_i = 2 * i;
            int v_i = 2 * i + 1;
            double u = velocity_grid[periodic_linear_Idx(u_i,y_i,2*n,m)];
            double v = velocity_grid[periodic_linear_Idx(v_i,y_i,2*n,m)];

            double u_left = velocity_grid[periodic_linear_Idx(u_i - 2,y_i,2*n,m)];
            double u_right = velocity_grid[periodic_linear_Idx(u_i + 2,y_i,2*n,m)];

            double v_down= velocity_grid[periodic_linear_Idx(v_i,y_i+1,2*n,m)];
            double v_up= velocity_grid[periodic_linear_Idx(v_i,y_i-1,2*n,m)];
            //central differences
            //double div = (u_right - u_left) / (2 * dx) + (v_up - v_down) / (2 * dy);
            double div = (u_right - u_left) / (2 * dx) + (v_down - v_up) / (2 * dx);
            //backward differences
            //double div = (u - u_left) / dx + (v - v_down) / dy;
            divergence[periodic_linear_Idx(i,y_i,n,m)] = div;
        }
    }
}

namespace gpu
{

__global__ void calculateDivergence(const double* velocity_grid,double*divergence,int n, int m, const double dx)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x; 
    int row = threadIdx.y + blockIdx.y * blockDim.y; 

    if (row < M && col < NUM_N)
    {
        int u_i = 2 * col;
        int v_i = 2 * col + 1;
        double u = velocity_grid[periodic_linear_Idx(u_i,row,2*n,m)];
        double v = velocity_grid[periodic_linear_Idx(v_i,row,2*n,m)];

        double u_left = velocity_grid[periodic_linear_Idx(u_i - 2,row,2*n,m)];
        double u_right = velocity_grid[periodic_linear_Idx(u_i + 2,row,2*n,m)];

        double v_down= velocity_grid[periodic_linear_Idx(v_i,row+1,2*n,m)];
        double v_up= velocity_grid[periodic_linear_Idx(v_i,row-1,2*n,m)];
        //central differences
        //double div = (u_right - u_left) / (2 * dx) + (v_up - v_down) / (2 * dy);
        double div = (u_right - u_left) / (2 * dx) + (v_down - v_up) / (2 * dx);
        //backward differences
        //double div = (u - u_left) / dx + (v - v_down) / dy;
        divergence[periodic_linear_Idx(col,row,n,m)] = div;

    }
}
}

void constructDiscretizedLaplacian(double* laplace_discrete,int n, const double dx){
    //discretized laplacian is always same for grid -> unit laplacian is sufficient
    //order 2
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

namespace gpu {
__global__  void fillLaplaceValues(double* laplace_discrete, int n, const double dx)
{
    /*CAUTION: expects 0 inititilized laplace_discrete*/
    int lp_idx = threadIdx.x + blockIdx.x * blockDim.x; 

    //if (lp_idx < n*n)
        //for (int i= 0; i< n*n*n*n; i++){
            //laplace_discrete[lp_idx * n*n + i] =0.0;
        //}
        //laplace_discrete[lp_idx*2] = 0.0;
        //laplace_discrete[lp_idx*1] = 0.0;
    //__syncthreads();
    if (lp_idx < n*n)
    {
        //one lp_row has one entry for all entries in source martrix
        int src_y = lp_idx / n; 
        int src_x = lp_idx % n; 

        int lp_center= src_y * n + src_x;
        laplace_discrete[lp_idx * n*n + lp_center] = 4.0 / (dx*dx);
        //neighbors
        int up_src = periodic_linear_Idx(src_x, src_y - 1, n, n);
        int down_src = periodic_linear_Idx(src_x, src_y + 1, n, n);
        int left_src = periodic_linear_Idx(src_x - 1, src_y, n, n);
        int right_src = periodic_linear_Idx(src_x + 1, src_y, n, n);
        double neighbor_weight = -1.0 *(dx*dx);
        laplace_discrete[lp_idx * n * n + up_src] = neighbor_weight;
        laplace_discrete[lp_idx * n * n + down_src] = neighbor_weight;
        laplace_discrete[lp_idx * n * n + left_src] = neighbor_weight;
        laplace_discrete[lp_idx * n * n + right_src] = neighbor_weight;
    }
}

void constructDiscretizedLaplacian(double* laplace_discrete,int n, const double dx)
{
    CHECK_CUDA(cudaMemset(laplace_discrete,0,n*n*n*n *sizeof(double)));
    dim3 blockDim(TILE_SIZE);
    dim3 gridDim((n*n + TILE_SIZE -1)/TILE_SIZE);
    gpu::fillLaplaceValues<<<gridDim,blockDim>>>(laplace_discrete,n,dx);
}

}
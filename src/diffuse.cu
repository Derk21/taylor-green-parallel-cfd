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

    dim3 blockDim(TILE_SIZE*2,TILE_SIZE);
    dim3 gridDim(((2*n) + blockDim.x-1)/blockDim.x,(m+ blockDim.y-1)/blockDim.y); 

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
    auto t_col = threadIdx.x + 2;
    auto t_row = threadIdx.y + 1;

    const dim3 PADDED_SIZE(blockDim.x+4,blockDim.y+2);  
    __shared__ double CURR[(TILE_SIZE*2 + 4) * (TILE_SIZE + 2)];
    __shared__ double NEXT[(TILE_SIZE*2 + 4) * (TILE_SIZE + 2)];

    //init
    if (row < m && col < 2*n)
    {
        if(threadIdx.x < blockDim.x && threadIdx.y < blockDim.y)
        {
            //fill inner
            double temp = velocity_grid[periodic_linear_Idx(col,row,2*n,m)];
            CURR[t_row * PADDED_SIZE.x + t_col] = temp;
            NEXT[t_row * PADDED_SIZE.x + t_col] = temp;

            //boundary
            if ( threadIdx.y == 0 ){ //top row
                temp = velocity_grid[periodic_linear_Idx(col,row-1,2*n,m)];
                CURR[(t_row - 1) * PADDED_SIZE.x + t_col]=temp;
                NEXT[(t_row - 1) * PADDED_SIZE.x + t_col]=temp;
            }
            if (threadIdx.y == blockDim.y-1)
            {
                temp = velocity_grid[periodic_linear_Idx(col,row+1,2*n,m)];
                CURR[(t_row + 1) * PADDED_SIZE.x + t_col]=temp;
                NEXT[(t_row + 1) * PADDED_SIZE.x + t_col]=temp;
            }
            if (threadIdx.x <= 1) //left
            {
                temp = velocity_grid[periodic_linear_Idx(col-2,row,2*n,m)];
                CURR[t_row  * PADDED_SIZE.x + (t_col-2)]=temp;
                NEXT[t_row  * PADDED_SIZE.x + (t_col-2)]=temp;
            }
            if (threadIdx.x >= blockDim.x - 2) //right
            {
                temp = velocity_grid[periodic_linear_Idx(col+2,row,2*n,m)];
                CURR[t_row  * PADDED_SIZE.x + (t_col+2)]=temp;
                NEXT[t_row  * PADDED_SIZE.x + (t_col+2)]=temp;
            }
        }
    }
    __syncthreads();
    //diffusion
    if (row < m && col < 2*n)
    {
        if (threadIdx.y < blockDim.y && threadIdx.x < blockDim.x){
            
            double c = CURR[t_row * PADDED_SIZE.x + t_col];

            double left = CURR[t_row * PADDED_SIZE.x + (t_col-2)];
            double right = CURR[t_row * PADDED_SIZE.x + (t_col+2)];
            double up = CURR[(t_row+1) * PADDED_SIZE.x + t_col];
            double down = CURR[(t_row-1) * PADDED_SIZE.x + t_col];

            double diffusion = (right - 2 * c + left) / (dx * dx) 
                                + (up - 2 * c + down) / (dx * dx);
            NEXT[t_row*PADDED_SIZE.x+t_col]= c + amount * diffusion;
        }
    }
    //copy back to global memory (might move this into if condition above)
    if (row < m && col < 2*n && (threadIdx.y < blockDim.y && threadIdx.x < blockDim.x))
    {
        velocity_grid[periodic_linear_Idx(col, row,2*n,m)] = NEXT[t_row*PADDED_SIZE.x + t_col];
    }
}
}



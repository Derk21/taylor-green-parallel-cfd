#include "advect.cuh"
#include <cassert>

__host__ __device__ void interpolateVelocity(double &u, double &v, const double x_d, const double y_d, const double *periodic_grid, const double *velocity_grid, int n, int m, const double dx)
{
    int u_i_closest, v_i_closest, y_i_closest;
    setClosestGridPointIdx(x_d, y_d, n, m, v_i_closest, y_i_closest);
    u_i_closest = v_i_closest - 1;

    // interpolation weights
    double x_closest = periodic_grid[periodic_linear_Idx(u_i_closest, y_i_closest, 2 * n, m)];
    double y_closest = periodic_grid[periodic_linear_Idx(v_i_closest, y_i_closest, 2 * n, m)];
    // double dx = (periodic_end - periodic_start) / (n - 1);
    // double dy = (periodic_end - periodic_start ) / (m - 1);
    double x_diff = (x_d - x_closest) / dx;
    double y_diff = (y_d - y_closest) / dx;

    // forward bilinear interpolation
    u = get_interpolated(u_i_closest, y_i_closest, x_diff, y_diff, velocity_grid, n, m);
    v = get_interpolated(v_i_closest, y_i_closest, x_diff, y_diff, velocity_grid, n, m);
}

__host__ __device__ double get_interpolated(const int &i_closest, const int &y_i_closest, const double &x_diff, const double &y_diff, const double *velocity_grid, int n, int m)
{
    double weight_contained = (1.0 - x_diff) * (1.0 - y_diff);
    double weight_next_x = x_diff * (1.0 - y_diff);
    double weight_next_x_y = x_diff * y_diff;
    double weight_next_y = (1.0 - x_diff) * y_diff;

    // double sum_weights = weight_contained + weight_next_x + weight_next_y + weight_next_x_y;
    // assert(is_close(sum_weights,1.0));
    double val = weight_contained * velocity_grid[periodic_linear_Idx(i_closest, y_i_closest, 2 * n, m)];
    // y_direction next grid cell
    val += weight_next_y * velocity_grid[periodic_linear_Idx(i_closest, y_i_closest + 1, 2 * n, m)];
    // x_direction next grid cell
    val += weight_next_x * velocity_grid[periodic_linear_Idx(i_closest + 2, y_i_closest, 2 * n, m)];
    // next grid cell in diagonal direction
    val += weight_next_x_y * velocity_grid[periodic_linear_Idx(i_closest + 2, y_i_closest + 1, 2 * n, m)];
    return val;
}

__host__ __device__ void integrateEuler(const double *velocity_grid, int &y_i, int &u_i, int &v_i, const double *periodic_grid, double &x_d, double &y_d, const double dt, int n, int m)
{
    double u_old = velocity_grid[periodic_linear_Idx(u_i, y_i, 2 * n, m)];
    double v_old = velocity_grid[periodic_linear_Idx(v_i, y_i, 2 * n, m)];

    double x = periodic_grid[periodic_linear_Idx(u_i, y_i, 2 * n, m)];
    double y = periodic_grid[periodic_linear_Idx(v_i, y_i, 2 * n, m)];

    x_d = fmod(x + dt * u_old + PERIODIC_END, PERIODIC_END) + PERIODIC_START;
    y_d = fmod(y + dt * v_old + PERIODIC_END, PERIODIC_END) + PERIODIC_START;
}

void advectSemiLagrange(double *velocity_grid, double *velocity_grid_next, const double *periodic_grid, const double dt, int n, int m, double dx)
{
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 0; i < n; i++)
        {
            int u_i = 2 * i;
            int v_i = (2 * i) + 1;
            double x_d, y_d, u, v;
            // backward euler
            integrateEuler(velocity_grid, y_i, u_i, v_i, periodic_grid, x_d, y_d, -dt, n, m);
            interpolateVelocity(u, v, x_d, y_d, periodic_grid, velocity_grid, n, m, dx);
            velocity_grid_next[periodic_linear_Idx(u_i, y_i, 2 * n, m)] = u;
            velocity_grid_next[periodic_linear_Idx(v_i, y_i, 2 * n, m)] = v;

            // clip velocity
            // double u = velocity_grid_next[periodic_linear_Idx(u_i,y_i,2*n,m)];
            // double min_,max_;
            // min_max_neighbors(min_,max_,u_i,y_i,velocity_grid,n,m);
            ////clip(u,min_,max_);
            // clip(u,min_,2.0);
            // velocity_grid_next[periodic_linear_Idx(u_i,y_i,2*n,m)] = u;

            // double v = velocity_grid_next[periodic_linear_Idx(v_i,y_i,2*n,m)];
            // min_max_neighbors(min_,max_,v_i,y_i,velocity_grid,n,m);
            ////clip(v,min_,max_);
            // clip(v,min_,2.0);
            // velocity_grid_next[periodic_linear_Idx(v_i,y_i,2*n,m)] = v;
            // assert(v <= 2.0);
            // assert(u <= 2.0);
        }
    }
    memcpy(velocity_grid, velocity_grid_next, 2 * n * m * sizeof(double));
}

void advectMacCormack(double *velocity_grid, double *velocity_grid_next, const double *periodic_grid, const double dt, const int n, const int m, const double dx)
{
    double *velocity_bw_fw = (double *)malloc(n * m * 2 * sizeof(double));
    // memcpy(velocity_fw,velocity_grid, 2*n*m*sizeof(double));

    double *velocity_bw = (double *)malloc(n * m * 2 * sizeof(double));
    double *integrated_fw = (double *)malloc(n * m * 2 * sizeof(double));
    // double *integrated_bw = (double *)malloc(n * m * 2 * sizeof(double));
    // memcpy(velocity_grid_update,velocity_grid, 2*n*m*sizeof(double));
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 0; i < n; i++)
        {
            int u_i = 2 * i;
            int v_i = (2 * i) + 1;
            double x_backward_d, y_backward_d, u_bw, v_bw;

            // backward euler -dt
            integrateEuler(velocity_grid, y_i, u_i, v_i, periodic_grid, x_backward_d, y_backward_d, -dt, n, m);
            interpolateVelocity(u_bw, v_bw, x_backward_d, y_backward_d, periodic_grid, velocity_grid, n, m, dx);
            velocity_bw[periodic_linear_Idx(u_i, y_i, 2 * n, m)] = u_bw;
            velocity_bw[periodic_linear_Idx(v_i, y_i, 2 * n, m)] = v_bw;
        }
    }

    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 0; i < n; i++)
        {
            int u_i = 2 * i;

            int v_i = (2 * i) + 1;
            double x_forward_d, y_forward_d;

            integrateEuler(velocity_grid, y_i, u_i, v_i, periodic_grid, x_forward_d, y_forward_d, dt, n, m);
            double u_bw_fw = velocity_bw[periodic_linear_Idx(u_i, y_i, 2 * n, m)];
            double v_bw_fw = velocity_bw[periodic_linear_Idx(v_i, y_i, 2 * n, m)];
            interpolateVelocity(u_bw_fw, v_bw_fw, x_forward_d, y_forward_d, periodic_grid, velocity_bw, n, m, dx);
            velocity_bw_fw[periodic_linear_Idx(u_i, y_i, 2 * n, m)] = u_bw_fw;
            velocity_bw_fw[periodic_linear_Idx(v_i, y_i, 2 * n, m)] = v_bw_fw;

            double u = mac_cormack_correction(u_i, y_i, velocity_grid, velocity_bw, velocity_bw_fw, n, m);
            velocity_grid_next[periodic_linear_Idx(u_i, y_i, 2 * n, m)] = u;

            double v = mac_cormack_correction(v_i, y_i, velocity_grid, velocity_bw, velocity_bw_fw, n, m);
            velocity_grid_next[periodic_linear_Idx(v_i, y_i, 2 * n, m)] = v;
        }
    }

    // int u_i = 6;
    // int v_i = 7;
    // int y_i = 0;
    // std::cout <<"after loop mat u=" <<velocity_grid_update[periodic_linear_Idx(u_i,y_i,2*n,m)]
    //<<" v=" <<velocity_grid_update[periodic_linear_Idx(v_i,y_i,2*n,m)] << std::endl;

    // std::cout<< "velocity grid next before copy"<< std::endl;
    // print_matrix_row_major(m,2*n,velocity_grid_update,2*n);
    memcpy(velocity_grid, velocity_grid_next, 2 * n * m * sizeof(double));
    // std::cout<< "velocity grid after copy"<< std::endl;
    // print_matrix_row_major(m,2*n,velocity_grid_update,2*n);
    // free(integrated_bw);
    free(integrated_fw);
    free(velocity_bw_fw);
    free(velocity_bw);
}

namespace gpu
{

    void advectSemiLagrange(
        double *velocity_grid,
        double *velocity_grid_next,
        const double *periodic_grid,
        const double dt, int n, int m, double dx)
    {
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
        gpu::integrateAndInterpolateKernel<<<gridDim, blockDim>>>(
            periodic_grid, velocity_grid, velocity_grid_next, -dt, n, m, dx);
        // clipping?
        CHECK_CUDA(cudaMemcpy(velocity_grid, velocity_grid_next, n * m * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    void advectSemiLagrangeSeparate(
        double *velocity_grid,
        double *velocity_grid_next,
        double *integrated_backward,
        const double *periodic_grid,
        const double dt, int n, int m, double dx)
    {
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        dim3 gridDimIntegrate(((2 * n) + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        gpu::integrateKernel<<<gridDimIntegrate, blockDim>>>(periodic_grid, velocity_grid, integrated_backward, -dt, n, m);
        gpu::interpolateKernel<<<gridDim, blockDim>>>(periodic_grid, velocity_grid, velocity_grid_next, integrated_backward, n, m, dx);
        // clipping?
        CHECK_CUDA(cudaMemcpy(velocity_grid, velocity_grid_next, n * m * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    void advectMacCormack(
        double *velocity_grid,
        double *velocity_bw,
        double *velocity_bw_fw,
        double *integrated_fw,
        double *integrated_bw,
        const double *periodic_grid,
        const double dt, int n, int m, double dx)
    {
        // forward and backward integration and interpolation are independent -> async streams
        cudaStream_t stream_forward, stream_backward;
        cudaStreamCreate(&stream_forward);
        cudaStreamCreate(&stream_backward);
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        dim3 gridDimIntegrate(((2 * n) + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

        gpu::integrateKernel<<<gridDimIntegrate, blockDim, 0, stream_backward>>>(
            periodic_grid, velocity_grid, integrated_bw, -dt, n, m);

        gpu::interpolateKernel<<<gridDim, blockDim, 0, stream_backward>>>(
            periodic_grid, velocity_grid, velocity_bw, integrated_bw, n, m, dx);

        gpu::integrateKernel<<<gridDimIntegrate, blockDim, 0, stream_forward>>>(
            periodic_grid, velocity_grid, integrated_fw, dt, n, m);

        cudaStreamSynchronize(stream_backward);
        cudaStreamSynchronize(stream_forward);
        cudaStreamDestroy(stream_forward);
        cudaStreamDestroy(stream_backward);

        gpu::interpolateKernel<<<gridDim, blockDim>>>(
            periodic_grid, velocity_bw, velocity_bw_fw, integrated_fw, n, m);

        dim3 blockDimCorrection(TILE_SIZE, TILE_SIZE);
        dim3 gridDimCorrection(((2 * n) + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        gpu::macCormackCorrectionKernel<<<gridDimCorrection, blockDimCorrection>>>(
            velocity_grid, velocity_bw, velocity_bw_fw, n, m);
    }

    __global__ void integrateKernel(const double *periodic_grid, const double *velocity_grid, double *integrated, const double dt, const int n, const int m)
    {
        // launch for 2*n,n
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;
        // int u_i = col * 2;
        // int v_i = (col * 2) + 1;
        if (row < m && col < 2 * n)
        {
            double x_d;

            double old = velocity_grid[periodic_linear_Idx(col, row, 2 * n, m)];
            // double v_old = velocity_grid[periodic_linear_Idx(v_i, row,2*n,m)];

            double x = periodic_grid[periodic_linear_Idx(col, row, 2 * n, m)];
            // double y = periodic_grid[periodic_linear_Idx(v_i,row,2*n,m)];

            x_d = fmod(x + dt * old + PERIODIC_END, PERIODIC_END) + PERIODIC_START;
            // y_d = fmod(y + dt * v_old+PERIODIC_END,PERIODIC_END)+PERIODIC_START;
            // integrateEuler(velocity_grid,row,u_i,v_i,periodic_grid,x_d,y_d,dt,n,m);
            integrated[periodic_linear_Idx(col, row, 2 * n, m)] = x_d;
            // integrated[periodic_linear_Idx(v_i,row,2*n,m)] = y_d;
        }
    }

    __global__ void interpolateKernel(const double *periodic_grid, const double *velocity_grid, double *velocity_grid_next, double *integrated, const int n, const int m, const double dx)
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;

        if (row < m && col < n)
        {
            int u_i = col * 2;
            int v_i = (col * 2) + 1;

            double x_d = integrated[periodic_linear_Idx(u_i, row, 2 * n, m)];
            double y_d = integrated[periodic_linear_Idx(v_i, row, 2 * n, m)];

            int u_i_closest, v_i_closest, y_i_closest;
            setClosestGridPointIdx(x_d, y_d, n, m, v_i_closest, y_i_closest);
            u_i_closest = v_i_closest - 1;

            double x_closest = periodic_grid[periodic_linear_Idx(u_i_closest, y_i_closest, 2 * n, m)];
            double y_closest = periodic_grid[periodic_linear_Idx(v_i_closest, y_i_closest, 2 * n, m)];

            double x_diff = (x_d - x_closest) / dx;
            double y_diff = (y_d - y_closest) / dx;

            double weight_contained = (1.0 - x_diff) * (1.0 - y_diff);
            double weight_next_x = x_diff * (1.0 - y_diff);
            double weight_next_x_y = x_diff * y_diff;
            double weight_next_y = (1.0 - x_diff) * y_diff;

            double u = weight_contained * velocity_grid[periodic_linear_Idx(u_i_closest, y_i_closest, 2 * n, m)];
            u += weight_next_y * velocity_grid[periodic_linear_Idx(u_i_closest, y_i_closest + 1, 2 * n, m)];
            u += weight_next_x * velocity_grid[periodic_linear_Idx(u_i_closest + 2, y_i_closest, 2 * n, m)];
            u += weight_next_x_y * velocity_grid[periodic_linear_Idx(u_i_closest + 2, y_i_closest + 1, 2 * n, m)];

            double v = weight_contained * velocity_grid[periodic_linear_Idx(v_i_closest, y_i_closest, 2 * n, m)];
            v += weight_next_y * velocity_grid[periodic_linear_Idx(v_i_closest, y_i_closest + 1, 2 * n, m)];
            v += weight_next_x * velocity_grid[periodic_linear_Idx(v_i_closest + 2, y_i_closest, 2 * n, m)];
            v += weight_next_x_y * velocity_grid[periodic_linear_Idx(v_i_closest + 2, y_i_closest + 1, 2 * n, m)];

            velocity_grid_next[periodic_linear_Idx(u_i, row, 2 * n, m)] = u;
            velocity_grid_next[periodic_linear_Idx(v_i, row, 2 * n, m)] = v;
        }
    }

    //__global__ void interpolateKernel(const double *periodic_grid, const double *velocity_grid, double * velocity_grid_next,double * integrated,const int n, const int m,const double dx)
    //{
    // int col = threadIdx.x + blockIdx.x * blockDim.x;
    // int row = threadIdx.y + blockIdx.y * blockDim.y;
    // int u_i = col * 2;
    // int v_i = (col * 2) + 1;
    // SADLY I COULDN'T FIX THE SHARED MEMORY IMPLEMENTATION IN TIME
    // const int PADDED_SIZE = TILE_SIZE+1;
    //__shared__ double2 VEL[(TILE_SIZE+1)*(TILE_SIZE+1)];
    //__shared__ double2 PERIODIC[(TILE_SIZE+1)*(TILE_SIZE+1)];
    ////init shared
    // TODO: switch to 1 thread per velocity component?
    // if (row < m && col < n)
    //{
    // if(threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE)
    //{
    ////fill inner
    // double2 temp;
    // temp.x = velocity_grid[periodic_linear_Idx(u_i,row,2*n,m)];
    // temp.y = velocity_grid[periodic_linear_Idx(v_i,row,2*n,m)];
    // VEL[threadIdx.y * PADDED_SIZE + threadIdx.x] = temp;
    // temp.x = periodic_grid[periodic_linear_Idx(u_i,row,2*n,m)];
    // temp.y = periodic_grid[periodic_linear_Idx(v_i,row,2*n,m)];
    // PERIODIC[threadIdx.y*PADDED_SIZE+threadIdx.x]=temp;
    ////down bound
    // if (threadIdx.y == (TILE_SIZE-1)){
    // temp.x = velocity_grid[periodic_linear_Idx(u_i,row+1,2*n,m)];
    // temp.y = velocity_grid[periodic_linear_Idx(v_i,row+1,2*n,m)];
    // VEL[(threadIdx.y+1) * PADDED_SIZE + threadIdx.x] = temp;
    // temp.x = periodic_grid[periodic_linear_Idx(u_i,row+1,2*n,m)];
    // temp.y = periodic_grid[periodic_linear_Idx(v_i,row+1,2*n,m)];
    // PERIODIC[(threadIdx.y+1)*PADDED_SIZE+threadIdx.x]=temp;

    //}
    // if (threadIdx.x == (TILE_SIZE-1)){
    // temp.x = velocity_grid[periodic_linear_Idx(u_i+2,row,2*n,m)];
    // temp.y = velocity_grid[periodic_linear_Idx(v_i+2,row,2*n,m)];
    // VEL[threadIdx.y * PADDED_SIZE + (threadIdx.x+1)] = temp;
    // temp.x = periodic_grid[periodic_linear_Idx(u_i+2,row,2*n,m)];
    // temp.y = periodic_grid[periodic_linear_Idx(v_i+2,row,2*n,m)];
    // PERIODIC[threadIdx.y*PADDED_SIZE+(threadIdx.x+1)]=temp;
    //}
    //}
    //}
    //__syncthreads();

    // if (row < m && col < n)
    //{
    // double x_d, y_d;
    // x_d = integrated[periodic_linear_Idx(u_i,row,2*n,m)];
    // y_d = integrated[periodic_linear_Idx(v_i,row,2*n,m)];
    // int u_i_closest, v_i_closest, y_i_closest;
    // setClosestGridPointIdx(x_d, y_d, n, m, v_i_closest, y_i_closest);
    // u_i_closest = v_i_closest - 1;

    ////int closest_x = (u_i_closest / 2);
    ////int closest_y = y_i_closest ;
    ////shared memory bound
    ////if ((closest_x+1) / PADDED_SIZE == blockIdx.x && (closest_y+1)/ PADDED_SIZE == blockIdx.y)
    ////if (closest_x >= blockIdx.x * TILE_SIZE &&
    ////closest_x + 1 < (blockIdx.x * TILE_SIZE + PADDED_SIZE) &&
    ////closest_y >= blockIdx.y * TILE_SIZE &&
    ////closest_y + 1 < (blockIdx.y * TILE_SIZE + PADDED_SIZE))
    ////{
    ////int t_closest_x = closest_x % PADDED_SIZE;
    ////int t_closest_y = closest_y % PADDED_SIZE;
    ////double2 test;
    ////test.x = periodic_grid[periodic_linear_Idx(u_i_closest, y_i_closest,2*n,m)];
    ////test.y = periodic_grid[periodic_linear_Idx(v_i_closest, y_i_closest,2*n,m)];

    ////double2 closest = PERIODIC[t_closest_y * PADDED_SIZE + t_closest_x];
    ////assert(test.x == closest.x);
    ////assert(test.y == closest.y);

    //////double dx = (periodic_end - periodic_start) / (n - 1);
    //////double dy = (periodic_end - periodic_start ) / (m - 1);
    ////double x_diff = (x_d - closest.x) / dx;
    ////double y_diff = (y_d - closest.y) / dx;

    ////double weight_contained = (1.0 - x_diff) * (1.0 - y_diff);
    ////double weight_next_x = x_diff * (1.0 - y_diff);
    ////double weight_next_x_y = x_diff * y_diff;
    ////double weight_next_y = (1.0 - x_diff) * y_diff;

    //////double sum_weights = weight_contained + weight_next_x + weight_next_y + weight_next_x_y;
    //////assert(is_close(sum_weights,1.0));
    ////double2 interpolated;
    ////interpolated.x = weight_contained * VEL[t_closest_y * PADDED_SIZE + t_closest_x].x;
    ////interpolated.y = weight_contained * VEL[t_closest_y * PADDED_SIZE + t_closest_x].y;

    ////double u_test = weight_contained * velocity_grid[periodic_linear_Idx(u_i, y_i_closest,2*n,m)];
    ////assert(interpolated.x==u_test);

    ////// y_direction next grid cell
    ////interpolated.x += weight_next_y * VEL[(t_closest_y+1)*PADDED_SIZE+t_closest_x].x;
    ////interpolated.y += weight_next_y * VEL[(t_closest_y+1)*PADDED_SIZE+t_closest_x].y;
    ////////// x_direction next grid cell
    ////interpolated.x += weight_next_x * VEL[t_closest_y*PADDED_SIZE+(t_closest_x+1)].x;
    ////interpolated.y += weight_next_x * VEL[t_closest_y*PADDED_SIZE+(t_closest_x+1)].y;
    ////////// next grid cell in diagonal direction
    ////interpolated.x += weight_next_x_y * VEL[(t_closest_y+1)*PADDED_SIZE+(t_closest_x+1)].x;
    ////interpolated.y += weight_next_x_y * VEL[(t_closest_y+1)*PADDED_SIZE+(t_closest_x+1)].y;
    ////// forward bilinear interpolation
    ////velocity_grid_next[periodic_linear_Idx(u_i,row,2*n,m)] = interpolated.x;
    ////velocity_grid_next[periodic_linear_Idx(v_i,row,2*n,m)] = interpolated.y;
    ////}
    ////else{
    // double x_closest = periodic_grid[periodic_linear_Idx(u_i_closest, y_i_closest,2*n,m)];
    // double y_closest = periodic_grid[periodic_linear_Idx(v_i_closest, y_i_closest,2*n,m)];
    ////double dx = (periodic_end - periodic_start) / (n - 1);
    ////double dy = (periodic_end - periodic_start ) / (m - 1);
    // double x_diff = (x_d - x_closest) / dx;
    // double y_diff = (y_d - y_closest) / dx;

    //// forward bilinear interpolation
    ////TODO: weights are calculated twice
    // velocity_grid_next[periodic_linear_Idx(u_i,row,2*n,m)] =
    // get_interpolated(u_i_closest,y_i_closest,x_diff,y_diff,velocity_grid,n,m);
    // velocity_grid_next[periodic_linear_Idx(v_i,row,2*n,m)] =
    // get_interpolated(v_i_closest,y_i_closest,x_diff,y_diff,velocity_grid,n,m);
    ////}
    //}
    //}

    __global__ void integrateAndInterpolateKernel(const double *periodic_grid, const double *velocity_grid, double *velocity_grid_next, const double dt, const int n, const int m, const double dx)
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;
        int u_i = col * 2;
        int v_i = (col * 2) + 1;
        if (row < m && col < n)
        {
            double x_d, y_d, u, v;
            integrateEuler(velocity_grid, row, u_i, v_i, periodic_grid, x_d, y_d, dt, n, m);
            interpolateVelocity(u, v, x_d, y_d, periodic_grid, velocity_grid, n, m, dx);
            velocity_grid_next[periodic_linear_Idx(u_i, row, 2 * n, m)] = u;
            velocity_grid_next[periodic_linear_Idx(v_i, row, 2 * n, m)] = v;
        }
    }
    __global__ void macCormackCorrectionKernel(double *velocity_grid, const double *velocity_grid_bw, const double *velocity_grid_bw_fw, int n, int m)
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;
        double v = 0;
        if (row < m && col < 2 * n)
        {
            v = gpu::mac_cormack_correction(col, row, velocity_grid, velocity_grid_bw, velocity_grid_bw_fw, n, m);
        }
        __syncthreads();
        if (row < m && col < 2 * n)
        {
            velocity_grid[periodic_linear_Idx(col, row, 2 * n, m)] = v;
        }
    }

    __device__ double mac_cormack_correction(const int idx_x, const int y_i, const double *velocity_grid, const double *velocity_grid_bw, const double *velocity_grid_bw_fw, int n, int m)
    {
        double bw = velocity_grid_bw[periodic_linear_Idx(idx_x, y_i, 2 * n, m)];
        double bw_fw = velocity_grid_bw_fw[periodic_linear_Idx(idx_x, y_i, 2 * n, m)];
        double field = velocity_grid[periodic_linear_Idx(idx_x, y_i, 2 * n, m)];
        // double out_val = 0.5 * (bw + fw); // temporal average
        double out_val = bw + MACCORMACK_CORRECTION * 0.5 * (field - bw_fw); // like in PHIflow, but clashes with wikipedia-definition
        double min_ = 1e6, max_ = 1e-6;
        // clipping where it looks more like the simulation of phiflow, but phiflow clips with closest in velocity grid, but of backward integrated points
        gpu::min_max_neighbors(min_, max_, idx_x, y_i, velocity_grid, n, m);
        gpu::clip(out_val, min_, max_);
        // phiflow clipping destroys simulation completly
        // gpu::min_max_neighbors(min_,max_,bw_idx_x,bw_idx_y,velocity_grid,n,m);
        // gpu::clip(out_val,min_,max_);
        return out_val;
    }

    __device__ void min_max_neighbors(double &min, double &max, const int idx, const int y_i, const double *velocity_grid, const int n, const int m)
    {
        // TODO: parallelize
        double neighbors[4];
        neighbors[0] = velocity_grid[periodic_linear_Idx(idx - 2, y_i, 2 * n, m)];
        neighbors[1] = velocity_grid[periodic_linear_Idx(idx + 2, y_i, 2 * n, m)];
        neighbors[2] = velocity_grid[periodic_linear_Idx(idx, y_i - 1, 2 * n, m)];
        neighbors[3] = velocity_grid[periodic_linear_Idx(idx, y_i + 1, 2 * n, m)];

        min = neighbors[0];
        max = neighbors[0];
        for (int i = 1; i < 4; i++)
        {
            if (neighbors[i] < min)
            {
                min = neighbors[i];
            }
            if (neighbors[i] > max)
            {
                max = neighbors[i];
            }
        }
    }
}

double mac_cormack_correction(const int idx_x, const int y_i, const double *velocity_grid, const double *velocity_grid_bw, const double *velocity_grid_fw, int n, int m)
{
    double bw = velocity_grid_bw[periodic_linear_Idx(idx_x, y_i, 2 * n, m)];
    double fw = velocity_grid_fw[periodic_linear_Idx(idx_x, y_i, 2 * n, m)];
    double field = velocity_grid[periodic_linear_Idx(idx_x, y_i, 2 * n, m)];
    // double out_val = 0.5 * (bw + fw); // temporal average
    double out_val = bw + MACCORMACK_CORRECTION * 0.5 * (field - fw); // like in PHIflow, but clashes with wikipedia-definition
    // clipping
    double min_, max_;
    min_max_neighbors(min_, max_, idx_x, y_i, velocity_grid, n, m);
    // clip(out_val,0.75*min_,2*max_);
    clip(out_val, min_, max_);
    return out_val;
}

void min_max_neighbors(double &min, double &max, const int idx, const int y_i, const double *velocity_grid, const int n, const int m)
{
    double neighbors[4];
    neighbors[0] = velocity_grid[periodic_linear_Idx(idx - 2, y_i, 2 * n, m)];
    neighbors[1] = velocity_grid[periodic_linear_Idx(idx + 2, y_i, 2 * n, m)];
    neighbors[2] = velocity_grid[periodic_linear_Idx(idx, y_i - 1, 2 * n, m)];
    neighbors[3] = velocity_grid[periodic_linear_Idx(idx, y_i + 1, 2 * n, m)];

    min = neighbors[0];
    max = neighbors[0];
    for (int i = 1; i < 4; i++)
    {
        if (neighbors[i] < min)
        {
            min = neighbors[i];
        }
        if (neighbors[i] > max)
        {
            max = neighbors[i];
        }
    }
}
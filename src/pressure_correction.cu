#include "pressure_correction.cuh"

void makeIncompressible(double *velocity_grid, double *divergence, double *pressure, double *laplace, int n, int m, const double dx)
{
    calculateDivergence(velocity_grid, divergence, n, n, dx);
    // cuBlas is column major
    // switchRowColMajor(divergence,m,n); //not needed, solver interprets this as 1D-vector anyways

    // double *laplace = (double *)malloc(n * m * n * m * sizeof(double));
    // std::cout << "Divergence" << std::endl;
    // print_matrix(m, n, divergence, n);
    // std::cout << "Laplacian" << std::endl;
    // constructDiscretizedLaplacian(laplace,n,dx); // LP^T = LP -> no need to transpose

    // print_matrix(n*m, n*m, laplace, n*m);
    size_t pressure_size = n * m;
    solveDense(laplace, divergence, pressure, pressure_size);
    // std::cout << "Pressure" << std::endl;
    // switchRowColMajor(pressure,n,m);
    // print_matrix_row_major(m, n, pressure, n);
    // free(laplace);
    correct_velocity(velocity_grid, pressure, n, m, dx);
}
void correct_velocity(double *velocity_grid, double *pressure, int n, int m, double dx)
{
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 0; i < n; i++)
        {
            int u_i = 2 * i;
            int v_i = 2 * i + 1;
            double p = pressure[periodic_linear_Idx(i, y_i, n, m)];
            double p_right = pressure[periodic_linear_Idx(i + 1, y_i, n, m)];
            double p_down = pressure[periodic_linear_Idx(i, y_i + 1, n, m)];
            double p_dx = (p_right - p) / dx;
            double p_dy = (p_down - p) / dx;

            velocity_grid[periodic_linear_Idx(u_i, y_i, 2 * n, m)] -= p_dx;
            velocity_grid[periodic_linear_Idx(v_i, y_i, 2 * n, m)] -= p_dy;
        }
    }
}

namespace gpu
{
    void makeIncompressibleSparse(double *velocity_grid, double *divergence, double *pressure, double *lp_values, int *lp_columns, int *lp_row_offsets, int n, int m, double dx)
    {

        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        gpu::calculateDivergence<<<gridDim, blockDim>>>(velocity_grid, divergence, n, m, dx);
        CHECK_CUDA(cudaDeviceSynchronize());

        gpu::solveSparse(lp_values, lp_columns, lp_row_offsets, divergence, pressure, 5 * n * m, n * m);
        CHECK_CUDA(cudaDeviceSynchronize());

        // TODO: parallelize u and v correction?
        dim3 gridDimVel((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        gpu::correct_velocity<<<gridDimVel, blockDim>>>(velocity_grid, pressure, n, m, dx);
    }

    void makeIncompressible(double *velocity_grid, double *d_B, double *laplace, int n, int m, double dx)
    {
        /*d_B is used for divergence and pressure data*/

        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        gpu::calculateDivergence<<<gridDim, blockDim>>>(velocity_grid, d_B, n, m, dx);
        CHECK_CUDA(cudaDeviceSynchronize());

        // laplace only stays constant if no pivot is used!
        gpu::solveDense(laplace, d_B, n * m);
        CHECK_CUDA(cudaDeviceSynchronize());

        // TODO: parallelize u and v correction?
        dim3 gridDimVel((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        gpu::correct_velocity<<<gridDimVel, blockDim>>>(velocity_grid, d_B, n, m, dx);
    }

    __global__ void correct_velocity(double *velocity_grid, double *pressure, int n, int m, double dx)
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;
        if (row < m && col < n)
        {
            int u_i = 2 * col;
            int v_i = 2 * col + 1;
            double p = pressure[periodic_linear_Idx(col, row, n, m)];
            double p_right = pressure[periodic_linear_Idx(col + 1, row, n, m)];
            double p_down = pressure[periodic_linear_Idx(col, row + 1, n, m)];
            double p_dx = (p_right - p) / dx;
            double p_dy = (p_down - p) / dx;

            velocity_grid[periodic_linear_Idx(u_i, row, 2 * n, m)] -= p_dx;
            velocity_grid[periodic_linear_Idx(v_i, row, 2 * n, m)] -= p_dy;
        }
        else
        {
            return;
        }
    }
}

void calculateDivergence(const double *velocity_grid, double *divergence, int n, int m, const double dx)
{

    // divergence by central differences
    // double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    // double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 0; i < n; i++)
        {
            int u_i = 2 * i;
            int v_i = 2 * i + 1;
            double u = velocity_grid[periodic_linear_Idx(u_i, y_i, 2 * n, m)];
            double v = velocity_grid[periodic_linear_Idx(v_i, y_i, 2 * n, m)];

            double u_left = velocity_grid[periodic_linear_Idx(u_i - 2, y_i, 2 * n, m)];
            double u_right = velocity_grid[periodic_linear_Idx(u_i + 2, y_i, 2 * n, m)];

            double v_down = velocity_grid[periodic_linear_Idx(v_i, y_i + 1, 2 * n, m)];
            double v_up = velocity_grid[periodic_linear_Idx(v_i, y_i - 1, 2 * n, m)];
            // central differences
            // double div = (u_right - u_left) / (2 * dx) + (v_up - v_down) / (2 * dy);
            double div = (u_right - u_left) / (2 * dx) + (v_down - v_up) / (2 * dx);
            // backward differences
            // double div = (u - u_left) / dx + (v - v_down) / dy;
            divergence[periodic_linear_Idx(i, y_i, n, m)] = div;
        }
    }
}

namespace gpu
{

    __global__ void calculateDivergence(const double *velocity_grid, double *divergence, int n, int m, const double dx)
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;

        // int u_i = 2 * col;
        // int v_i = 2 * col + 1;
        // const dim3 PADDED_SIZE(TILE_SIZE+2,TILE_SIZE+2);
        const int PADDED_SIZE = TILE_SIZE + 2;
        __shared__ double2 VEL[(TILE_SIZE + 2) * (TILE_SIZE + 2)];
        //__shared__ double DIV[(TILE_SIZE) * (TILE_SIZE)];
        auto t_col = threadIdx.x + 1;
        auto t_u_i = 2 * t_col;
        auto t_v_i = 2 * t_col + 1;
        auto t_row = threadIdx.y + 1;
        if (row < m && col < 2 * n)
        {
            if (threadIdx.x < blockDim.x && threadIdx.y < blockDim.y)
            {
                // fill inner
                double2 temp;
                temp.x = velocity_grid[periodic_linear_Idx(t_u_i, row, 2 * n, m)];
                temp.y = velocity_grid[periodic_linear_Idx(t_v_i, row, 2 * n, m)];
                VEL[t_row * PADDED_SIZE + t_col] = temp;

                // boundary
                if (threadIdx.y == 0)
                { // top row
                    temp.x = velocity_grid[periodic_linear_Idx(t_u_i, row - 1, 2 * n, m)];
                    temp.y = velocity_grid[periodic_linear_Idx(t_v_i, row - 1, 2 * n, m)];
                    VEL[(t_row - 1) * PADDED_SIZE + t_col] = temp;
                    // printf("Shared v_up : %f\n", temp.y);
                }
                if (threadIdx.y == blockDim.y - 1)
                {
                    temp.x = velocity_grid[periodic_linear_Idx(t_u_i, row + 1, 2 * n, m)];
                    temp.y = velocity_grid[periodic_linear_Idx(t_v_i, row + 1, 2 * n, m)];
                    VEL[(t_row + 1) * PADDED_SIZE + t_col] = temp;
                }
                if (threadIdx.x == 0) // left
                {
                    temp.x = velocity_grid[periodic_linear_Idx(t_u_i - 2, row, 2 * n, m)];
                    temp.y = velocity_grid[periodic_linear_Idx(t_v_i - 2, row, 2 * n, m)];
                    VEL[t_row * PADDED_SIZE + (t_col - 1)] = temp;
                }
                if (threadIdx.x == blockDim.x - 1) // right
                {
                    temp.x = velocity_grid[periodic_linear_Idx(t_u_i + 2, row, 2 * n, m)];
                    temp.y = velocity_grid[periodic_linear_Idx(t_v_i + 2, row, 2 * n, m)];
                    VEL[t_row * PADDED_SIZE + (t_col + 1)] = temp;
                }
            }
        }

        __syncthreads();
        if ((row < m) && (col < n))
        {
            if (threadIdx.y < blockDim.y && threadIdx.x < blockDim.x)
            {

                double2 left = VEL[t_row * PADDED_SIZE + t_col - 1];
                double2 right = VEL[t_row * PADDED_SIZE + t_col - 1];

                double2 down = VEL[(t_row + 1) * PADDED_SIZE + t_col];
                double2 up = VEL[(t_row - 1) * PADDED_SIZE + t_col];
                // central differences

                // backward differences
                // double div = (u - u_left) / dx + (v - v_down) / dy;
                divergence[periodic_linear_Idx(col, row, n, m)] =
                    (right.x - left.x) / (2 * dx) + (down.y - up.y) / (2 * dx);
                // divergence[periodic_linear_Idx(col,row,n,m)] = VEL[(t_row+1) * PADDED_SIZE.x + t_col].y;
            }
        }
    }
}

void constructDiscretizedLaplacian(double *laplace_discrete, int n, const double dx)
{
    // discretized laplacian is always same for grid -> unit laplacian is sufficient
    // order 2
    std::vector<double> lp_(n * n * n * n, 0);
    std::copy(lp_.begin(), lp_.end(), laplace_discrete);
    for (int lp_row = 0; lp_row < n * n; lp_row++)
    {
        // one lp_row has one entry for all entries in source martrix
        int src_y = lp_row / n;
        int src_x = lp_row % n;

        int lp_center = src_y * n + src_x;
        laplace_discrete[lp_row * n * n + lp_center] = -4.0 / (dx * dx);
        // neighbors
        int up_src = periodic_linear_Idx(src_x, src_y - 1, n, n);
        int down_src = periodic_linear_Idx(src_x, src_y + 1, n, n);
        int left_src = periodic_linear_Idx(src_x - 1, src_y, n, n);
        int right_src = periodic_linear_Idx(src_x + 1, src_y, n, n);
        laplace_discrete[lp_row * n * n + up_src] = 1.0 * (dx * dx);
        laplace_discrete[lp_row * n * n + down_src] = 1.0 * (dx * dx);
        laplace_discrete[lp_row * n * n + left_src] = 1.0 * (dx * dx);
        laplace_discrete[lp_row * n * n + right_src] = 1.0 * (dx * dx);
    }
}

void constructLaplaceSparseCSR(double *values, int *row_offsets, int *col_indices, const int n, const double dx)
{

    // nonzero count 5 per row -> 5*n*n -> allocate val, offsets, cols accordingly
    //
    // int nnz = 5*n*n;
    double diag_value = -4.0 / (dx * dx);
    double neighbor_value = 1.0 / (dx * dx);
    int current_nnz = 0;
    row_offsets[0] = 0;
    for (int row = 0; row < n * n; row++)
    {

        int src_x = row % n;
        int src_y = row / n;

        col_indices[current_nnz] = row;
        values[current_nnz] = diag_value;
        current_nnz++;

        int up_src = periodic_linear_Idx(src_x, src_y - 1, n, n);
        col_indices[current_nnz] = up_src;
        values[current_nnz] = neighbor_value;
        current_nnz++;

        int down_src = periodic_linear_Idx(src_x, src_y + 1, n, n);
        col_indices[current_nnz] = down_src;
        values[current_nnz] = neighbor_value;
        current_nnz++;

        int left_src = periodic_linear_Idx(src_x - 1, src_y, n, n);
        col_indices[current_nnz] = left_src;
        values[current_nnz] = neighbor_value;
        current_nnz++;

        int right_src = periodic_linear_Idx(src_x + 1, src_y, n, n);
        col_indices[current_nnz] = right_src;
        values[current_nnz] = neighbor_value;
        current_nnz++;

        row_offsets[row + 1] = current_nnz; // is (0,5,10...) can also do it separately
    }
}

namespace gpu
{

    __global__ void constructLaplaceSparseCSR(double *values, int *row_offsets, int *col_indices, const int n, const double dx)
    {
        int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_nnz = 5 * n * n; // only threads for non-zeros
        if (global_idx < total_nnz)
        {

            double diag_value = -4.0 / (dx * dx);
            double neighbor_value = 1.0 / (dx * dx);

            int row = global_idx / 5;
            int local_idx = global_idx % 5; // per row (0-4)

            int src_x = row % n;
            int src_y = row / n;

            // row_offsets only for the first thread in each row
            if (local_idx == 0 && row == 0)
                row_offsets[0] = 0;
            if (local_idx == 0)
                row_offsets[row + 1] = global_idx + 5;

            switch (local_idx)
            {
            case 0:
                values[global_idx] = diag_value;
                col_indices[global_idx] = row;
                break;
            case 1: // up
                values[global_idx] = neighbor_value;
                col_indices[global_idx] = periodic_linear_Idx(src_x, src_y - 1, n, n);
                break;
            case 2: // down
                values[global_idx] = neighbor_value;
                col_indices[global_idx] = periodic_linear_Idx(src_x, src_y + 1, n, n);
                break;
            case 3: // left
                values[global_idx] = neighbor_value;
                col_indices[global_idx] = periodic_linear_Idx(src_x - 1, src_y, n, n);
                break;
            case 4: // right
                values[global_idx] = neighbor_value;
                col_indices[global_idx] = periodic_linear_Idx(src_x + 1, src_y, n, n);
                break;
            }
        }
    }

    __global__ void fillLaplaceValues(double *laplace_discrete, int n, const double dx)
    {
        /*CAUTION: expects 0 inititilized laplace_discrete*/
        int lp_idx = threadIdx.x + blockIdx.x * blockDim.x;

        // if (lp_idx < n*n)
        // for (int i= 0; i< n*n*n*n; i++){
        // laplace_discrete[lp_idx * n*n + i] =0.0;
        //}
        // laplace_discrete[lp_idx*2] = 0.0;
        // laplace_discrete[lp_idx*1] = 0.0;
        //__syncthreads();
        if (lp_idx < n * n)
        {
            // one lp_row has one entry for all entries in source martrix
            int src_y = lp_idx / n;
            int src_x = lp_idx % n;

            int lp_center = src_y * n + src_x;
            laplace_discrete[lp_idx * n * n + lp_center] = -4.0 / (dx * dx);
            // neighbors
            int up_src = periodic_linear_Idx(src_x, src_y - 1, n, n);
            int down_src = periodic_linear_Idx(src_x, src_y + 1, n, n);
            int left_src = periodic_linear_Idx(src_x - 1, src_y, n, n);
            int right_src = periodic_linear_Idx(src_x + 1, src_y, n, n);
            double neighbor_weight = 1.0 * (dx * dx);
            laplace_discrete[lp_idx * n * n + up_src] = neighbor_weight;
            laplace_discrete[lp_idx * n * n + down_src] = neighbor_weight;
            laplace_discrete[lp_idx * n * n + left_src] = neighbor_weight;
            laplace_discrete[lp_idx * n * n + right_src] = neighbor_weight;
        }
    }

    void constructDiscretizedLaplacian(double *laplace_discrete, int n, const double dx)
    {
        CHECK_CUDA(cudaMemset(laplace_discrete, 0, n * n * n * n * sizeof(double)));
        dim3 blockDim(TILE_SIZE);
        dim3 gridDim((n * n + TILE_SIZE - 1) / TILE_SIZE);
        gpu::fillLaplaceValues<<<gridDim, blockDim>>>(laplace_discrete, n, dx);
    }

}
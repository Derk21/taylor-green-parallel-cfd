#include "advect.cuh"
#include "init.cuh"
#include <cassert>


void test_interpolation(){
    int n = 4;
    int m = 4;
    double *periodic_grid = (double *)malloc(n * m * 2 * sizeof(double));
    double *velocity_grid = (double *)malloc(n * m * 2 * sizeof(double));
    double *velocity_grid_next = (double *)malloc(n * m * 2 * sizeof(double));
    if (periodic_grid == NULL || velocity_grid == NULL || velocity_grid_next == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return exit(EXIT_FAILURE);
    }
    double periodic_start = 1.0;
    double periodic_end = 4.0;
    // Initialize periodic grid
    initializePeriodicGrid(periodic_grid,n,m,1.0,4.0); 
    //for (int y_i = 0; y_i < m; y_i++) {
        //for (int i = 1; i < 2 * n; i+=2) {
            //periodic_grid[periodic_linear_Idx(i-1, y_i,2*n,m)] = i * 1.0 + y_i;
            //periodic_grid[periodic_linear_Idx(i, y_i,2*n,m)] = i * 1.0 + y_i;
            ////std::cout << periodic_grid[periodic_linear_Idx(i,y_i)] << " ";
        //}
        ////std::cout << "\n";
    //}
    std::cout << "periodic grid" <<std::endl;
    //print_matrix_row_major(m,2*n,periodic_grid,2*n);

    // Initialize velocity grid with known values
    for (int y_i = 0; y_i < m; y_i++) {
        for (int i = 0; i < 2 * n; i++) {
            velocity_grid[periodic_linear_Idx(i, y_i,2*n,m)] = i + y_i;
            velocity_grid_next[periodic_linear_Idx(i, y_i,2*n,m)] = i + y_i;
            //velocity_grid[periodic_linear_Idx(i, y_i,2*n,m)] = 0.0;
            //velocity_grid_next[periodic_linear_Idx(i, y_i,2*n,m)] = 0.0;
        }
    }
    std::cout << "velocity grid " <<std::endl;
    //print_matrix_row_major(m,2*n,velocity_grid,2*n);

    // Test point for interpolation
    double x_d = 0.25;
    double y_d = 0.25;

    // Expected results for bilinear interpolation
    double expected_u = (0.75*0.75) * 0.0 + (0.75*0.25) * 2.0 + (0.75*0.25) * 1 + (0.25*0.25) * 3;
    double expected_v = (0.75*0.75) * 1.0 + (0.75*0.25) * 3.0 + (0.75*0.25) * 2 + (0.25*0.25) * 4;
    //double expected_v = 0.25 * 1.0 + 0.25 * 3.0 + 0.25 * 2 + 0.25 * 4;
    //double expected_v = 1.5 * (1.0 - 1.5) * velocity_grid[periodic_linear_Idx(1, 1,2*n,m)] +
                        //1.5 * 1.5 * velocity_grid[periodic_linear_Idx(1, 2,2*n,m)] +
                        //(1.0 - 1.5) * (1.0 - 1.5) * velocity_grid[periodic_linear_Idx(3, 1,2*n,m)] +
                        //(1.0 - 1.5) * 1.5 * velocity_grid[periodic_linear_Idx(3, 2,2*n,m)];

    // Perform interpolation
    interpolateVelocity(x_d, y_d, n, m,periodic_start,periodic_end, periodic_grid, velocity_grid, velocity_grid_next);
    std::cout << "velocity grid after interpolation" <<std::endl;
    //print_matrix_row_major(m,2*n,velocity_grid_next,2*n);
    double u_interpolated = velocity_grid_next[periodic_linear_Idx(0,0,2*n,m)];
    double v_interpolated = velocity_grid_next[periodic_linear_Idx(1,0,2*n,m)];
    std::cout <<"expected velocity at intepolated 1.5 point: " << expected_u << "," << expected_v << std::endl;
    std::cout <<"actual velocity at intepolated 1.5 point: " << u_interpolated << "," <<v_interpolated << std::endl;
    assert(is_close(expected_u,u_interpolated));
    assert(is_close(expected_v,v_interpolated));

    //// Check results
    //assert(is_close(velocity_grid_next[periodic_linear_Idx(0,0,2*n,m)], expected_u));
    //assert(is_close(velocity_grid_next[periodic_linear_Idx(1, 1,2*n,m)], expected_v));

    free(periodic_grid);
    free(velocity_grid);
    free(velocity_grid_next);
    std::cout << "Interpolation test passed!" << std::endl;

}

int main()
{
    test_interpolation();
    std::cout << "All tests passed" << std::endl;
}

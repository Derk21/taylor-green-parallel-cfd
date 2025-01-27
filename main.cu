#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
//#include <cusolverSp.h>
#include <cusolverDn.h>
#include <iomanip>
#include "gnuplot-iostream.h"
#include "plotting.h"
#include "constants.h"
#include "utils.h"
#include "init.h"
#include "advect.h"
#include "diffuse.h"
#include <cassert>


void calculateDivergence(const double* velocity_grid,double*divergence,int n=N, int m=M){

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

void jacobiStep(double*pressure,double * pressure_next, double* divergence,int n=N, int m=M)
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

void constructDiscretizedLaplacian(double* laplace_discrete,int n=N){
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
        laplace_discrete[lp_row * n*n + lp_center] = 4.0;
        //neighbors
        int up_src = periodic_linear_Idx(src_x, src_y - 1, n, n);
        int down_src = periodic_linear_Idx(src_x, src_y + 1, n, n);
        int left_src = periodic_linear_Idx(src_x - 1, src_y, n, n);
        int right_src = periodic_linear_Idx(src_x + 1, src_y, n, n);
        laplace_discrete[lp_row * n * n + up_src] = -1.0;
        laplace_discrete[lp_row * n * n + down_src] = -1.0;
        laplace_discrete[lp_row * n * n + left_src] = -1.0;
        laplace_discrete[lp_row * n * n + right_src] = -1.0;
    }
    //print_matrix(n*n, n*n, laplace_discrete, n*n);
    //test
    //for (int i = 0; i < n*n; i++)
    //{
        //double row_sum = 0.0;
        //for (int j = 0; j < n*n; j++)
        //{
            //row_sum += laplace_discrete[i*n*n+j];
        //}
        //assert(round(row_sum) == 0.0);
    //}
    //std::cout << "Laplacian has correct row sums" << std::endl;
}

//void solveDense(const std::vector<double> &A, const std::vector<double>& B, std::vector<double> & X, size_t m=2*N){
void solveDense(const double * A, const double *  B, double * X, size_t m=2*N){
    //A is discretized laplacian
    //b is divergence (flat)
    //x is pressure (flat)

    //LU decomposition with partial pivoting
    //needs to be double for cusolver
    //probably many are 0 -> TODO: sparse solver

    //adapted from https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSOLVER/getrf
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    const int lda = m;
    const int ldb = m;

    std::vector<double> LU(lda * m, 0);
    std::vector<int> Ipiv(m, 0);
    int info = 0;

    double *d_A = nullptr; /* device copy of A */
    double *d_B = nullptr; /* device copy of B */
    int *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr; /* error info */

    int lwork = 0;            /* size of workspace */
    double *d_work = nullptr; /* device workspace for getrf */

    const int pivot_on = 1;

    if (pivot_on) {
        printf("pivot is on : compute P*A = L*U \n");
    } else {
        printf("pivot is off: compute A = L*U (not numerically stable)\n");
    }

    //printf("A = (matlab base-1)\n");
    //print_matrix(m, m, A, lda);
    //printf("=====\n");

    //printf("B = (matlab base-1)\n");
    //print_matrix(m, 1, B, ldb);
    //printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* step 2: copy A to device */
    CHECK_CUDA(cudaMalloc(&d_A, sizeof(double) * m * m));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(double) * m));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int) * Ipiv.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CHECK_CUDA(
        cudaMemcpyAsync(d_A, A, sizeof(double) * m * m, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(
        cudaMemcpyAsync(d_B, B, sizeof(double) * m, cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of getrf */
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork));

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    /* step 4: LU factorization */
    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, NULL, d_info));
    }

    if (pivot_on) {
        CHECK_CUDA(cudaMemcpyAsync(Ipiv.data(), d_Ipiv, sizeof(int) * Ipiv.size(),
                                   cudaMemcpyDeviceToHost, stream));
    }
    //CHECK_CUDA(
        //cudaMemcpyAsync(LU.data(), d_A, sizeof(double) * m * m, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (0 > info) {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    //if (pivot_on) {
        //printf("pivoting sequence, matlab base-1\n");
        //for (int j = 0; j < m; j++) {
            //printf("Ipiv(%d) = %d\n", j + 1, Ipiv[j]);
        //}
    //}
    //printf("L and U = (matlab base-1)\n");
    //print_matrix(m, m, LU.data(), lda);
    //printf("=====\n");

    
    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */
                                        d_A, lda, d_Ipiv, d_B, ldb, d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */
                                        d_A, lda, NULL, d_B, ldb, d_info));
    }

    CHECK_CUDA(
        cudaMemcpyAsync(X, d_B, sizeof(double) * m, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    //printf("X = (matlab base-1)\n");
    //print_matrix(m, 1, X, ldb);
    //printf("=====\n");

    /* free resources */
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_Ipiv));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUDA(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaDeviceReset());
   }

void testSolveDense(){
    int m = 3;
    std::vector<double> A_ = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
    double * A = (double *)malloc(m*m * sizeof(double));
    double * B = (double *)malloc(m * sizeof(double));
    double * X = (double *)malloc(m * sizeof(double));
    for (int i = 0; i < m; i++)
    {
        X[i] = 0.0;
        B[i] = i + 1;
    }
    std::copy(A_.begin(), A_.end(), A);
    solveDense(A,B,X,m);
    print_matrix(m, 1, X, m);
    /*
     * step 5: solve A*X = B
     *       | 1 |       | -0.3333 |
     *   B = | 2 |,  X = |  0.6667 |
     *       | 3 |       |  0      |
     *
     */
}

double get_max_diff(double* a, double* b, int n){
    //mean reduction
    double maximum = 0.0;
    for (int i = 0; i < n; i++)
    {
        maximum = max(abs(a[i]-b[i]),maximum);
    }
    return maximum;
}

void make_incompressible(double* velocity_grid, double* divergence, double*pressure, int n=N, int m=M){
    calculateDivergence(velocity_grid,divergence);

    //THERE IS NO EXPLICIT SOLVER FOR PRESSURE CORRECTION (https://en.wikipedia.org/wiki/Discrete_Poisson_equation)
    //explicit solve for now 
    //const double TOLERANCE = 1.0;
    //double *pressure_next = (double *)malloc(n * m * sizeof(double));
    //memcpy(pressure_next,pressure,n*m*sizeof(double));
    //for(int i = 0; i < 10000; i++){
        //jacobiStep(pressure,pressure_next,divergence);
        //double loss = get_max_diff(pressure,divergence,n*m);
        //std::cout << "Loss: " << loss << std::endl;
        //if (loss < TOLERANCE){
            //break;
        //}
        ////TODO: l1 norm of pressure - divergence
    //}

    //free(pressure_next);
    //TODO: implicit solver 
    double *laplace = (double *)malloc(n * m * n * m * sizeof(double));
    //std::cout << "Divergence" << std::endl;
    //print_matrix(m, n, divergence, n);
    //std::cout << "Laplacian" << std::endl;
    constructDiscretizedLaplacian(laplace);
    //print_matrix(n*m, n*m, laplace, n*m);
    solveDense(laplace,divergence,pressure,n*m);
    std::cout << "Pressure" << std::endl;
    print_matrix(m, n, pressure, n);
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
        }
    }
    

}


int main()
{   
    test_setClosestGridPointIdx();
    int s = 4;
    double * lp = (double *)malloc(s*s*s*s * sizeof(double));
    if (lp == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }
    constructDiscretizedLaplacian(lp,s);
    //std::cout << "Laplacian" << std::endl;
    //print_matrix(8, 8, lp, 8);
    free(lp);
    //testSolveDense();
    //vortex decays exponentially -> use double to stabilize
    double *periodic_grid = (double *)malloc(N * M * 2 * sizeof(double));
    double *velocity_grid = (double *)malloc(N * M * 2 * sizeof(double));
    double *velocity_grid_next = (double *)malloc(N * M * 2 * sizeof(double));
    //doubles to make compatible with cuSolver
    double *divergence = (double *)malloc(N * M * sizeof(double));
    double *pressure = (double *)malloc(N * M * sizeof(double));


    // Check for allocation failures
    if (periodic_grid == NULL || velocity_grid == NULL || velocity_grid_next == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    //setPressureGroundTruth(pressure,periodic_grid,1,N,M);
    initilizePressure(pressure,N,M);
    initializePeriodicGrid(periodic_grid,N,M);
    initilizeVelocityGrid(velocity_grid,periodic_grid,N,M);
    memcpy(velocity_grid_next,velocity_grid,N*M*2*sizeof(double));
    //double *d_curr;
    //allocate memory on device    
    //CHECK_CUDA(cudaMalloc(&d_curr, N * M * sizeof(double)));
    //copy data to device
    //CHECK_CUDA(cudaMemcpy(d_curr, curr, N * M * sizeof(double), cudaMemcpyHostToDevice));
    std::string dirName = createTimestampedDirectory();
    //plotPeriodicGrid(periodic_grid, N, M);
    std::string plot_name("velocity_0000");
    plotVelocityGrid(periodic_grid, velocity_grid, N, M, PERIODIC_START, PERIODIC_END,plot_name, dirName);
    for (int i = 1; i < ITERATIONS+1; i++){
        diffuseExplicit(velocity_grid,velocity_grid_next,N,M);
        std::stringstream plot_name;
        //plot_name << "velocity_"<< std::setw(4) << std::setfill('0') << i << "_diffused";
        //plotVelocityGrid(periodic_grid, velocity_grid, N, M, PERIODIC_START, PERIODIC_END,plot_name.str(), dirName);
        //plot_name.str("");
        advectSemiLagrange(velocity_grid,velocity_grid_next,periodic_grid,TIMESTEP,N,M);
        make_incompressible(velocity_grid,divergence,pressure);
        //taylorGreenGroundTruth(periodic_grid,velocity_grid_next,i,N,M);
        //std::swap(velocity_grid,velocity_grid_next);
        plot_name << "velocity_"<< std::setw(4) << std::setfill('0') << i;
        plotVelocityGrid(periodic_grid, velocity_grid, N, M, PERIODIC_START, PERIODIC_END,plot_name.str(), dirName);
        plot_name.str("");
    }
    std::cout << "Creating velocity animation" << std::endl;
    createGifFromPngs(dirName,"animation_velocity.gif",PERIODIC_START,PERIODIC_END);

    //for (int y_i = 0; y_i < 5; ++y_i)
    //{
        //for (int x_i = 1; x_i < 10; x_i+=2)
        //{
            //std::cout << periodic_grid[y_i * (2*N) + x_i-1] << "," << periodic_grid[y_i * (2*N) + x_i] <<" ";
        //}
        //std::cout << std::endl;
    //}

    free(periodic_grid);
    free(velocity_grid);
    free(velocity_grid_next);
    free(divergence);
    free(pressure);
}
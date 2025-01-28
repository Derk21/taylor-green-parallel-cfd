#include "solve.h"

void solveDense(const double * A, const double *  B, double * X, size_t m){
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
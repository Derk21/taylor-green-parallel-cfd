#include "solve.cuh"

void solveDense(const double * A, const double *  B, double * X, size_t m){
    /*CAUTION: INPUT NEEDS TO BE COLUMN MAJOR */

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

    const int pivot_on = 0;

    //if (pivot_on) {
        //printf("pivot is on : compute P*A = L*U \n");
    //} else {
        //printf("pivot is off: compute A = L*U (not numerically stable)\n");
    //}

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

    //CUBLAS_OP_N transposes solution?
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

namespace gpu
{
void solveDense(double * d_A, double *  d_B, size_t m)
{
    /*
    Solves d_A X = d_B, result is stored in d_B
    */
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    const int lda = m;
    const int ldb = m;

    //std::vector<double> LU(lda * m, 0);
    std::vector<int> Ipiv(m, 0);
    int info = 0;

    //double *d_A = nullptr; /* device copy of A */
    //double *d_B = nullptr; /* device copy of B */
    int *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr; /* error info */

    int lwork = 0;            /* size of workspace */
    double *d_work = nullptr; /* device workspace for getrf */

    const int pivot_on = 0; //works well without in this case

    //if (pivot_on) {
        //printf("pivot is on : compute P*A = L*U \n");
    //} else {
        //printf("pivot is off: compute A = L*U (not numerically stable)\n");
    //}

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
    //CHECK_CUDA(cudaMalloc(&d_A, sizeof(double) * m * m));
    //CHECK_CUDA(cudaMalloc(&d_B, sizeof(double) * m));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int) * Ipiv.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    //CHECK_CUDA(
        //cudaMemcpyAsync(d_A, A, sizeof(double) * m * m, cudaMemcpyHostToDevice, stream));
    //CHECK_CUDA(
        //cudaMemcpyAsync(d_B, B, sizeof(double) * m, cudaMemcpyHostToDevice, stream));

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

    //CUBLAS_OP_N transposes solution?
    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */
                                        d_A, lda, d_Ipiv, d_B, ldb, d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */
                                        d_A, lda, NULL, d_B, ldb, d_info));
    }

    //CHECK_CUDA(
        //cudaMemcpyAsync(X, d_B, sizeof(double) * m, cudaMemcpyDeviceToHost, stream));
    //CHECK_CUDA(cudaStreamSynchronize(stream));

    //printf("X = (matlab base-1)\n");
    //print_matrix(m, 1, X, ldb);
    //printf("=====\n");

    /* free resources */
    //CHECK_CUDA(cudaFree(d_A));
    //CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_Ipiv));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUDA(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CHECK_CUDA(cudaStreamDestroy(stream));

    //CHECK_CUDA(cudaDeviceReset());

}

void solveSparse(double* dA_values,int *dA_columns, int *dA_csrOffsets, double* d_divergence, double*d_pressure,int A_nnz, const int m)
{
    //d_Y is just a placeholder
    //adapted from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spsv_csr/spsv_csr_example.c
    
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseSpSVDescr_t  spsvDescr;
    float     alpha           = 1.0f;

    CHECK_CUSPARSE( cusparseCreate(&handle) );
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, m, m, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, m, d_divergence, CUDA_R_64F) );
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, m, d_pressure, CUDA_R_64F) );
    //CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, m, d_X, CUDA_R_64F) );

    // Create opaque data structure, that holds analysis data between calls.
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescr) );

    cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE,
                                              &fillmode, sizeof(fillmode)) );

    cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;//laplace diag is not unit
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diagtype, sizeof(diagtype)) );
    // allocate an external buffer for analysis
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, vecY, CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
                                &bufferSize) );

    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) );
    CHECK_CUSPARSE( cusparseSpSV_analysis(
                                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, vecY, CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer) );
    // execute SpSV
    CHECK_CUSPARSE( cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, matA, vecX, vecY, CUDA_R_64F,
                                       CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr) );



    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) );
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) );
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) );
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsvDescr));
    CHECK_CUSPARSE( cusparseDestroy(handle) );
    CHECK_CUDA( cudaFree(dBuffer) );

}

}
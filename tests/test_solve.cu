#include "solve.cuh"
#include "utils.cuh"
#include <cassert>

void testSolveDense(){
    size_t m = 3;
    std::vector<double> A_ = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
    double * A = (double *)malloc(m*m * sizeof(double));
    double * B = (double *)malloc(m * sizeof(double));
    double * X = (double *)malloc(m * sizeof(double));
    for (size_t i = 0; i < m; i++)
    {
        X[i] = 0.0;
        B[i] = i + 1;
    }
    std::copy(A_.begin(), A_.end(), A);


    //CPU check
    solveDense(A,B,X,m);
    print_matrix(m, 1, X, m);
    /*
     * step 5: solve A*X = B
     *       | 1 |       | -0.3333 |
     *   B = | 2 |,  X = |  0.6667 |
     *       | 3 |       |  0      |
     *
     */
    assert(is_close(X[0], -0.333333));
    assert(is_close(X[1], 0.6666666));
    assert(is_close(X[2], 0.0));
    
    //GPU check 

    double * d_A,*d_B;
    double *h_B =(double *)malloc(m * sizeof(double));
 ;
    CHECK_CUDA(cudaMalloc(&d_A, m*m*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, m*sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, A,m*m* sizeof(double) , cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B,m* sizeof(double) , cudaMemcpyHostToDevice));

    gpu::solveDense(d_A,d_B,m);
    CHECK_CUDA(cudaMemcpy(h_B, d_B, m * sizeof(double) , cudaMemcpyDeviceToHost));

    print_matrix(m,1,h_B,m);
    assert(is_close(h_B[0], -0.333333));
    assert(is_close(h_B[1], 0.6666666));
    assert(is_close(h_B[2], 0.0));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));

}

void testSolveSparse(){
    //adapted from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spsv_csr/spsv_csr_example.c
    const int A_num_rows      = 4;
    const int A_num_cols      = 4;
    const int A_nnz           = 9;
    int       hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    int       hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    double hA_values[]     = { 1.0, 2.0, 3.0, 4.0, 5.0,
                                  6.0f, 7.0f, 8.0, 9.0 };
    double hX[]            = { 1.0, 8.0, 23., 52.0 };
    double hY[]            = { 0.0, 0.0, 0.0, 0.0 };
    double hY_result[]     = { 1.0, 2.0, 3.0, 4.0 };
     // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    double *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(double))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         A_num_cols * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         A_num_rows * sizeof(double)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_cols * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, A_num_rows * sizeof(double),
                           cudaMemcpyHostToDevice) )

    gpu::solveSparse(dA_values,dA_columns,dA_csrOffsets,dX,dY,A_nnz,A_num_rows);

    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        if (!is_close(hY[i],hY_result[i])) { 
            break;
            correct=0;
        }
    }
    if (correct)
        printf("spsv_csr_example test PASSED\n");
    else
        printf("spsv_csr_example test FAILED: wrong result\n");
    
    CHECK_CUDA(cudaFree(dA_columns));
    CHECK_CUDA(cudaFree(dA_csrOffsets));
    CHECK_CUDA(cudaFree(dA_values));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
}

int main(){
    testSolveDense();
    testSolveSparse();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
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
}

int main(){
    testSolveDense();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
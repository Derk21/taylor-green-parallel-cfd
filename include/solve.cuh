#pragma once
#ifndef SOLVE_H
#define SOLVE_H

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <vector>
#include "constants.cuh"
#include "utils.cuh"

void solveDense(const double * A, const double *  B, double * X, size_t m=2*NUM_N);

namespace gpu{
    void solveDense(double * d_A, double *  d_B, size_t m);

    void solveSparse(double* dA_values,int *dA_columns, int *dA_csrOffsets,double* d_divergence, double*d_pressure,int A_nnz=5*NUM_N*NUM_N, const int m=2*NUM_N);
}
#endif // SOLVE_H
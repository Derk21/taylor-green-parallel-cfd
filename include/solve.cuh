#pragma once
#ifndef SOLVE_H
#define SOLVE_H

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>
#include "constants.cuh"
#include "utils.cuh"

void solveDense(const double * A, const double *  B, double * X, size_t m=2*NUM_N);

namespace gpu{
    void solveDense(double * d_A, double *  d_B, size_t m);
}
#endif // SOLVE_H
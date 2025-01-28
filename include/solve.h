#pragma once
#ifndef SOLVE_H
#define SOLVE_H

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>
#include "constants.h"
#include "utils.h"

void solveDense(const double * A, const double *  B, double * X, size_t m=2*NUM_N);

#endif // SOLVE_H
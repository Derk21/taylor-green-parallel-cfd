#pragma once
#ifndef INIT_H
#define INIT_H

#include "constants.cuh"
#include "utils.cuh"

void initializePeriodicGrid(double *periodic_grid, int n, int m,const double periodic_start=PERIODIC_START,const double periodic_end=PERIODIC_END);

void initializeVelocityGrid(double *velocity_grid, double *periodic_grid, int n, int m);

void initializePressure(double *pressure_grid, int n, int m);

void initializeGaussianBlob(double *velocity_grid, double *periodic_grid, int n, int m,
                         double sigma, double amplitude);

#endif // INIT_H
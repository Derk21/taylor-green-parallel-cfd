#pragma once
#ifndef INIT_H
#define INIT_H

#include "constants.h"
#include "utils.h"

void initializePeriodicGrid(double *periodic_grid, int n, int m);
void initializeVelocityGrid(double *velocity_grid, double *periodic_grid, int n, int m);
void initializePressure(double *pressure_grid, int n, int m);

#endif // INIT_H
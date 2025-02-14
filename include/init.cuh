#pragma once
#ifndef INIT_H
#define INIT_H

#include "constants.cuh"
#include "utils.cuh"

/**
 * @brief fills periodic grid with evenly spaced values 
 * 
 * @param periodic_grid 
 * @param n 
 * @param m 
 * @param periodic_start 
 * @param periodic_end 
 */
void initializePeriodicGrid(double *periodic_grid, int n, int m, const double periodic_start = PERIODIC_START, const double periodic_end = PERIODIC_END);
/**
 * @brief initilizes velocity grid with taylor green vortex velocities at timestep 0
 * 
 * @param velocity_grid 
 * @param periodic_grid 
 * @param n 
 * @param m 
 */
void initializeVelocityGrid(double *velocity_grid, double *periodic_grid, int n, int m);

/**
 * @brief initilizes pressure with zeros
 * 
 * @param pressure_grid 
 * @param n 
 * @param m 
 */
void initializePressure(double *pressure_grid, int n, int m);

/**
 * @brief arbitrary velocity with some local blob where unidirectional velocities fade out
 * 
 * for advection testing
 * 
 * @param velocity_grid 
 * @param periodic_grid 
 * @param n 
 * @param m 
 * @param sigma 
 * @param amplitude 
 */
void initializeGaussianBlob(double *velocity_grid, double *periodic_grid, int n, int m,
                            double sigma, double amplitude);

#endif // INIT_H
#pragma once
#ifndef CONSTANTS_H
#define CONSTANTS_H
// Performance Optimizing Parameters
    #define BATCH_SIZE 10 // number of simulation steps before copy to host
    //TILING etc
//Simulation
    #define NUM_N 64              // Grid size X
    #define M 64              // Grid size Y
    #define ITERATIONS 10    // Number of iterations
    #define PERIODIC_START 0.0
    #define PERIODIC_END 2 * M_PI
    #define DIFFUSIVITY 0.1
    #define TIMESTEP 0.5 
    //explicit diffusion
    #define SUBSTEPS_EXPLICIT 40 //40-80 are recommended


#endif // CONSTANTS_H

#pragma once
#ifndef CONSTANTS_H
#define CONSTANTS_H
// Performance Optimizing Parameters
    #define GPU true
    #define CPU false
    #define BUFFER_SIZE 10 // number of simulation steps before copy to host
    #define TILE_SIZE 16
    //TILING etc
//Simulation
    #define NUM_N 64              // Grid size X
    #define M 64              // Grid size Y
    #define ITERATIONS 60    // Number of iterations
    #define PERIODIC_START 0.0
    #define PERIODIC_END (2 * M_PI)
    #define DIFFUSIVITY 0.1
    #define TIMESTEP  0.5 //0.5 
    #define DX ((PERIODIC_END-PERIODIC_START) / (NUM_N - 1))
    //explicit diffusion
    #define SUBSTEPS_EXPLICIT 40 //40-80 are recommended
    //advection
    #define MACCORMACK_CORRECTION 1.0//0.8//1.0


#endif // CONSTANTS_H

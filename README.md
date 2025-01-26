# Tailor Green Vortex 
 University Project for GPU Computing class at TU Berlin
- Development of a parallelized fluid simulation (diffusion, advection, pressure correction)
- Currently working on cpu-implementation to verify before parallelizing

## Current state of simulation (without pressure correction):
![Simulation](progress_documentation/diffusion_advection_correct.gif)

## Analytical Solution:
![Analytical Solution](progress_documentation/ground_truth.gif)
# Requirements
- cuda
- gnuplot-iostream-dev

# References
https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex
https://github.com/tum-pbs/PhiFlow/
https://tum-pbs.github.io/PhiFlow/examples/grids/Taylor_Green.html

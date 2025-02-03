# Tailor Green Vortex 
 University Project for GPU Computing class at TU Berlin
- Development of a parallelized fluid simulation (diffusion, advection, pressure correction)
- Currently working on cpu-implementation to verify before parallelizing

## Current state of simulation on cpu:
(explicit diffusion, mac_cormack advection, pressure correction)

<table>
  <tr>
    <th>Simulation</th>
    <th>Analytical Solution</th>
  </tr>
  <tr>
    <td>
      <img src="progress_documentation/mac_cormack_advection.gif" alt="Simulation" width="100%"/>
    </td>
    <td>
      <img src="progress_documentation/ground_truth.gif" alt="Analytical Solution" width="100%"/>
    </td>
  </tr>
</table>


# Requirements
- cuda
- gnuplot-iostream-dev

## Features:
### GPU:
- in development

### CPU:
Diffusion:
- explicit

Advection:
- semi-lagrangian (very unstable for taylor-green)
- mac cormack 

Pressure Correction:
- partly with CUDA-solver


# References
https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex
https://github.com/tum-pbs/PhiFlow/
https://tum-pbs.github.io/PhiFlow/examples/grids/Taylor_Green.html

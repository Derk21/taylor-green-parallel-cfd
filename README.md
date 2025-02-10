# Tailor Green Vortex 
 University Project for GPU Computing class at TU Berlin
- Development of a parallelized fluid simulation (diffusion, advection, pressure correction)

## Current state of simulation:
(gpu: explicit diffusion + pressure correction, advection methods are currently wrong)

<table>
  <tr>
    <th>Simulation</th>
    <th>Analytical Solution</th>
  </tr>
  <tr>
    <td>
      <img src="progress_documentation/gpu_diffusion_pressure_correction.gif" alt="Simulation" width="100%"/>
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
Diffusion:
- explicit

Advection:
- semi-lagrangian (very unstable for taylor-green)
- mac cormack

Pressure Correction:
- Dense CUDA-solver  (could be done with cuSparse)


# References
https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex
https://github.com/tum-pbs/PhiFlow/
https://tum-pbs.github.io/PhiFlow/examples/grids/Taylor_Green.html

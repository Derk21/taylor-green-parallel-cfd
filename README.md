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
- might need to add your cuda architecture to [CMakeLists.txt](CMakeLists.txt)

# Running the simulation
```
make main && ./bin/main
```
run tests like this:
```
make test_diffusion && ./bin/diffusion
```

## Features:
Diffusion:
- explicit

Advection:
- semi-lagrangian (very unstable for taylor-green)
- mac cormack

Pressure Correction:
- with Dense CUDA-solver  
- with Sparse CUDA-solver (~5 times faster than dense)
# Practices used from the course
- shared memory (used in divergence, interpolation and diffusion)
- coalescing (only in parts (divergence and interpolation use it partly in shared memory))
- biggest speedup came from implementation of sparse pressure correction (~5x faster)
- optimization of launch configurations 

# Issues
- advection looks wrong and gpu-cpu implementation of mac-cormack have different results
- lacking coalsescing in global memory (velocity is stored in alternating u,v,u,v ... format instead of in double2)




# References
https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex
https://github.com/tum-pbs/PhiFlow/
https://tum-pbs.github.io/PhiFlow/examples/grids/Taylor_Green.html

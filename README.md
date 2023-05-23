[![DOI](https://zenodo.org/badge/643925360.svg)](https://zenodo.org/badge/latestdoi/643925360)


This repository contains the python scripts to reproduce the computational examples presented in "A discontinuous Galerkin approach for atmospheric flows with implicit condensation" by S. Hittmeir, P. L. Lederer. J. Schöberl and H. v. Wahl.

# Files
```
|- README.md                                                    // This file
|- LICENSE                                                      // The licence file
|- install.txt                                                  // Installation help
|- Example 4.1
|  |- run.bash
|  |- cloudy_air_gravity_waves.py
|  |- postprocess_numerical_convergence.py
|  |- results_cloudy_gravity_waves_dg_ssprk43_artdiff0quads1struc1_order1_pert.txt
|  |- results_cloudy_gravity_waves_dg_ssprk43_artdiff0quads1struc1_order2_pert.txt
|  |- results_cloudy_gravity_waves_dg_ssprk43_artdiff0quads1struc1_order3_pert.txt
|  |- results_cloudy_gravity_waves_dg_ssprk43_artdiff0quads1struc1_order4_pert.txt
|  |- results_cloudy_gravity_waves_dg_ssprk43_artdiff0quads1struc1_raw.dat
|- Example 4.2
|  |- run.bash
|  |- cloudy_air_bryan_fritsch.py
|- Example 4.3
|  |- run.bash
|  |- cloudy_air_gravity_waves_no_init_clouds.py
|- Example 4.4
|  |- run.bash
|  |- rainy_air_hydrostatic_mountain.py
|- Example 4.5
|  |- run.bash
|  |- rainy_air_unsaturated_rising_bubble.py
|- Example 4.6
|  |- run.bash
|  |- rainy_air_unsaturated_rising_bubble_3d.py
```

# Installation

Detailed instructions and the specific version of NGSolve used are given in `install.txt`.

# How to reproduce
The specific examples presented in our work are implemented in the python files of the corresponding folders. The `run.bash` file in each subdirectory contains the command-line options used to compute the presented results. 

**IMPORTANT:** The computations require large computational resources, depending on the discretisation parameters used. It may also be beneficial to change some of the parameters inside the python files. In particular it may be worth changing 
- `compile_flag=True` to produce and compile c++ code of the forms for faster evaluation
- `vtk_flag=True` to export VTK files to inspect the results visually after the simulation. 
- `SetNumThreads(X)` to set the number of shared-memory parallel threads to `X`.

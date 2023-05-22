#!/bin/bash

ngspy cloudy_air_gravity_waves.py -hmax 1000 -qm 1 -sm 1 -o 1 -dt 1.0    -ad 0
ngspy cloudy_air_gravity_waves.py -hmax  500 -qm 1 -sm 1 -o 1 -dt 0.5    -ad 0
ngspy cloudy_air_gravity_waves.py -hmax  250 -qm 1 -sm 1 -o 1 -dt 0.25   -ad 0
ngspy cloudy_air_gravity_waves.py -hmax  125 -qm 1 -sm 1 -o 1 -dt 0.125  -ad 0

ngspy cloudy_air_gravity_waves.py -hmax 1000 -qm 1 -sm 1 -o 2 -dt 0.5    -ad 0
ngspy cloudy_air_gravity_waves.py -hmax  500 -qm 1 -sm 1 -o 2 -dt 0.25   -ad 0
ngspy cloudy_air_gravity_waves.py -hmax  250 -qm 1 -sm 1 -o 2 -dt 0.125  -ad 0
ngspy cloudy_air_gravity_waves.py -hmax  125 -qm 1 -sm 1 -o 2 -dt 0.0625 -ad 0

ngspy cloudy_air_gravity_waves.py -hmax 1000 -qm 1 -sm 1 -o 3 -dt 0.3    -ad 0
ngspy cloudy_air_gravity_waves.py -hmax  500 -qm 1 -sm 1 -o 3 -dt 0.15   -ad 0
ngspy cloudy_air_gravity_waves.py -hmax  250 -qm 1 -sm 1 -o 3 -dt 0.075  -ad 0
ngspy cloudy_air_gravity_waves.py -hmax  125 -qm 1 -sm 1 -o 3 -dt 0.0375 -ad 0

ngspy cloudy_air_gravity_waves.py -hmax 1000 -qm 1 -sm 1 -o 4 -dt 0.2    -ad 0
ngspy cloudy_air_gravity_waves.py -hmax  500 -qm 1 -sm 1 -o 4 -dt 0.1    -ad 0
ngspy cloudy_air_gravity_waves.py -hmax  250 -qm 1 -sm 1 -o 4 -dt 0.05   -ad 0
ngspy cloudy_air_gravity_waves.py -hmax  125 -qm 1 -sm 1 -o 4 -dt 0.025  -ad 0

ngspy postprocess_numerical_convergence.py

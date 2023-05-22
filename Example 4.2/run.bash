#!/bin/bash

mpirun -np 24 ngspy cloudy_air_bryan_fritsch.py -hmax 100 -qm 1 -sm 1 -o 1 -dt 0.08   -ad 0
mpirun -np 24 ngspy cloudy_air_bryan_fritsch.py -hmax 100 -qm 0 -sm 0 -o 1 -dt 0.08   -ad 0
mpirun -np 24 ngspy cloudy_air_bryan_fritsch.py -hmax 100 -qm 0 -sm 0 -o 2 -dt 0.04   -ad 0
mpirun -np 24 ngspy cloudy_air_bryan_fritsch.py -hmax 100 -qm 0 -sm 0 -o 3 -dt 0.025  -ad 0
mpirun -np 24 ngspy cloudy_air_bryan_fritsch.py -hmax  50 -qm 1 -sm 1 -o 1 -dt 0.04   -ad 0
mpirun -np 24 ngspy cloudy_air_bryan_fritsch.py -hmax  50 -qm 0 -sm 0 -o 1 -dt 0.04   -ad 0
mpirun -np 24 ngspy cloudy_air_bryan_fritsch.py -hmax  50 -qm 0 -sm 0 -o 2 -dt 0.02   -ad 0
mpirun -np 24 ngspy cloudy_air_bryan_fritsch.py -hmax  50 -qm 0 -sm 0 -o 3 -dt 0.0125 -ad 0

#!/bin/bash
#SBATCH -C cpu
#SBATCH -A m1657
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH -J calculate_plume_mean_properties.py
#SBATCH -e calculate_plume_mean_properties.err
#SBATCH -o calculate_plume_mean_properties.out

date
srun -n 1 --cpu-bind=cores python -u calculate_plume_mean_properties.py
date

#!/bin/bash
#SBATCH -C cpu
#SBATCH -A m1657
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH -J calc_tracked_clouds_stats_alt_2.py
#SBATCH -e calc_tracked_clouds_stats_alt_2.err
#SBATCH -o calc_tracked_clouds_stats_alt_2.out

date
srun -n 1 --cpu-bind=cores python -u calc_tracked_clouds_stats_alt_2.py
date

#!/bin/bash
#SBATCH -C cpu
#SBATCH -A m1657
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH -J calc_tracked_clouds_stats_alt_sparse
#SBATCH -e calc_tracked_clouds_stats_alt_sparse.err
#SBATCH -o calc_tracked_clouds_stats_alt_sparse.out

set -euo pipefail

date

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate h5pyenv
fi

# Prevent threaded math libraries from oversubscribing CPUs inside worker processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun -n 1 --cpu-bind=cores python -u calc_tracked_clouds_stats_alt_sparse.py

date

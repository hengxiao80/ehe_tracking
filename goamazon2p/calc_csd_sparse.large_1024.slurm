#!/bin/bash
#SBATCH -C cpu
#SBATCH -A m1657
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J calc_csd_sparse_1024
#SBATCH -e calc_csd_sparse.large_1024.err
#SBATCH -o calc_csd_sparse.large_1024.out

cd /global/cfs/cdirs/m1657/xiao169/ehe_tracking/goamazon2p

date
python -u calc_csd_sparse.large_1024.py
date

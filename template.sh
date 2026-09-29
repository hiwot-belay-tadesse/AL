#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-010:00 # Runtime in D-HH:MM
#SBATCH -p serial_requeue # odyssey partition
#SBATCH --mem=30GB # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ADARP/results/logs/out_%j.txt # File to which STDOUT will be written
#SBATCH -e ADARP/results/logs/err_%j.txt # File to which STDERR will be written

export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=false

mkdir -p ADARP/results/logs

# python -u run.py EXPDIR EXPNAME KWARGS
python -u refactor_run.py EXPDIR EXPNAME KWARGS

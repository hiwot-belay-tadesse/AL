#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-010:00 # Runtime in D-HH:MM
#SBATCH -p serial_requeue # odyssey partition
#SBATCH --mem=30GB # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../../../../holylabs/LABS/doshi-velez_labs/Users/htadesse/out_%j.txt 
#SBATCH -e ../../../../holylabs/LABS/doshi-velez_labs/Users/htadesse/err_%j.txt
#SBATCH -o output/ID21/mixed/Nectarine_Crave/typiclust/UF80_K40_B7_T30_DR30/out_%j.txt # File to which STDOUT will be written
#SBATCH -e output/ID21/mixed/Nectarine_Crave/typiclust/UF80_K40_B7_T30_DR30/err_%j.txt # File to which STDERR will be written

python -u run.py output/ID21/mixed/Nectarine_Crave/typiclust/UF80_K40_B7_T30_DR30 typiclust '{"user": "ID21", "pool": "mixed", "fruit": "Nectarine", "scenario": "Crave", "T": 30, "K": 40, "Budget": 7, "unlabeled_frac": 0.8, "dropout_rate": 0.3}'

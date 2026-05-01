#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-010:00 # Runtime in D-HH:MM
#SBATCH -p serial_requeue # odyssey partition
#SBATCH --mem=30GB # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../../../../holylabs/LABS/doshi-velez_labs/Users/htadesse/out_%j.txt 
#SBATCH -e ../../../../holylabs/LABS/doshi-velez_labs/Users/htadesse/err_%j.txt
#SBATCH -o Cardiomate_AL/G_SSL_no_augment_nn/global/15/BP_spike/coreset/out_%j.txt # File to which STDOUT will be written
#SBATCH -e Cardiomate_AL/G_SSL_no_augment_nn/global/15/BP_spike/coreset/err_%j.txt # File to which STDERR will be written

# python -u run.py Cardiomate_AL/G_SSL_no_augment_nn/global/15/BP_spike/coreset coreset '{"user": "15", "pool": "global", "fruit": "BP", "scenario": "spike", "task": "bp", "participant_id": "15", "K": 100, "Budget": null, "unlabeled_frac": 0.22, "dropout_rate": 0.5, "warm_start": false, "input_df": "raw"}'
python -u refactor_run.py Cardiomate_AL/G_SSL_no_augment_nn/global/15/BP_spike/coreset coreset '{"user": "15", "pool": "global", "fruit": "BP", "scenario": "spike", "task": "bp", "participant_id": "15", "K": 100, "Budget": null, "unlabeled_frac": 0.22, "dropout_rate": 0.5, "warm_start": false, "input_df": "raw"}'

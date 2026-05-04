#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-010:00 # Runtime in D-HH:MM
#SBATCH -p serial_requeue # odyssey partition
#SBATCH --mem=30GB # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../../../../holylabs/LABS/doshi-velez_labs/Users/htadesse/out_%j.txt 
#SBATCH -e ../../../../holylabs/LABS/doshi-velez_labs/Users/htadesse/err_%j.txt
#SBATCH -o multiseeds/global/15/BP_spike/random/seed_41/out_%j.txt # File to which STDOUT will be written
#SBATCH -e multiseeds/global/15/BP_spike/random/seed_41/err_%j.txt # File to which STDERR will be written

cd "/Users/hiwotbelaytadesse/Harvard University Dropbox/Hiwot Belay Tadesse/Banaware_AL-2"
export BAN_AL_OUTPUT_DIR="multiseeds"
python -u run.py "multiseeds/global/15/BP_spike/random/seed_41" "random" '{"user": "15", "pool": "global", "fruit": "BP", "scenario": "spike", "unlabeled_frac": 0.22, "dropout_rate": 0.5, "warm_start": 0, "seed": 41, "aq": "random", "K": 100, "Budget": null, "T": 50, "input_df": "raw", "task": "bp", "participant_id": "15", "output_dir": "multiseeds"}'

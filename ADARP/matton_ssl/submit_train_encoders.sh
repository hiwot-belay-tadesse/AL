#!/bin/bash
# Train ADARP SimCLR encoders with Matton et al. EDA augmentations on raw 4 Hz EDA,
# then (optionally) score them with the one-class SVM. Submit from the repo root:
#
#     sbatch ADARP/matton_ssl/submit_train_encoders.sh                   # global encoders, 1 job
#     POOL=personal sbatch ADARP/matton_ssl/submit_train_encoders.sh     # all personal encoders, 1 job
#     POOL=personal sbatch --array=0-10 ADARP/matton_ssl/submit_train_encoders.sh
#                                   # one job per participant (11 jobs); scoring is skipped
#                                   # inside array tasks -- run it once afterwards with
#     python ADARP/matton_ssl/score_one_class_svm.py --channels eda
#     SEED=43 CHANNELS="eda" RUN_OCSVM=0 sbatch ADARP/matton_ssl/submit_train_encoders.sh
#
# Knobs (all optional):
#     SEED        split seed (default 42)          POOL      global | personal (default global)
#     CHANNELS    "eda hr" (default) or "eda"      OUT_DIR   default ADARP/matton_ssl/encoders
#     ADARP_BATCH_SSL / ADARP_SSL_EPOCHS            default 32 / 100, as the pipeline
#     RUN_OCSVM   1 (default) also runs ADARP/matton_ssl/score_one_class_svm.py on the result
#     EXTRA       extra args for train_encoders.py, e.g. "--augmentations low_pass band_pass"
#     PARTICIPANTS  space-separated ids indexed by SLURM_ARRAY_TASK_ID in --array mode
#                 (default: the 11 ADARP participants, index 0 = 101 ... 10 = 112)
#     SENSOR_DIR  override for the raw E4 'Part <id>C' folder. Not needed when the data
#                 is at ~/AL/DATA/ADARP/Sensor Data or <repo>/DATA/ADARP/Sensor Data,
#                 which data4hz.resolve_sensor_dir finds on its own.

#SBATCH --job-name=matton_ssl_adarp
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-12:00
#SBATCH -p serial_requeue
#SBATCH --mem=48GB
#SBATCH -o ADARP/results/logs/matton_ssl_out_%A_%a.txt
#SBATCH -e ADARP/results/logs/matton_ssl_err_%A_%a.txt

set -euo pipefail

export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=3
export KERAS_BACKEND=tensorflow
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export ADARP_BATCH_SSL="${ADARP_BATCH_SSL:-32}"
export ADARP_SSL_EPOCHS="${ADARP_SSL_EPOCHS:-100}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p ADARP/results/logs

# raw 4 Hz EDA location: explicit override only; otherwise data4hz searches
# ~/AL/DATA/ADARP/Sensor Data, then <repo>/AL/DATA/..., then <repo>/DATA/...
if [ -n "${SENSOR_DIR:-}" ]; then
  export ADARP_SENSOR_DIR="${SENSOR_DIR}"
fi

SEED="${SEED:-42}"
POOL="${POOL:-global}"
CHANNELS="${CHANNELS:-eda hr}"
OUT_DIR="${OUT_DIR:-ADARP/matton_ssl/encoders}"
RUN_OCSVM="${RUN_OCSVM:-1}"
EXTRA="${EXTRA:-}"
PARTICIPANTS="${PARTICIPANTS:-101 102 104 105 106 107 108 109 110 111 112}"

# --array mode: this task trains ONE participant's personal encoders and does not score,
# since scoring needs every participant's encoder to exist first.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  read -r -a PIDS <<< "${PARTICIPANTS}"
  if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#PIDS[@]}" ]; then
    echo "[matton-ssl] array index ${SLURM_ARRAY_TASK_ID} beyond ${#PIDS[@]} participants; nothing to do"
    exit 0
  fi
  PID="${PIDS[${SLURM_ARRAY_TASK_ID}]}"
  POOL="personal"
  RUN_OCSVM="0"
  EXTRA="${EXTRA} --participants ${PID}"
  echo "[matton-ssl] array task ${SLURM_ARRAY_TASK_ID}: participant ${PID}"
fi

echo "[matton-ssl] host=$(hostname) seed=${SEED} pool=${POOL} channels=[${CHANNELS}] out=${OUT_DIR}"
echo "[matton-ssl] batch=${ADARP_BATCH_SSL} epochs=${ADARP_SSL_EPOCHS} extra=[${EXTRA}]"

# shellcheck disable=SC2086
python -u ADARP/matton_ssl/train_encoders.py \
  --seed "${SEED}" --pool "${POOL}" --channels ${CHANNELS} --out_dir "${OUT_DIR}" ${EXTRA}

if [ "${RUN_OCSVM}" = "1" ]; then
  echo "[matton-ssl] === one-class SVM on Matton 4 Hz encoders / eda ==="
  python -u ADARP/matton_ssl/score_one_class_svm.py --channels eda \
    --run_dir "${OUT_DIR}" --seed "${SEED}" --results_dir ADARP/matton_ssl/results
  if [[ " ${CHANNELS} " == *" hr "* ]]; then
    echo "[matton-ssl] === one-class SVM on Matton 4 Hz encoders / hr_eda ==="
    python -u ADARP/matton_ssl/score_one_class_svm.py --channels hr_eda \
      --run_dir "${OUT_DIR}" --seed "${SEED}" --results_dir ADARP/matton_ssl/results
  fi
fi

echo "[matton-ssl] done -> ${OUT_DIR}/seed_${SEED}"

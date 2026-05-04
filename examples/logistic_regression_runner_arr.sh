#!/usr/bin/env bash
#SBATCH --job-name=eval_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-36
#SBATCH --open-mode=append

set -euo pipefail

mkdir -p logs

module load stack/2024-06 gcc/12.2.0 python/3.12.8 cuda/12.8.0 eth_proxy

export ROOT_DIR_BRAINTREEBANK="$SCRATCH/neuroprobe/wang_barbu_braintreebank_2023"

subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
trials=(1 2 0 4 0 1 0 1 0 1 0 1)

eval_names=(
    frame_brightness
    global_flow
    local_flow
    face_num
    volume
    pitch
    delta_volume
    speech
    onset
    gpt2_surprisal
    word_length
    word_gap
    word_index
    word_head_pos
    word_part_speech
)

# Run all eval tasks sequentially inside each job.
EVAL_NAME="$(IFS=,; echo "${eval_names[*]}")"

splits_type=(
    within_session
    cross_session
    cross_subject
)

num_pairs="${#subjects[@]}"
num_splits="${#splits_type[@]}"
expected_tasks=$((num_pairs * num_splits))

if [[ "${#subjects[@]}" -ne "${#trials[@]}" ]]; then
    echo "ERROR: subjects and trials arrays have different lengths." >&2
    exit 1
fi

if [[ "${SLURM_ARRAY_TASK_ID}" -lt 1 || "${SLURM_ARRAY_TASK_ID}" -gt "${expected_tasks}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, but expected 1-${expected_tasks}." >&2
    exit 1
fi

task_idx=$((SLURM_ARRAY_TASK_ID - 1))

PAIR_IDX=$((task_idx % num_pairs))
SPLITS_TYPE_IDX=$((task_idx / num_pairs))

SUBJECT="${subjects[$PAIR_IDX]}"
TRIAL="${trials[$PAIR_IDX]}"
SPLITS_TYPE="${splits_type[$SPLITS_TYPE_IDX]}"

echo "Started at $(date)"
echo "Task ${SLURM_ARRAY_TASK_ID}/${expected_tasks}"
echo "Subject=${SUBJECT}, Trial=${TRIAL}, Split=${SPLITS_TYPE}"
echo "Eval tasks=${EVAL_NAME}"

uv run examples/logistic_regression_runner.py \
    tasks="${EVAL_NAME}" \
    subject_trials="[${SUBJECT}, ${TRIAL}]" \
    eval_splits="${SPLITS_TYPE}"

echo "Evaluation completed at $(date)."

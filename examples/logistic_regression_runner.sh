#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

module load stack/2024-06 gcc/12.2.0 python/3.12.8 cuda/12.8.0 eth_proxy

export ROOT_DIR_BRAINTREEBANK="$SCRATCH/neuroprobe/wang_barbu_braintreebank_2023"

echo "Running Logistic Regression evaluation at $(date)..."

uv run examples/logistic_regression_runner.py

echo "Evaluation completed at $(date)."
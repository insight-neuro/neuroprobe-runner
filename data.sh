#!/usr/bin/env bash
# Script to download the Braintreebank dataset
# Run as: ./data.sh
# The following flags can be passed to the brainsets prepare command:
# --lite: Only download files necessary for the Neuroprobe benchmark, reducing the number of downloaded files by >50%.
# --nano: Even smaller download for the Nano version of the benchmark.
# --skip_initial_download: Use this flag if you have already downloaded the raw data and just want to re-run the processing steps. 
# --overwrite: Use this flag to re-download and re-process the data, overwriting any existing files.

# Options for SLURM job scheduler (adjust as needed)
#SBATCH --job-name=data
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

# Load environment variables from .env if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
  echo "Loaded environment variables from .env file."
fi

RAW_DIR="$ROOT_DIR_BRAINTREEBANK/raw"

# Use insight-neuro's fork of brainsets to access the datasets
BRAINSETS=git+https://github.com/insight-neuro/brainsets
DATASET=wang_barbu_braintreebank_2023

echo "Downloading Braintreebank dataset at $(date)..."

uvx --from "$BRAINSETS" brainsets prepare "$DATASET" \
    --raw-dir "$RAW_DIR" --processed-dir "$ROOT_DIR_BRAINTREEBANK" \
    "$@"

# Delete raw data to save storage space
rm -rf "$RAW_DIR"
    
echo "Dataset preparation completed at $(date)."
echo "Data available at $ROOT_DIR_BRAINTREEBANK/wang_barbu_braintreebank_2023."
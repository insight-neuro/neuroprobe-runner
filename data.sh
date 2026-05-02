#!/usr/bin/env bash
# Script to download the Braintreebank dataset
# Run as: ./data.sh
# The following flags can be passed to the brainsets prepare command:
# --lite: Only download files necessary for the Neuroprobe benchmark, reducing the number of downloaded files by >50%.
# --nano: Even smaller download for the Nano version of the benchmark.
# --skip_initial_download: Use this flag if you have already downloaded the raw data and just want to re-run the processing steps. 
# --overwrite: Use this flag to re-download and re-process the data, overwriting any existing files.


OUT_DIR="data/braintreebank"
RAW_DIR="$OUT_DIR/raw"

# Use insight-neuro's fork of brainsets to access the datasets
BRAINSETS=git+https://github.com/insight-neuro/brainsets
DATASET=wang_barbu_braintreebank_2023

echo "Downloading Braintreebank dataset at $(date)..."

uvx --from "$BRAINSETS" brainsets prepare "$DATASET" \
    --raw-dir "$RAW_DIR" --processed-dir "$OUT_DIR" \
    "$@"

# Delete raw data to save storage space
rm -rf "$RAW_DIR"
    
echo "Dataset preparation completed at $(date)."
echo "Data available at $OUT_DIR."

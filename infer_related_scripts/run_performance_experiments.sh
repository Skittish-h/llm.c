#!/bin/bash

# Default precision (choose from FP16, FP32, BF16)
PRECISION="FP16"

# Parse command-line argument for precision
if [ "$1" ]; then
    PRECISION="$1"
    echo "Using precision: $PRECISION"
else
    echo "No precision specified. Using default: $PRECISION"
fi

# Token sizes to iterate over
TOKEN_SIZES=(64 128 256 512 1024)

# Define output directory path
OUTPUT_DIR="timings"
PROMPTSET_DIR="promptset"

# Create output and prompt directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROMPTSET_DIR"

# Build required files for the specified precision type using the Makefile in the parent directory
echo "Building files for precision: $PRECISION"

make -C ../ clean
make -C ../ infer_gpt2_timing_hostcu PRECISION="$PRECISION" || { echo "Build failed for infer_gpt2_timing_hostcu"; exit 1; }
make -C ../ infer_gpt2_timing_devicecu PRECISION="$PRECISION" || { echo "Build failed for infer_gpt2_timing_devicecu"; exit 1; }

# Generate datasets for each token size
for T in "${TOKEN_SIZES[@]}"; do
    python promptset.py -t "$T" || { echo "Failed to generate dataset for token size $T"; exit 1; }
    echo "Generated dataset for token size $T"
done

# Change working directory to the parent directory
cd .. || { echo "Failed to change directory to parent"; exit 1; }

# Loop over each token size and run the commands
for T in "${TOKEN_SIZES[@]}"; do
    echo "Running inference for token size: $T"

    # Construct input file path and output filenames
    INPUT_FILE="infer_related_scripts/promptset/prompt_${T}.bin"
    OUTPUT_DEVICE="timings/timings_device_${PRECISION}_${T}.json"
    OUTPUT_HOST="timings/timings_host_${PRECISION}_${T}.json"

    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Input file $INPUT_FILE does not exist. Skipping token size $T."
        continue
    fi

    # Run the first command
    ./infer_gpt2_timing_hostcu --out "$OUTPUT_DEVICE" --t "$T" --in "$INPUT_FILE" || {
        echo "Failed to run infer_gpt2_timing_hostcu for token size $T";
        continue;
    }
    echo "Generated $OUTPUT_DEVICE"

    # Run the second command
    ./infer_gpt2_timing_devicecu --out "$OUTPUT_HOST" --t "$T" --in "$INPUT_FILE" || {
        echo "Failed to run infer_gpt2_timing_devicecu for token size $T";
        continue;
    }
    echo "Generated $OUTPUT_HOST"
done

echo "All inferences completed."
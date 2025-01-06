#!/bin/bash

# Token sizes to iterate over
TOKEN_SIZES=(64 128 256 512 1024)
# TOKEN_SIZES=(1024)


# Loop over each token size and run the commands
for T in "${TOKEN_SIZES[@]}"; do
    echo "Running inference for token size: $T"

    # Construct input file path and output filenames
    INPUT_FILE="dev/data/promptset/prompt_${T}.bin"
    OUTPUT_KERNEL="timings_gpu32_${T}.json"
    OUTPUT_DATALOADER="timings32_${T}.json"

    # Run the first command
    ./infer_gpt2_kerneljancu --out "$OUTPUT_KERNEL" --t "$T" --in "$INPUT_FILE"
    echo "Generated $OUTPUT_KERNEL"

    # Run the second command
    ./infer_gpt2_antoncu --out "$OUTPUT_DATALOADER" --t "$T" --in "$INPUT_FILE"
    echo "Generated $OUTPUT_DATALOADER"
done

echo "All inferences completed."
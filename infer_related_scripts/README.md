# Aplied GPU Programming - Inference Related Scripts

## Overview

This sections gives a short overview of newly created and edited files in this `LLM.c` fork.

### Created files

[`infer_gpt2.cu`](../infer_gpt2.cu) Inference script for handling user input tokens\
[`infer_gpt2_timing_host.cu`](../infer_gpt2_timing_host.cu) Inference script with timings and dataloader.\
[`infer_gpt2_timing_device.cu`](../infer_gpt2_timing_device.cu) Inference script with timings and dataloader with token sampling on device.\
[`infer_gpt2_zvono_accuracy.cu`](../infer_gpt2_zvono_accuracy.cu) Inference script for collecting logits for perplexity analysis.\

[`llmc/promptloader.h`](../llmc/promptloader.h) Dataloader for tokenized prompts\
[`infer_related_scripts/`](../infer_related_scripts/) Directory with python evaluation scripts, dataset generation and dataset files


### Edited files

[`Makefile`](../Makefile) Modified to support FP16 and compile infernece CUDA scripts\
[`train_gpt2.cu`](../train_gpt2.cu) Added mode inference mode which disables gradient and optimizer allocations.\
[`llmc/cuda_common.h`](../llmc/cuda_common.h) Online type conversion from FP32 to FP16 in `inline void file_to_devic` and added new precision mode.\
[`llmc/cuda_common.h`](../llmc/cuda_common.h) Added utility functions for copying, casting and handling FP16 precision type.\
[`llmc/sampler.h`](../llmc/cuda_common.h) Added a sampling fucntion that support top k and nucleus sampling and function for computing log probabilities.\

## Setup

1. Download model weights by executing the statrter pack bash script\
    `./dev/download_starter_pack.sh`
2. Comnpile the project\
    `make`\
    or optionally specify precision\
    `make PRESICION=BF16`
3. Install python requirements for tokenizen the prompt dataset and evaluating metrics\
    `pip install -r requirements.txt`
4. Done!

All inference scripts default to the current state of the repo - FP16 accuracy.

## Run inference

To run inference execute `./infer_gpt2cu` with default settings and default input tokens.
`

## Perplexity Scripts

A notebook is provided that will run inference on the PTB dataset is located at `run_perplexity.ipynb`, it is meant to be opened in colab
i.e. `https://colab.research.google.com/github/Skittish-h/llm.c/blob/master/infer_related_scripts/run_perplexity.ipynb`

## Model Divergence Scripts

To collect logit data for divergence analysis, follow the regular run script, but compile and run `infer_gpt2_zvono_accuracy.cu` instead of `infer_gpt2.cu`.
First make sure the result of the model has a directory to save the results to with `mkdir saved_logits`.
Then run the script with the same flags as `infer_gpt2.cu` and a name flag which will identify the results of the run.
```bash
make infer_gpt2_zvono_accuracy
./infer_gpt2_zvono_accuracy --name fp16 # ... same flags as infer_gpt2.cu 
```

Then to run analysis, modify the variables of compare.py to match the name flags used when collecting logit data.
```python
# compare.py
inputs_dir = "saved_logits" # or whatever directory you saved the logits to
output_dir = "{whateveryouwant}"
names = ["FP16", "BF16", "FP32"] # matching the names when collecting
reference_name = "BF16" # will be used as the reference for the divergence analysis
```

Then run the script with `python compare.py` to generate the divergence analysis, all your graphs will be saved in the output directory.
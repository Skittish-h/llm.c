# Applied GPU Programming - Inference Related Scripts

## Overview

This sections gives a short overview of newly created and edited files in this `LLM.c` fork.

### Created files

[`infer_gpt2.cu`](../infer_gpt2.cu) Inference script for handling user input tokens\
[`infer_gpt2_timing_host.cu`](../infer_gpt2_timing_host.cu) Inference script with timings and dataloader with token sampling on host.\
[`infer_gpt2_timing_device.cu`](../infer_gpt2_timing_device.cu) Inference script with timings and dataloader with token sampling on device with argmax kernel.\
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

To run inference, execute the command: \
`./infer_gpt2cu`
- **`--tokens`**: List of integer tokens for input. Default: `{12295, 8066, 1577, 345, 510, 11, 1239, 8066}`.
By default, the script uses the following input tokens and settings:

- **`--tokens`**: Specifies the list of integer tokens for input. Default: `{12295, 8066, 1577, 345, 510, 11, 1239, 8066}`.
- **`--n_gen`**: Total number of tokens to generate, including the input tokens. Default: `64`.
- **`--top_k`**: Top-K sampling parameter to control randomness in token selection. Default: `10`.
- **`--temp`**: Sampling temperature, affecting diversity in token generation. Default: `1.0`.
- **`--top_p`**: Top-P (nucleus) sampling parameter for probability thresholding. Default: `0.8`.
- **`--seed`**: Random seed for reproducibility of results. Default: `42` (`-1` to disable seed setting).
- **`--t`**: Token length for input allocation. Default: `1024`.

You can override these defaults by specifying arguments during execution. For example:
```bash
./infer_gpt2cu –tokens 12295, 8066, 1577 –n_gen 128 –top_k 20 –temp 0.7 –top_p 0.9 –seed 123
```
You can also tokenize a prompt and pass the tokens to the inference script in a single command:
```bash
./infer_gpt2cu --tokens $(python infer_related_scripts/tokenize.py "I want to go to bed")
```

## Run timing experiments

To run our timing experiments, execute the following bash script. You can specify the floating point precision through an argument in the script:
```bash
cd infer_related_scripts/
sh run_performance_experiments.sh FP16
sh run_performance_experiments.sh FP32
sh run_performance_experiments.sh BF16
```

Attentione! You can only run the script from within the directory [`infer_related_scripts/`](../infer_related_scripts/).

The timings will be written as JSON files to `infer_related_scripts/timimgs/`.

To execute the dedicated `timing_host` and `timimg_device` cuda scripts, refer to the content of the bash script.

## Perplexity Scripts

A notebook is provided that will run inference on the PTB dataset is located at `run_perplexity.ipynb`, it is meant to be opened in colab
i.e. https://colab.research.google.com/github/Skittish-h/llm.c/blob/master/infer_related_scripts/run_perplexity.ipynb

If you want to run the perplexity experiments in your shell:
```bash
python infer_related_scripts/promptset.py -t 64 -i ../ptb/ptb.test.txt
make perplexity_gpt2cu
./perplexity_gpt2cu
```

## Model Divergence Scripts

To collect logit data for divergence analysis, run `run_divergence.ipynb` in Google colab
i.e. https://colab.research.google.com/github/Skittish-h/llm.c/blob/master/infer_related_scripts/run_divergence.ipynb

If you want to run the perplexity experiments in your shell:
```bash
python infer_related_scripts/promptset.py -t 64
mkdir -p saved_logits
make infer_gpt2_divergencecu
./infer_gpt2_divergencecu --name fp16 --t 64 --in "infer_related_scripts/promptset/prompt_64.bin"
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
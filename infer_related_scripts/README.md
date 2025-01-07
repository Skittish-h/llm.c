# Aplied GPU Programming - Inference Related Scripts

All inference notebooks default to the current state of the repo - FP16 accuracy.
To modify the FP accuracy, modify the PRECISION within make to `FP32` and or `BF16` and recompile the repo.
BF 16 additionally requires modifying the given inference script to load the bf16 model, i.e. 
```c
const char* load_filename = "gpt2_124M.bin";
```
Should be changed to:
```c
const char* load_filename = "gpt2_124M_bf16.bin";
```


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
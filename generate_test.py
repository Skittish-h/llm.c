from scipy.special import softmax
import os
import json
import numpy as np

inputs_dir = "inputs"
os.makedirs(inputs_dir, exist_ok=True)

names = ["fp8", "fp16", "fp32", "fp64", "fp4"]
num_files_per_name = 10
vocab_size = 10
logit_scale = 10

np.random.seed(42)

# Generate sample data
for name in names:
    for i in range(1, num_files_per_name + 1):
        logits = np.random.uniform(-logit_scale, logit_scale, vocab_size).tolist()
        reference_logits = np.random.uniform(-logit_scale, logit_scale, vocab_size).tolist()
        logprob = np.log(softmax(logits)).max()
        reference_logprob = np.log(softmax(reference_logits)).max()

        data = [
            {
                "logits": logits,
                "reference_logits": reference_logits,
                "logprob": logprob,
                "reference_logprob": reference_logprob,
                "step": 1,
                "text": f"Sample text {i}",
                "token": i
            },
            {
                "avg_logprob": logprob,
                "perplexity": np.exp(-logprob)
            }
        ]

        filename = f"generation_{name}_{i}.json"
        filepath = os.path.join(inputs_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

print(f"Generated sample data for {len(names)} names, each with {num_files_per_name} files.")

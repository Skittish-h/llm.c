import json
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_kl_divergence(p1, p2):
    return entropy(p1, p2)

def compute_tvd(p1, p2):
    return 0.5 * np.sum(np.abs(p1 - p2))

def compute_mse(logits1, logits2):
    return np.mean((np.array(logits1) - np.array(logits2))**2)

def compute_cosine_similarity(logits1, logits2):
    return 1 - cosine(logits1, logits2)

def compare_runs_with_reference(inputs_dir, names, reference_name, metrics_to_plot):
    data_per_token = {metric: {name: [] for name in names} for metric in metrics_to_plot}
    reference_data = []

    # Load reference data
    reference_files = sorted(
        [f for f in os.listdir(inputs_dir) if f.startswith(f"generation_{reference_name}_") and f.endswith(".json")]
    )
    for file in reference_files:
        with open(os.path.join(inputs_dir, file), "r") as f:
            run_data = json.load(f)
            reference_data.append(run_data[0])  # Assuming single token per file

    # Compare each run with the reference
    for name in names:
        files = sorted(
            [f for f in os.listdir(inputs_dir) if f.startswith(f"generation_{name}_") and f.endswith(".json")]
        )
        for i, file in enumerate(files):
            with open(os.path.join(inputs_dir, file), "r") as f:
                run_data = json.load(f)
                token_data = run_data[0]  # Assuming single token per file

                logits1 = softmax(token_data["logits"])
                logits2 = softmax(reference_data[i]["logits"])

                kl_divergence = entropy(logits1, logits2)
                tvd = 0.5 * np.sum(np.abs(logits1 - logits2))
                mse = np.mean((np.array(token_data["logits"]) - np.array(reference_data[i]["logits"]))**2)
                cosine_sim = 1 - cosine(token_data["logits"], reference_data[i]["logits"])
                logprob_diff = token_data["logprob"]

                metrics = {
                    "kl_divergence": kl_divergence,
                    "tvd": tvd,
                    "mse": mse,
                    "cosine_similarity": cosine_sim,
                    "logprob_difference": logprob_diff,
                }

                for metric, value in metrics.items():
                    data_per_token[metric][name].append(value)

    return data_per_token

def generate_reference_comparison_plots(data_per_token, names, metrics_to_plot, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Per-token plots
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        token_indices = range(len(data_per_token[metric][names[0]]))  # Assuming same number of tokens per precision
        for token_index in token_indices:
            values = [data_per_token[metric][name][token_index] for name in names]
            plt.scatter(names, values, label=f"Generated Token {token_index + 1}")
        plt.xlabel("Precision")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} Per Token Compared to Reference")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_per_token_reference.png"))
        plt.close()

    # Aggregated bar plot
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        means = [np.mean(data_per_token[metric][name]) for name in names]
        stds = [np.std(data_per_token[metric][name]) for name in names]
        plt.bar(names, means, yerr=stds, capsize=5, color="skyblue")
        plt.xlabel("Precision")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Mean {metric.replace('_', ' ').title()} with Error Bars Across All Generated Tokens")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_mean_with_error_bars.png"))
        plt.close()

def compare_and_plot_with_reference(inputs_dir, output_dir, names, reference_name):
    metrics_to_plot = ["kl_divergence", "tvd", "mse", "cosine_similarity", "logprob_difference"]
    data_per_token = compare_runs_with_reference(inputs_dir, names, reference_name, metrics_to_plot)
    generate_reference_comparison_plots(data_per_token, names, metrics_to_plot, output_dir)

if __name__ == "__main__":
    inputs_dir = "saved_logits"
    output_dir = "lolout"
    names = ["FP16", "BF16", "FP32"]
    reference_name = "FP32"
    compare_and_plot_with_reference(inputs_dir, output_dir, names, reference_name)
    print(f"Plots comparing metrics to reference '{reference_name}' have been generated in '{output_dir}'.")

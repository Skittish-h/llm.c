/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#include <math.h>
#include <stdlib.h>

// Simple xorshift RNG
unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

struct indexed_value {
    int idx;
    float val;
};

// Comparison function for qsort to sort in descending order by val
static int compare_desc(const void *a, const void *b) {
    float valA = ((struct indexed_value*)a)->val;
    float valB = ((struct indexed_value*)b)->val;
    return (valA > valB) ? -1 : (valA < valB) ? 1 : 0;
}

int sample_softmax_topk(const float *logits, int n, float coin, int k, float temp) {
    if (k <= 0 || k > n) {
        // fallback to sampling over all if top_k is invalid
        k = n;
    }

    // Create an array of (index, value)
    struct indexed_value *arr = (struct indexed_value*)malloc(n * sizeof(struct indexed_value));
    for (int i = 0; i < n; i++) {
        arr[i].idx = i;
        arr[i].val = logits[i] / temp;
    }

    // Sort by value descending
    qsort(arr, n, sizeof(struct indexed_value), compare_desc);

    // Now the top_k elements are in arr[0:top_k-1]
    double norm = 0.0;
    for (int i = 0; i < k; i++) {
        norm += expf(arr[i].val);
    }

    // Scale the random coin [0,1) by the normalized sum
    coin *= norm;

    float cdf = 0.0f;
    for (int i = 0; i < k; i++) {
        cdf += expf(arr[i].val);
        if (coin < cdf) {
            int chosen = arr[i].idx;
            free(arr);
            return chosen;
        }
    }

    // Fallback (rare in case of floating-point issues)
    int chosen = arr[k - 1].idx;
    free(arr);
    return chosen;
}

int sample_softmax_topk_topp(const float *logits, int n, float coin, int k, float p, float temp) {
    if (k <= 0 || k > n) {
        k = n;  // Fallback: consider all tokens if top_k is invalid
    }
    if (p <= 0.0f || p > 1.0f) {
        p = 1.0f;  // Fallback: consider full cumulative probability if p is invalid
    }

    // Create an array of (index, value)
    struct indexed_value *arr = (struct indexed_value*)malloc(n * sizeof(struct indexed_value));
    for (int i = 0; i < n; i++) {
        arr[i].idx = i;
        arr[i].val = logits[i] / temp;
    }

    // Sort by value descending
    qsort(arr, n, sizeof(struct indexed_value), compare_desc);

    // Apply Top-K: Restrict to top_k tokens
    int k_count = k;
    double norm = 0.0;
    for (int i = 0; i < k_count; i++) {
        norm += expf(arr[i].val);
    }

    // Apply Top-P: Find subset within Top-K that satisfies cumulative probability >= p
    float cumulative_prob = 0.0f;
    int p_count = 0;
    for (int i = 0; i < k_count; i++) {
        cumulative_prob += expf(arr[i].val) / norm;
        p_count++;
        if (cumulative_prob >= p) {
            break;
        }
    }

    // Normalize within the selected subset (p_count tokens)
    double subset_norm = 0.0;
    for (int i = 0; i < p_count; i++) {
        subset_norm += expf(arr[i].val);
    }

    // Scale the random coin [0,1) by the normalized sum of the top-p+k logits
    coin *= subset_norm;

    // Sample from the filtered subset
    float cdf = 0.0f;
    for (int i = 0; i < p_count; i++) {
        cdf += expf(arr[i].val);
        if (coin < cdf) {
            int chosen = arr[i].idx;
            free(arr);
            return chosen;
        }
    }

    // Fallback (rare in case of floating-point issues)
    int chosen = arr[p_count - 1].idx;
    free(arr);
    return chosen;
}

int sample_softmax(const float* logits, int n, float coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int sample_argmax(const float* logits, int n) {
    assert(n > 0);

    int idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            idx = i;
        }
    }
    return idx;
}

#endif
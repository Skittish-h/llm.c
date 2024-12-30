#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#define TESTING
#include "train_gpt2.cu"
#include "llmc/promptloader.h"

struct ParsedArgs {
    std::vector<int> tokens;
    int n_gen;
    int top_k;    // Top-K sampling
    float temp;   // Temperature for sampling
    float top_p;  // Top-P (nucleus) sampling
    int seed;     // Random seed for reproducibility
    char* in;     // input file path
    char* out;    // output file path
};

ParsedArgs parse_args(int argc, char *argv[]) {
    ParsedArgs result;
    result.n_gen = 100;
    result.top_k = 50;  
    result.temp = 1.0;
    result.top_p = 1.0;  // Default value for top_p
    result.seed = 42;    // Default value for seed (-1 means not set)
    result.in = "dev/data/testtest/test_32.bin";
    result.out = "out.txt";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tokens") {
            int j = i + 1;
            for (; j < argc; ++j) {
                std::string nextArg = argv[j];
                if (nextArg.rfind("--", 0) == 0) {
                    break;
                }
                result.tokens.push_back(std::atoi(nextArg.c_str()));
            }
            i = j - 1;
        } else if (arg == "--n_gen") {
            if (i + 1 < argc) {
                result.n_gen = std::atoi(argv[i + 1]);
                i += 1;
            } else {
                std::cerr << "Error: --n_gen flag provided but no integer value found.\n";
            }
        } else if (arg == "--top_k") {
            if (i + 1 < argc) {
                result.top_k = std::atoi(argv[i + 1]);
                i += 1;
            } else {
                std::cerr << "Error: --top_k flag provided but no integer value found.\n";
            }
        } else if (arg == "--temp") {
            if (i + 1 < argc) {
                result.temp = std::atof(argv[i + 1]);
                i += 1;
            } else {
                std::cerr << "Error: --temp flag provided but no float value found.\n";
            }
        } else if (arg == "--top_p") {
            if (i + 1 < argc) {
                result.top_p = std::atof(argv[i + 1]);
                i += 1;
            } else {
                std::cerr << "Error: --top_p flag provided but no float value found.\n";
            }
        } else if (arg == "--seed") {
            if (i + 1 < argc) {
                result.seed = std::atoi(argv[i + 1]);
                i += 1;
            } else {
                std::cerr << "Error: --seed flag provided but no integer value found.\n";
            }
        } else if (arg == "--in") {
            if (i + 1 < argc) {
                result.in = argv[i + 1];
                i += 1;
            } else {
                std::cerr << "Error: --in flag provided but no string found.\n";
            }
        } else if (arg == "--out") {
            if (i + 1 < argc) {
                result.out = argv[i + 1];
                i += 1;
            } else {
                std::cerr << "Error: --out flag provided but no string found.\n";
            }
        }
    }
    return result;
}

int main(int argc, char *argv[]) {
    ParsedArgs args = parse_args(argc, argv);

    // this is a very important line
    common_start(false, true);

    //todo: extract from args
    const char* load_filename = "gpt2_124M.bin";

    // load model
    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, load_filename);
    model.requires_grad = false;

    // load tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    char nccl_init_method[256] = "mpi";  // "tcp" or "fs" or "mpi"
    char server_ip[256] = "";  // doesn't matter when using MPI
    char fs_path[256] = "";  // doesn't matter when using MPI
    multi_gpu_config = multi_gpu_config_init(
        -1, // num processes
        -1, // process rank
        -1, // gpus per node
        server_ip,
        fs_path,
        nccl_init_method
    );
    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);

    size_t V = model.config.vocab_size;
    size_t Vp = model.config.padded_vocab_size;
    size_t maxT = model.config.max_seq_len;
    size_t L = model.config.num_layers;
    size_t C = model.config.channels;

    // batch size
    int B = 1;
    // sequence length
    int T = 32;
    assert(0 <= T && T <= maxT);

    PromptLoader loader;
    promptloader_init(&loader, args.in, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes);
    gpt2_allocate_state(&model, B, T);
 
    int gen_tokens[B * T];
    double logprob_sum = 0;
    floatX* cpu_logits_raw = (floatX*) mallocCheck(model.config.vocab_size * sizeof(floatX));
    float* cpu_logits = (float*) mallocCheck(model.config.vocab_size * sizeof(float));
    int eot_token = tokenizer.eot_token;

    unsigned long long sample_rng_state = (unsigned long long)args.seed;

    promptloader_next_batch(&loader);
    int t = 1;
    while (t < T && loader.inputs[t] != eot_token)
    {
        t += 1;
    }
    for (; t < T; t++) {
        gpt2_forward(&model, loader.inputs, B, CEIL_DIV(t, min(T, 256)) * min(T, 256));
        // get the V-dimensional vector probs[0, t-1, :]
        floatX* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
        // move logits back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
        cudaCheck(cudaMemcpy(cpu_logits_raw, logits, model.config.vocab_size * sizeof(floatX), cudaMemcpyDeviceToHost));
        // convert to FP32 into cpu_logits (this does nothing useful if floatX == float)
        for (int i = 0; i < model.config.vocab_size; i++) {
            cpu_logits[i] = (float)cpu_logits_raw[i];
        }
        // sample the next token
        float coin = random_f32(&sample_rng_state);
        int next_token = sample_softmax_topk_topp(cpu_logits, model.config.vocab_size, coin, args.top_k, args.top_p, args.temp);
        // int next_token = sample_argmax(cpu_logits, model.config.vocab_size);
        gen_tokens[t] = next_token;

        float logprob = compute_logprob(cpu_logits, model.config.vocab_size, next_token);
        logprob_sum += logprob;

        const char* token_str = tokenizer_decode(&tokenizer, next_token);
        safe_printf(token_str);
        fflush(stdout);
    }

    float avg_logprob = logprob_sum / args.n_gen;
    float perplexity = expf(-avg_logprob);

    printf("\nAvg logp: %f\n", avg_logprob);
    printf("Perplexity: %f\n", perplexity);

    gpt2_free(&model);
    promptloader_free(&loader);
    tokenizer_free(&tokenizer);

    return 0;
}
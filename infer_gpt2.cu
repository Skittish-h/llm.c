#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#define TESTING
#include "train_gpt2.cu"


struct ParsedArgs {
    std::vector<int> tokens;
    int n_gen;
    int top_k;    // New flag: top_k
    float temp;   // New flag: temperature
};

ParsedArgs parse_args(int argc, char *argv[]) {
    ParsedArgs result;
    result.n_gen = 100;
    result.top_k = 50;  
    result.temp = 1.0;

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

    model.requires_grad = true;

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
    int T = 1024;
    assert(0 <= T && T <= maxT);

    gpt2_allocate_state(&model, B, T);
 
    int gen_tokens[B * T];
    floatX* cpu_logits_raw = (floatX*) mallocCheck(model.config.vocab_size * sizeof(floatX));
    float* cpu_logits = (float*) mallocCheck(model.config.vocab_size * sizeof(float));
    int eot_token = tokenizer.eot_token;

    for(size_t i = 0; i < B * T; ++i) {
        gen_tokens[i] = eot_token;
    }

    for (size_t i = 0; i < args.tokens.size() && i < B * T; ++i) {
        gen_tokens[i] = args.tokens[i];
    }

    int genT = args.n_gen + args.tokens.size();
    unsigned long long sample_rng_state = 42;

    for (int t = args.tokens.size(); t < genT; t++) {
        gpt2_forward(&model, gen_tokens, B, CEIL_DIV(t, min(T, 256)) * min(T, 256));
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
        int next_token = sample_softmax_topk(cpu_logits, model.config.vocab_size, coin, args.top_k, args.temp);
        // int next_token = sample_argmax(cpu_logits, model.config.vocab_size);
        gen_tokens[t] = next_token;

        const char* token_str = tokenizer_decode(&tokenizer, next_token);
        safe_printf(token_str);
        fflush(stdout);
    }

    gpt2_free(&model);
    tokenizer_free(&tokenizer);

    return 0;
}
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <cmath>     // for expf
#include "packages/json.h"

#define TESTING
#include "llmc/promptloader.h"
#include "train_gpt2.cu"

using json = nlohmann::json;

/**
 * Program arguments
 */
struct ParsedArgs {
    std::vector<int> tokens;  // Not used heavily, but keep for consistency
    int n_gen;                // We'll no longer use this for generation
    int top_k;                // Unused in perplexity mode
    float temp;               // Unused in perplexity mode
    float top_p;              // Unused in perplexity mode
    int seed;                 // RNG seed (if needed)
    std::string in;           // Input file path
    std::string out;          // Output file path
    int T;                    // Token length
    std::string name;         // Used for naming output files
};

ParsedArgs parse_args(int argc, char* argv[]) {
    ParsedArgs result;
    result.n_gen = 100;              // Not used in perplexity mode
    result.top_k = 50;               // Not used in perplexity mode
    result.temp = 1.0;               // Not used in perplexity mode
    result.top_p = 1.0;              // Not used in perplexity mode
    result.seed = 42;                // Default value for seed
    result.in = "infer_related_scripts/promptset/prompt_64.bin";
    result.out = "out.txt";
    result.T = 64;
    result.name = "default";

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
        } else if (arg == "--T") {
            if (i + 1 < argc) {
                result.T = std::atoi(argv[i + 1]);
                i += 1;
            } else {
                std::cerr << "Error: --T flag provided but no int found.\n";
            }
        } else if (arg == "--name") {
            if (i + 1 < argc) {
                result.name = argv[i + 1];
                i += 1;
            } else {
                std::cerr << "Error: --name flag provided but no string found.\n";
            }
        }
    }
    return result;
}

/**
 * The forward pass (same as original).
 */
void gpt2_forward_copyfree(GPT2 *model, size_t B, size_t T) {
    NVTX_RANGE_FN();

    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    const size_t V  = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L  = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C  = model->config.channels;

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;

    // Embedding + position
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C, main_stream);

    // First layernorm
    layernorm_forward((model->recompute < 2) ? acts.ln1 : acts.lnf,
                      acts.ln1_mean,
                      acts.ln1_rstd,
                      acts.encoded,
                      params.ln1w,
                      params.ln1b,
                      B, T, C, main_stream);

    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);

        floatX* residual = (l == 0) ? acts.encoded
                                    : acts.residual3 + (l - 1) * B * T * C;

        floatX* l_qkvw     = params.qkvw + l * 3*C * C;
        floatX* l_qkvb     = params.qkvb + l * 3*C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_attprojb = params.attprojb + l * C;
        floatX* l_ln2w     = params.ln2w + l * C;
        floatX* l_ln2b     = params.ln2b + l * C;
        floatX* l_fcw      = params.fcw + l * 4*C * C;
        floatX* l_fcb      = params.fcb + l * 4*C;
        floatX* l_fcprojw  = params.fcprojw + l * C * 4*C;
        floatX* l_fcprojb  = params.fcprojb + l * C;

        floatX* l_ln1       = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        floatX* l_qkvr      = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty      = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2       = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float*  l_ln2_mean  = acts.ln2_mean + l * B * T;
        float*  l_ln2_rstd  = acts.ln2_rstd + l * B * T;
        floatX* l_fch       = acts.fch + l * B * T * 4*C;
        floatX* l_fch_gelu  = (model->recompute < 1) ? acts.fch_gelu + l * B * T * 4*C
                                                     : acts.fch_gelu;
        floatX* l_residual3 = acts.residual3 + l * B * T * C;
        floatX* scratch     = (floatX*)acts.output;

#ifdef ENABLE_CUDNN
        float* l_att = (float*)acts.att + l * B * NH * T;
        matmul_forward_cublaslt(l_qkvr, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        attention_forward_cudnn(l_atty, (float*)l_att, l_qkvr, B, T, NH, C, main_stream);
#else
        floatX* l_att = acts.att + l * B * NH * T * T;
        cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX)));
        matmul_forward_cublaslt(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH, main_stream);
#endif

        // Project attention
        matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, main_stream);

        // Residual + Next LN
        fused_residual_forward5(l_residual2, l_ln2, l_ln2_mean, l_ln2_rstd,
                                residual, scratch, l_ln2w, l_ln2b, B*T, C, main_stream);

        // FC
        matmul_forward_cublaslt(l_fch_gelu, l_ln2, l_fcw, l_fcb,
                                B, T, C, 4*C, main_stream,
                                l_fch, /*use_gelu_fusion=*/model->gelu_fusion);

        matmul_forward_cublaslt(scratch, l_fch_gelu, l_fcprojw, l_fcprojb,
                                B, T, 4*C, C, main_stream);

        // Residual for the next layer
        if (l + 1 != L) {
            floatX* l_ln1_next    = (model->recompute < 2) ? acts.ln1 + (l+1) * B * T * C : acts.lnf;
            float*  l_ln1_mean    = acts.ln1_mean + (l+1) * B * T;
            float*  l_ln1_rstd    = acts.ln1_rstd + (l+1) * B * T;
            floatX* l_ln1w_next   = params.ln1w + (l+1) * C;
            floatX* l_ln1b_next   = params.ln1b + (l+1) * C;

            fused_residual_forward5(l_residual3, l_ln1_next,
                                    l_ln1_mean, l_ln1_rstd,
                                    l_residual2, scratch, l_ln1w_next, l_ln1b_next,
                                    B*T, C, main_stream);
        } else {
            fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd,
                                    l_residual2, scratch,
                                    params.lnfw, params.lnfb,
                                    B*T, C, main_stream);
        }
    }

    // Final projection to vocab
    matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL,
                            B, T, C, Vp, main_stream);

    cudaCheck(cudaDeviceSynchronize());
}

/**
 * Main: rolling-window perplexity
 */
int main(int argc, char *argv[]) {
    ParsedArgs args = parse_args(argc, argv);

    // Start environment
    common_start(false, true);

    // batch size
    int B = 1;
    // token length
    int T = args.T;

    // Load the model
    const char* load_filename = "gpt2_124M.bin";
    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, load_filename);
    model.requires_grad = false;

    // Safety check
    assert(0 <= T && T <= model.config.max_seq_len);

    // Init multi gpu config
    char nccl_init_method[256] = "mpi";
    char server_ip[256] = "";
    char fs_path[256] = "";
    multi_gpu_config = multi_gpu_config_init(-1, -1, -1, server_ip, fs_path, nccl_init_method);
    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);

    // Allocate state
    gpt2_allocate_state(&model, B, T);

    // Load tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // Load promptloader
    PromptLoader loader;
    promptloader_init(&loader, args.in.c_str(), B, T,
                      multi_gpu_config.process_rank,
                      multi_gpu_config.num_processes);

    // We'll need host-side buffers for logits
    floatX* cpu_logits_raw = (floatX*) mallocCheck(model.config.vocab_size * sizeof(floatX));
    float*  cpu_logits     = (float*)  mallocCheck(model.config.vocab_size * sizeof(float));

    // Keep track of total log prob
    double total_logprob_sum = 0.0;
    // Keep track of total tokens across all samples (excluding first position)
    size_t total_token_count = 0;

    int eot_token = tokenizer.eot_token;
    printf("Total tokens in loader: %zu\n", loader.shard_num_samples);

    printf("\n---\n");

    // For each sample in the prompt set
    for (size_t i = 0; i < loader.shard_num_samples; i++) {
        // Load the prompt from host -> device
        promptloader_next_batch(&loader);
        cudaCheck(cudaMemcpy(model.inputs, loader.inputs,
                             B * T * sizeof(int),
                             cudaMemcpyHostToDevice));

        // Print out the text (optional)
        printf("Sample %zu prompt:\n", i);
        int prompt_len = 0;
        while (prompt_len < T && loader.inputs[prompt_len] != eot_token) {
            safe_printf(tokenizer_decode(&tokenizer, loader.inputs[prompt_len]));
            prompt_len++;
        }
        printf("\n");


        // We'll compute rolling log probabilities for each next token
        // from t = 1 up to T-1 (since to get the next token log prob, we need
        // at least 1 token in context).
        for (int t = 1; t < T; t++) {
            // If the previous token was EOT, we can stop
            if (loader.inputs[t-1] == eot_token) {
                break;
            }
            // If the "target" token is also EOT, we can stop
            if (loader.inputs[t] == eot_token) {
                break;
            }

            // We'll run forward on the first "t" tokens of the prompt
            // Some models do better with chunking, but we'll keep it direct.
            int forward_len = t;  // from index 0..(t-1)

            // GPU kernels might need chunking. For small T it’s fine:
            int forward_len_ceiled = CEIL_DIV(forward_len, std::min(T, 256)) * std::min(T, 256);
            gpt2_forward_copyfree(&model, B, forward_len_ceiled);

            // The logits for token (t-1) are in offset (t - 1)
            int logits_index = t - 1;
            floatX* logits = model.acts.output + logits_index * model.config.padded_vocab_size;

            // Pull logits back to CPU
            cudaCheck(cudaMemcpy(cpu_logits_raw, logits,
                                 model.config.vocab_size * sizeof(floatX),
                                 cudaMemcpyDeviceToHost));
            for (int vi = 0; vi < model.config.vocab_size; vi++) {
                cpu_logits[vi] = (float) cpu_logits_raw[vi];
            }

            // The *actual* next token is loader.inputs[t].
            int actual_next = loader.inputs[t];

            // Log probability of the correct next token
            float logprob = compute_logprob(cpu_logits, model.config.vocab_size, actual_next);

            total_logprob_sum += logprob;
            total_token_count++;



            // If we want a "sliding" context rather than "prefix" context,
            // we could shift the tokens in model.inputs. But here we do
            // prefix-based rolling.
        }

        printf("Finished sample %zu.\n---\n", i);
    }

    // Final stats across the entire prompt set
    float final_avg_logprob = (float) (total_logprob_sum / (double) total_token_count);
    float final_ppl = expf(-final_avg_logprob);

    printf("Final avg logp across all tokens: %.6f\n", final_avg_logprob);
    printf("Final perplexity across all tokens: %.6f\n", final_ppl);

    // Clean up
    gpt2_free(&model);
    promptloader_free(&loader);
    tokenizer_free(&tokenizer);

    return 0;
}

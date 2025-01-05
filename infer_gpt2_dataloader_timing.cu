#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include "packages/json.h"
#define TESTING
#include "llmc/promptloader.h"
#include "train_gpt2.cu"

using json = nlohmann::json;

struct ParsedArgs {
    std::vector<int> tokens;  // List of integer tokens
    int n_gen;                // Number of tokens to generate (we'll override to 5)
    int top_k;                // Top-K sampling
    float temp;               // Temperature for sampling
    float top_p;              // Top-P (nucleus) sampling
    int seed;                 // Random seed for reproducibility
    std::string in;           // Input file path
    std::string out;          // Output file path
    int T;                    // Token length
    std::string name;
};

ParsedArgs parse_args(int argc, char* argv[]) {
    ParsedArgs result;
    result.n_gen = 100;
    result.top_k = 50;
    result.temp = 1.0;
    result.top_p = 1.0;  // Default value for top_p
    result.seed = 42;    // Default value for seed
    result.in = "dev/data/promptset/prompt_64.bin";
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
        }
        // parse --name - name of generated files
        else if (arg == "--name") {
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

// propagate inputs through the network to produce logits.
// right now, this function is fully synchronous with the host
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
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C, main_stream);

    // first layernorm isn't fused
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

        floatX* l_ln1        = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        floatX* l_qkvr       = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty       = acts.atty + l * B * T * C;
        floatX* l_residual2  = acts.residual2 + l * B * T * C;
        floatX* l_ln2        = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float*  l_ln2_mean   = acts.ln2_mean + l * B * T;
        float*  l_ln2_rstd   = acts.ln2_rstd + l * B * T;
        floatX* l_fch        = acts.fch + l * B * T * 4*C;
        floatX* l_fch_gelu   = (model->recompute < 1) ? acts.fch_gelu + l * B * T * 4*C
                                                      : acts.fch_gelu;
        floatX* l_residual3  = acts.residual3 + l * B * T * C;
        floatX* scratch      = (floatX*)acts.output;

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

        matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, main_stream);
        fused_residual_forward5(l_residual2, l_ln2, l_ln2_mean, l_ln2_rstd,
                                residual, scratch, l_ln2w, l_ln2b, B*T, C, main_stream);

        matmul_forward_cublaslt(l_fch_gelu, l_ln2, l_fcw, l_fcb,
                                B, T, C, 4*C, main_stream,
                                l_fch, /*use_gelu_fusion=*/model->gelu_fusion);

        matmul_forward_cublaslt(scratch, l_fch_gelu, l_fcprojw, l_fcprojb,
                                B, T, 4*C, C, main_stream);

        if (l+1 != L) {
            floatX* l_ln1_next     = (model->recompute < 2) ? acts.ln1 + (l+1) * B * T * C : acts.lnf;
            float*  l_ln1_mean     = acts.ln1_mean + (l+1) * B * T;
            float*  l_ln1_rstd     = acts.ln1_rstd + (l+1) * B * T;
            floatX* l_ln1w_next    = params.ln1w + (l+1) * C;
            floatX* l_ln1b_next    = params.ln1b + (l+1) * C;
            fused_residual_forward5(l_residual3, l_ln1_next, l_ln1_mean, l_ln1_rstd,
                                    l_residual2, scratch, l_ln1w_next, l_ln1b_next,
                                    B*T, C, main_stream);
        } else {
            fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd,
                                    l_residual2, scratch,
                                    params.lnfw, params.lnfb,
                                    B*T, C, main_stream);
        }
    }
    matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL,
                            B, T, C, Vp, main_stream);
    cudaCheck(cudaDeviceSynchronize());
}

int main(int argc, char *argv[]) {
    ParsedArgs args = parse_args(argc, argv);

    // Force only 5 tokens to be generated
    args.n_gen = 5;

    // Start environment
    common_start(false, true);

    // batch size
    int B = 1;
    // token length
    int T = args.T;

    const char* load_filename = "gpt2_124M.bin";
    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, load_filename);
    model.requires_grad = false;

    // safety check
    assert(0 <= T && T <= model.config.max_seq_len);

    // init multi gpu config
    char nccl_init_method[256] = "mpi";
    char server_ip[256] = "";
    char fs_path[256] = "";
    multi_gpu_config = multi_gpu_config_init(-1, -1, -1, server_ip, fs_path, nccl_init_method);
    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);

    gpt2_allocate_state(&model, B, T);

    // load tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // load promptloader
    PromptLoader loader;
    promptloader_init(&loader, args.in.c_str(), B, T,
                      multi_gpu_config.process_rank, multi_gpu_config.num_processes);

    // inference memory
    floatX* cpu_logits_raw = (floatX*) mallocCheck(model.config.vocab_size * sizeof(floatX));
    float*  cpu_logits     = (float*)  mallocCheck(model.config.vocab_size * sizeof(float));
    double  logprob_sum    = 0.0;

    int eot_token = tokenizer.eot_token;
    unsigned long long sample_rng_state = (unsigned long long)args.seed;

    printf("\n---\n");

    for (size_t i = 0; i < loader.shard_num_samples; i++) {
        // load the prompt
        promptloader_next_batch(&loader);
        cudaCheck(cudaMemcpy(model.inputs, loader.inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));

        // Print the current prompt
        printf("Prompt:\n");
        int t = 0;
        while (t < T && loader.inputs[t] != eot_token) {
            safe_printf(tokenizer_decode(&tokenizer, loader.inputs[t]));
            t++;
        }
        printf("\n");

        // JSON array to collect data for the N generated tokens
        json log_data = json::array();

        // We'll generate 5 tokens total:
        for (int gen_i = 0; gen_i < 5; gen_i++) {
            // Step 1: do the forward pass up to t tokens
            // But clamp forward_len to at least 1
            int forward_len = (t == 0) ? 1 : t;
            // Some GPU kernels often require T to be a multiple of 256 for streaming,
            // so we do something like: CEIL_DIV(forward_len, chunk_size) * chunk_size
            // but for small T, it's less relevant. We'll keep it for consistency:
            gpt2_forward_copyfree(&model, B,
                                  CEIL_DIV(forward_len, std::min(T, 256)) * std::min(T, 256));

            // Step 2: compute the next token from position t
            // If t == 0, then after forward pass, the logits for position 0 are in offset 0
            // else the logits for position t-1 are in offset (t - 1)
            int logits_index = (t == 0) ? 0 : (t - 1);
            floatX* logits = model.acts.output + logits_index * model.config.padded_vocab_size;

            cudaCheck(cudaMemcpy(cpu_logits_raw, logits,
                                 model.config.vocab_size * sizeof(floatX),
                                 cudaMemcpyDeviceToHost));
            for (int vi = 0; vi < model.config.vocab_size; vi++) {
                cpu_logits[vi] = (float) cpu_logits_raw[vi];
            }

            // sample the token
            float coin = random_f32(&sample_rng_state);
            // top-k, top-p, etc. For simplicity, we do argmax here:
            int next_token = sample_argmax(cpu_logits, model.config.vocab_size);

            // store the next token back onto the GPU if there's still space
            if (t < T) {
                cudaCheck(cudaMemcpy(model.inputs + t, &next_token, sizeof(int), cudaMemcpyHostToDevice));
            }

            // compute logprob
            float logprob = compute_logprob(cpu_logits, model.config.vocab_size, next_token);
            logprob_sum += logprob;

            // decode token text
            const char* token_str = tokenizer_decode(&tokenizer, next_token);

            printf("Generated token %d:\n", gen_i + 1);
            safe_printf(token_str);
            printf("\n---\n");

            // JSON object for this step
            {
                json token_data;
                token_data["step"]        = t;  // or t+1, depending how you want to index
//                token_data["token"]       = next_token;
//                token_data["text"]        = token_str;
                token_data["logprob"]     = logprob;

                // store input tokens up to t
                json input_tokens_json = json::array();
                for (int idx = 0; idx < t; idx++) {
                    input_tokens_json.push_back(loader.inputs[idx]);
                }
//                token_data["input_tokens"] = input_tokens_json;

                // store all logits
                json logits_json = json::array();
                for (int vi = 0; vi < model.config.vocab_size; vi++) {
                    logits_json.push_back(cpu_logits[vi]);
                }
                token_data["logits"] = logits_json;

                log_data.push_back(token_data);
            }

            // increment t to point to the newly generated token
            t++;

            // If we hit EOT, break early
            if (next_token == eot_token) {
                printf("EOT token encountered, stopping generation.\n");
                break;
            }

            // If we've used up all T positions, break
            if (t >= T) {
                printf("Reached maximum token length.\n");
                break;
            }
        } // end for (gen_i)

        // Once we have generated 5 tokens (or less, if we break early), write them out
        {
            char filename[512];
            sprintf(filename, "saved_logits/generation_%s_%zu.json", args.name.c_str(), i);
            std::ofstream outfs(filename);
            outfs << log_data.dump(4) << std::endl;
            outfs.close();
        }
    }

    // Optionally print final average logprob across all prompts
    float final_avg_logprob = (float)logprob_sum / loader.shard_num_samples;
    float final_ppl         = expf(-final_avg_logprob);
    printf("Final avg logp across all prompts: %f\n", final_avg_logprob);
    printf("Final perplexity across all prompts: %f\n", final_ppl);

    // cleanup
    gpt2_free(&model);
    promptloader_free(&loader);
    tokenizer_free(&tokenizer);

    return 0;
}

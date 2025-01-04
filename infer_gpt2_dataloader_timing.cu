#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>

#define TESTING
#include "llmc/promptloader.h"
#include "train_gpt2.cu"
#include "packages/json.h"

using json = nlohmann::json;

struct ParsedArgs {
    std::string in;  // input file path
    std::string out; // output file path
    int T;
};

ParsedArgs parse_args(int argc, char *argv[]) {
    ParsedArgs result;
    result.in = "dev/data/promptset/prompt_64.bin";
    result.out = "timings.json";
    result.T = 64;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i]; // Convert argv[i] to std::string
        
        if (arg == "--in") {
            if (i + 1 < argc) {
                result.in = argv[i + 1]; // Direct assignment works for std::string
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
        } else if (arg == "--t") {
            if (i + 1 < argc) {
                result.T = atoi(argv[i + 1]);
                i += 1;
            } else {
                std::cerr << "Error: --T flag provided but no int found.\n";
            }
        }
    }
    return result;
}

// propagate inputs through the network to produce logits.
// right now, this function is fully synchronous with the host
void gpt2_forward_copyfree(GPT2 *model, size_t B, size_t T) {
    NVTX_RANGE_FN();
    // we must be careful and use size_t instead of int, otherwise
    // we could overflow int. E.g. l * B * NH * T * T overflows int at B 16.

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;


    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C, main_stream); // encoding goes into residual[0]

    // first layernorm isn't fused
    layernorm_forward((model->recompute < 2) ? acts.ln1 : acts.lnf, acts.ln1_mean, acts.ln1_rstd, acts.encoded, params.ln1w, params.ln1b, B, T, C, main_stream);

    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);

        floatX* residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        floatX* l_qkvw = params.qkvw + l * 3*C * C;
        floatX* l_qkvb = params.qkvb + l * 3*C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_attprojb = params.attprojb + l * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = params.ln2b + l * C;
        floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcb = params.fcb + l * 4*C;
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        floatX* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch = acts.fch + l * B * T * 4*C;
        // reuse the same activation buffer at each layer, as we'll re-compute the gelu during backward
        // very useful because we dramatically reduce VRAM usage, and may be able to fit larger batch size
        floatX* l_fch_gelu = (model->recompute < 1) ? acts.fch_gelu + l * B * T * 4*C : acts.fch_gelu;
        floatX* l_residual3 = acts.residual3 + l * B * T * C;
        floatX* scratch = (floatX*)acts.output; // used for non-cudnn attention, fcproj, attproj, etc.

        // now do the forward pass
        #ifdef ENABLE_CUDNN
        float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
        matmul_forward_cublaslt(l_qkvr, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        attention_forward_cudnn(l_atty, (float*)l_att, l_qkvr, B, T, NH, C, main_stream);
        #else
        floatX* l_att = acts.att + l * B * NH * T * T;
        if (T != model->seq_len) { // unused parts of attention buffer must be zeroed (T-dependent)
            cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX)));
        }
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        matmul_forward_cublaslt(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH, main_stream);
        #endif

        matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, main_stream);
        fused_residual_forward5(l_residual2, l_ln2, l_ln2_mean, l_ln2_rstd, residual, scratch, l_ln2w, l_ln2b, B*T, C, main_stream);
        matmul_forward_cublaslt(l_fch_gelu, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, main_stream, l_fch, model->gelu_fusion);
        matmul_forward_cublaslt(scratch, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, main_stream);
        // OK, fusion across blocks.
        if(l+1 != L) {
            floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + (l + 1) * B * T * C : acts.lnf;
            float* l_ln1_mean = acts.ln1_mean + (l + 1) * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + (l + 1) * B * T;
            const floatX* l_ln1w = params.ln1w + (l + 1) * C;
            const floatX* l_ln1b = params.ln1b + (l + 1) * C;
            fused_residual_forward5(l_residual3, l_ln1, l_ln1_mean, l_ln1_rstd, l_residual2, scratch, l_ln1w, l_ln1b,
                                    B * T, C, main_stream);
        } else {
            fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd, l_residual2, scratch,
                                    params.lnfw, params.lnfb,
                                    B * T, C, main_stream);
        }
    }
    matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp, main_stream);
    cudaCheck(cudaDeviceSynchronize());
}

int main(int argc, char *argv[]) {
    ParsedArgs args = parse_args(argc, argv);

    // this is a very important line
    common_start(false, true);

    // Declare CUDA events
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Containers for tracking durations
    std::vector<float> model_init_durations;
    std::vector<float> copy_to_device_durations;
    std::vector<float> forward_pass_durations;
    std::vector<float> copy_to_host_durations;
    std::vector<float> sampling_durations;
    std::vector<float> copy_next_token_durations;

    float current_duration;

    // batch size
    int B = 1;
    // token length, need to be the same as the prompt datafile
    // TODO: set automatically from datafile header
    int T = args.T;

    const char* load_filename = "gpt2_124M.bin";
    GPT2 model;
    // init multi gpu config
    char nccl_init_method[256] = "mpi";  // "tcp" or "fs" or "mpi"
    char server_ip[256] = "";  // doesn't matter when using MPI
    char fs_path[256] = "";  // doesn't matter when using MPI

    // load model
    int num_initializations = 5;
    for (int i=0; i<num_initializations; i++) {
        cudaEventRecord(start);
        gpt2_init_common(&model);
        gpt2_build_from_checkpoint(&model, load_filename);
        model.requires_grad = false;

        assert(0 <= T && T <= model.config.max_seq_len);

        multi_gpu_config = multi_gpu_config_init(
            -1, // num processes
            -1, // process rank
            -1, // gpus per node
            server_ip,
            fs_path,
            nccl_init_method
        );
        set_zero_configs(&multi_gpu_config, 0, model.num_parameters);

        gpt2_allocate_state(&model, B, T);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        
        cudaEventElapsedTime(&current_duration, start, end);
        model_init_durations.push_back(current_duration);

        if (i < num_initializations - 1) gpt2_free(&model);
    }

    // load tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // load promptloader, T is read from the dataset file
    PromptLoader loader;
    promptloader_init(&loader, args.in.c_str(), B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes);

    //inference related memeroy allocation and settings
    floatX* cpu_logits_raw = (floatX*) mallocCheck(model.config.vocab_size * sizeof(floatX));
    float* cpu_logits = (float*) mallocCheck(model.config.vocab_size * sizeof(float));

    int eot_token = tokenizer.eot_token;

    printf("\n---\n");

    for (size_t i = 0; i < loader.shard_num_samples; i++)
    {
        // Print progress
        printf("Processing sample %zu of %zu (%.2f%% completed)\n", 
           i + 1, loader.shard_num_samples, 
           ((float)(i + 1) / loader.shard_num_samples) * 100);
        
        promptloader_next_batch(&loader);
        // copy inputs/targets to the model
        cudaEventRecord(start);
        cudaCheck(cudaMemcpy(model.inputs, loader.inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        
        cudaEventElapsedTime(&current_duration, start, end);
        copy_to_device_durations.push_back(current_duration);


        int t = 0;
        while (t < T && loader.inputs[t] != eot_token)
        {
            t++;
        }

        for (; t < T; t++) {
            cudaEventRecord(start);
            gpt2_forward_copyfree(&model, B, CEIL_DIV(t, min(T, 256)) * min(T, 256));
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&current_duration, start, end);
            forward_pass_durations.push_back(current_duration);

            // get the V-dimensional vector probs[0, t-1, :]
            floatX* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
            // move logits back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
            cudaEventRecord(start);
            cudaCheck(cudaMemcpy(cpu_logits_raw, logits, model.config.vocab_size * sizeof(floatX), cudaMemcpyDeviceToHost));
            // convert to FP32 into cpu_logits (this does nothing useful if floatX == float)
            for (int i = 0; i < model.config.vocab_size; i++) {
                cpu_logits[i] = (float)cpu_logits_raw[i];
            }
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&current_duration, start, end);
            copy_to_host_durations.push_back(current_duration);

            // sample the next token
            cudaEventRecord(start);
            float coin = random_f32(&sample_rng_state);
            int next_token = sample_argmax(cpu_logits, model.config.vocab_size);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&current_duration, start, end);
            sampling_durations.push_back(current_duration);

            cudaEventRecord(start);
            cudaCheck(cudaMemcpy(model.inputs + t, &next_token, sizeof(int), cudaMemcpyHostToDevice));
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&current_duration, start, end);
            copy_next_token_durations.push_back(current_duration);
        }

    }

    fflush(stdout);

    gpt2_free(&model);
    promptloader_free(&loader);
    tokenizer_free(&tokenizer);

    json timings;

    // Store raw durations for each category in the JSON object
    timings["model_initialization"] = model_init_durations;
    timings["copy_to_device"] = copy_to_device_durations;
    timings["forward_pass"] = forward_pass_durations;
    timings["copy_to_host"] = copy_to_host_durations;
    timings["sampling"] = sampling_durations;
    timings["copy_next_token"] = copy_next_token_durations;

    // Write JSON object to file
    std::ofstream out_file(args.out);
    if (out_file.is_open()) {
        out_file << timings.dump(4); // Pretty print with 4-space indentation
        out_file.close();
        printf("Raw durations written to %s\n", args.out.c_str());
    } else {
        printf("Error: Unable to open output file %s\n", args.out.c_str());
    }

    return 0;
}
#include <iostream>

#define TESTING
#include "train_gpt2.cu"


int main(int argc, char *argv[]) {
    // this is a very important line
    common_start(false, true);

    //todo: extract from args
    const char* load_filename = "gpt2_124M.bin";

    // load model
    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, load_filename);

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
    floatX* cpu_logits_raw = (floatX*)mallocCheck(model.config.vocab_size * sizeof(floatX));
    float* cpu_logits = (float*)mallocCheck(model.config.vocab_size * sizeof(float));
    int eot_token = tokenizer.eot_token;

    for(int i = 0; i < B * T; ++i) {
        prompt[i] = eot_token;
    }

    // "Hello Sir! "
    prompt[1] = 15496;
    prompt[2] = 7361;
    prompt[3] = 0;
    prompt[4] = 220;

    for (int t = 5; t < B * T; t++) {
        gpt2_forward(&model, prompt, B, T);

        // get the V-dimensional vector probs[0, t-1, :]
        floatX* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
        // move probs back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
        cudaCheck(cudaMemcpy(cpu_logits_raw, logits, model.config.vocab_size * sizeof(floatX), cudaMemcpyDeviceToHost));
        // convert to FP32 into cpu_logits (this does nothing useful if floatX == float)
        for (int i = 0; i < model.config.vocab_size; i++) {
            cpu_logits[i] = (float)cpu_logits_raw[i];
        }
        // sample the next token
        float coin = random_f32(&sample_rng_state);
        int next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;
    }

    gpt2_free(&model);
    tokenizer_free(&tokenizer);

    return 0;
}
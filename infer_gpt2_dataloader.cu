#include <iostream>

#define TESTING
#include "train_gpt2.cu"


int main(int argc, char *argv[]) {
    // this is a very important line
    common_start(false, true);

    //todo: extract from args
    const char* load_filename = "gpt2_124M.bin";

    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, load_filename);

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
    int T = 64;
    assert(0 <= T && T <= maxT);

    gpt2_allocate_state(&model, B, T);

    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_val.bin", B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, 0);

    for (int step = 0; step < 10; step++) {
        dataloader_next_batch(&loader);
        gpt2_forward(&model, loader.inputs, B, T);

        // logits shape: (B, T, Vp)
        floatX* logits = model.acts.output;
        floatX* logits_cpu = (floatX*)mallocCheck(B * T * Vp * sizeof(floatX));

        cudaCheck(cudaMemcpy(logits_cpu, logits, B * T * Vp * sizeof(floatX), cudaMemcpyDeviceToHost));
    }

    gpt2_free(&model);
    dataloader_free(&loader);

    return 0;
}
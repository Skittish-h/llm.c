"""
Tokenizes our GPU project prompt dataset for inference.
- Prompt from the input.txt are padded with eot tokens to the desired token length
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created promptset/ folder.
Timing inference trials need to be to the same token length as the input file.

"""

import argparse
import os
import tiktoken
import numpy as np

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "promptset")
debug = False

HEADERS_INFO = {
    "gpt-2": {
        "magic": 20240520,
        "version": 1,
        "token_dtype": np.uint16,
    },
    "llama-3": {
        "magic": 20240801,
        "version": 7,
        "token_dtype": np.uint32,
    },
}

def write_datafile(filename, toks, model_desc="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as uint16 (gpt-2) or uint32 (llama)
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    assert model_desc in ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = HEADERS_INFO[model_desc]
    # construct the header
    header = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
    header[0] = info["magic"]
    header[1] = info["version"]
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    num_bytes = (256 * 4) + (len(toks) * toks_np.itemsize)
    print(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def tokenize(n_tokens: int, input_filename: str):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode_ordinary(s)
    eot = enc._special_tokens['<|endoftext|>'] # end of text token

    data_filename = os.path.join(DATA_CACHE_DIR, input_filename)
    text = open(data_filename, 'r').read()

    prompts = text.split("\n")
    tokens = []
    for i, p in enumerate(prompts):
        temp = encode(p)
        tokens.extend(temp)
        tokens.extend([eot] * (n_tokens - len(temp)))

    # save to file
    tokens_filename = os.path.join(DATA_CACHE_DIR, f"prompt_{n_tokens}.bin")
    write_datafile(tokens_filename, tokens)
    if debug:
        plain_filename = os.path.join(DATA_CACHE_DIR, f"prompt_{n_tokens}.text")
        with open(plain_filename, 'w') as f:
            f.write(f'{tokens}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU project test set data preprocessing")
    parser.add_argument("-t", "--tokens", type=int, default=1024, help="Number of tokens for input")
    parser.add_argument("-i", "--input", type=str, default=os.path.join(DATA_CACHE_DIR, "input.txt"), help="Path to input file to tokenize")
    args = parser.parse_args()
    tokenize(args.tokens, args.input)
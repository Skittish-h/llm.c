"""
Tokenizes our GPU project prompt dataset for inference.
- Prompt from the input.txt are padded with eot tokens to the desired token length
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created promptset/ folder.
The script prints:
"""

import argparse
import os
import tiktoken
from data_common import download_file, write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "promptset")
debug = False

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
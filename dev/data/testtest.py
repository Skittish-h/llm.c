"""
Tokenizes our GPU project prompt dataset for inference.
- The download is from Github.
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created testtest/ folder.
The script prints:

$ python dev/data/testtest.py
writing 32,768 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin (66,560 bytes) in the gpt-2 format
writing 305,260 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin (611,544 bytes) in the gpt-2 format

And runs in a few seconds depending on your internet
connection and computer. The .bin files are raw byte
streams of uint16 numbers indicating the token ids.
"""

import argparse
import os
import tiktoken
from data_common import download_file, write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "testtest")

def tokenize(n_tokens: int, input_filename: str):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode_ordinary(s)
    eot = enc._special_tokens['<|endoftext|>'] # end of text token

    data_filename = os.path.join(DATA_CACHE_DIR, input_filename)
    text = open(data_filename, 'r').read()

    prompts = text.split("\n")
    tokens = []
    for i, p in enumerate(prompts):
        tokens.append(eot)
        temp = encode(p)
        tokens.extend(temp)
        tokens.extend([eot] * (n_tokens - len(temp) - 1))

    # save to file
    tokens_filename = os.path.join(DATA_CACHE_DIR, f"test_{n_tokens}.bin")
    write_datafile(tokens_filename, tokens)
    plain_filename = os.path.join(DATA_CACHE_DIR, f"test_{n_tokens}.text")
    with open(plain_filename, 'w') as f:
        f.write(f'{tokens}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU project test set data preprocessing")
    parser.add_argument("-t", "--tokens", type=int, default=1024, help="Number of tokens for input")
    parser.add_argument("-i", "--input", type=str, default=os.path.join(DATA_CACHE_DIR, "input.txt"), help="Path to input file to tokenize")
    args = parser.parse_args()
    tokenize(args.tokens, args.input)
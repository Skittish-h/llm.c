import argparse
import os
import requests
import tiktoken

# -----------------------------------------------------------------------------
# Adjust these paths as needed.
# We'll store:
#   1) The downloaded test file in PTB_DIR
#   2) The tokenized output .bin (and optional .txt for debugging) in OUTPUT_DIR
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(__file__)
PTB_DIR = os.path.join(SCRIPT_DIR, "ptb_data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "promptset")
os.makedirs(PTB_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Link to PTB test file
PTB_TEST_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
PTB_TEST_FILENAME = os.path.join(PTB_DIR, "ptb.test.txt")

# Debug flag: if True, we will write a .txt file listing all token IDs
debug = False

def download_ptb_test_file():
    """
    Downloads the Penn Treebank test file if not present locally.
    """
    if not os.path.exists(PTB_TEST_FILENAME):
        print(f"Downloading PTB test data from:\n  {PTB_TEST_URL}\n")
        r = requests.get(PTB_TEST_URL, timeout=30)
        r.raise_for_status()
        with open(PTB_TEST_FILENAME, "wb") as f:
            f.write(r.content)
    else:
        print(f"PTB test file already exists at:\n  {PTB_TEST_FILENAME}\n")

def write_datafile(path: str, data: list):
    """
    Writes a list of integers (token IDs) to a binary file in little-endian format.
    Adapt as needed for your pipeline.
    """
    import struct
    with open(path, "wb") as f:
        for token_id in data:
            f.write(struct.pack("<I", token_id))

def tokenize_ptb_test(input_file: str, n_tokens: int, output_prefix: str):
    """
    Tokenizes the PTB test file line-by-line using GPT-2 (tiktoken),
    pads or truncates each line to `n_tokens` tokens, then writes out
    a .bin file (and optionally a debug .txt).
    """
    # Initialize GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode_ordinary(s)

    # End-of-text token
    # NOTE: Accessing enc._special_tokens is not a public API;
    #       you could do eot = enc.encode("<|endoftext|>")[0] if you prefer.
    eot = enc._special_tokens['<|endoftext|>']

    # Read and strip lines
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    all_tokens = []
    # only take the first 100 lines
    for line in lines[:100]:
        # Encode line
        encoded_line = encode(line)

        # Truncate or pad
        if len(encoded_line) >= n_tokens:
            # Option A: Truncate
            padded_line = encoded_line[:n_tokens]
        else:
            # Option B: Pad with EOT tokens
            padded_line = encoded_line + [eot] * (n_tokens - len(encoded_line))

        all_tokens.extend(padded_line)

    # Write out the tokens to a .bin file
    output_bin = os.path.join(OUTPUT_DIR, f"{output_prefix}_{n_tokens}.bin")
    write_datafile(output_bin, all_tokens)
    print(f"Tokenized data written to: {output_bin}")

    # Optional debug: write a .text file listing the token IDs
    if debug:
        output_txt = os.path.join(OUTPUT_DIR, f"{output_prefix}_{n_tokens}.text")
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(str(all_tokens))
        print(f"(Debug) Token IDs written to: {output_txt}")

def main():
    parser = argparse.ArgumentParser(description="Download and tokenize PTB test file with GPT-2 tokenizer")
    parser.add_argument(
        "-t", "--tokens",
        type=int,
        default=128,
        help="Number of tokens per line (padding/truncation). Default=1024"
    )
    args = parser.parse_args()

    # 1) Download the PTB test file if needed
    download_ptb_test_file()

    # 2) Tokenize the file
    tokenize_ptb_test(PTB_TEST_FILENAME, args.tokens, "ptb_test")

if __name__ == "__main__":
    main()

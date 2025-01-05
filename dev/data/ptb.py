import argparse
import os
import requests
import tiktoken
import struct

SCRIPT_DIR = os.path.dirname(__file__)
PTB_DIR = os.path.join(SCRIPT_DIR, "ptb_data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "promptset")
os.makedirs(PTB_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Link to PTB test file
PTB_TEST_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
PTB_TEST_FILENAME = os.path.join(PTB_DIR, "ptb.test.txt")

# Debug flag
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
    """
    with open(path, "wb") as f:
        for token_id in data:
            f.write(struct.pack("<I", token_id))

def tokenize_ptb_test(input_file: str, n_tokens: int, output_prefix: str):
    """
    Tokenizes the PTB test file line-by-line (up to the first 100 lines),
    then pads each line to exactly `n_tokens` tokens (just like File 1).
    """
    # Initialize GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode_ordinary(s)

    # End-of-text token (same approach as File 1)
    eot = enc._special_tokens['<|endoftext|>']

    # Read lines
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    all_tokens = []
    # Process only the first 100 lines
    for line in lines[:100]:
        encoded_line = encode(line)

        # Pad with EOT tokens so each line becomes exactly n_tokens
        if len(encoded_line) < n_tokens:
            encoded_line += [eot] * (n_tokens - len(encoded_line))
        else:
            # If line is longer than n_tokens, just keep the first n_tokens
            encoded_line = encoded_line[:n_tokens]

        # Extend the cumulative token list
        all_tokens.extend(encoded_line)

    # Write out the tokens to a .bin file
    output_bin = os.path.join(OUTPUT_DIR, f"{output_prefix}_{n_tokens}.bin")
    write_datafile(output_bin, all_tokens)
    print(f"Tokenized data written to: {output_bin}")

    # Optional debug output
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
        default=64,  # Match File 1's default
        help="Number of tokens for each line (padding/truncation). Default=1024"
    )
    args = parser.parse_args()

    # 1) Download if needed
    download_ptb_test_file()

    # 2) Tokenize line-by-line, the first 100 lines, just like File 1
    tokenize_ptb_test(PTB_TEST_FILENAME, args.tokens, "ptb_test")

if __name__ == "__main__":
    main()

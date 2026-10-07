"""
SmoLLM Corpus dataset (for srs pretraining)
"""
import os
import argparse
import multiprocessing as mp

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer


from data_common import write_datafile
# ------------------------------------------

parser = argparse.ArgumentParser(description="Dataset preprocessing")
parser.add_argument("-v", "--version", type=str, default="cosmo", help="Fineweb data sample size, 10B|100B")
parser.add_argument("-m", "--model_desc", type=str, default="llama-3", help="Model descriptor llama-3")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each data shard in the output .bin files, in tokens")
args = parser.parse_args()

# FineWeb has a few possible subsamples available
assert args.version in {"cosmo", "python", "fineweb", "smollm"}, "version must be one of: cosmo, python, fineweb"
directories = {
    ("cosmo", "llama-3"): ("cosmo-llama3", "cosmopedia-v2"),
    ("python", "llama-3"): ("python-llama3", "python-edu"),
    ("fineweb", "llama-3"): ("fineweb-ddp-llama3", "fineweb-edu-dedup"),
    ("smollm", "llama-3"): ("smollm-llama3", "cosmopedia-v2"),
}
local_dir, remote_name = directories[(args.version, args.model_desc)]

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", local_dir)

# download the dataset
fw = load_dataset("HuggingFaceTB/smollm-corpus", name=remote_name, split="train")
def map_to_common_format(example):
    return {
        'text': example['text'],
    }

name = "smollm-corpus"

def tokenize_llama(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    # Add special tokens if they're not already in the tokenizer
    special_tokens_dict = {'additional_special_tokens': ['<PROMPT>', '<RESPONSE>']}

    encode = lambda s: tokenizer.encode(s, add_special_tokens=False, verbose=False, split_special_tokens=True)
    eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    tokens = [eot] # the special <|endoftext|> token delimits all documents

    # Combine prompt and text with special tokens
    text = doc["text"]

    # Tokenize the combined text
    tokens.extend(encode(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
    tokens_np_uint = tokens_np.astype(np.uint32)
    return tokens_np_uint

token_dtype = {
    "llama-3": np.uint32
}[args.model_desc]

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((args.shard_size,), dtype=token_dtype)
    token_count = 0
    progress_bar = None

    tokenize = lambda x: None
    if args.model_desc == "llama-3":
        tokenize = tokenize_llama
    else:
        raise ValueError(f"unknown model {args.model_desc}")

    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np.tolist(), args.model_desc)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_tokens_np[:token_count]).tolist(), args.model_desc)
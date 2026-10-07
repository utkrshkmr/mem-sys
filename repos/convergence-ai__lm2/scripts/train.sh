#!/bin/bash

# Define your Hydra parameters as variables
input_bin="datasets/cosmo-llama3-text/smollm-corpus_train_*.bin"
input_val_bin="datasets/cosmo-llama3-text/smollm-corpus_val_*.bin"
batch_size=1
sequence_length=2048
dtype="bfloat16"
learning_rate=0.0002
warmup_iters=700
learning_rate_decay_frac=0.0

# Run the Hydra-enabled Python script with the parameters passed as command-line overrides
torchrun --nproc_per_node=1 train.py model=llama3.2-1b\
    pretrain=default \
    input_bin="$input_bin" \
    input_val_bin="$input_val_bin" \
    model.sequence_length=$sequence_length \
    model.use_memory=true \
    train.batch_size=$batch_size \
    train.dtype=$dtype \
    train.learning_rate=$learning_rate \
    train.warmup_iters=$warmup_iters \
    train.lr_decay_frac=$learning_rate_decay_frac \
    train.max_iters=600000 \
    train.log_freq=10 \
    train.save_freq=10000
"""
Author: Jaxon Kang
Date: 2024-10-09 08:21:51
LastEditTime: 2024-10-09 08:26:01
LastEditors: Jaxon Kang
FilePath: /LMLM/src/data_loader.py
"""

import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from src.utils import print0


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    # if header[0] != 20240520:
    if header[0] != 20240801:
        print0("ERROR: magic number mismatch in the data .bin file!")
        print0("---> HINT: Are you passing in a correct file with --input_bin?")
        print0("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print0("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    # assert header[1] == 1, "unsupported version"
    assert header[1] == 7, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        # assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[0] == 20240801, "magic number mismatch in the data .bin file"
        # assert header[1] == 1, "unsupported version"
        assert header[1] == 7, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        # tokens = np.frombuffer(f.read(), dtype=np.uint16)
        tokens = np.frombuffer(f.read(), dtype=np.uint32)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class TokenDataset(Dataset):
    """Dataset wrapper for the token data to work with DistributedSampler"""

    def __init__(self, tokens: np.ndarray, B: int, T: int):
        self.tokens = tokens
        self.B = B
        self.T = T
        # Calculate number of complete batches
        self.n_batches = (len(tokens) - 1) // (B * T)

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.B * self.T
        buf = self.tokens[start_idx : start_idx + self.B * self.T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(self.B, self.T)
        y = (buf[1:]).view(self.B, self.T)
        return x, y


class DistributedDataLoader:
    def __init__(self, filename_pattern: str, B: int, T: int, process_rank: int, num_processes: int):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total

        if process_rank == 0:
            print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        self.current_shard = None
        self.reset()

    def reset(self):
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])

        # Create dataset and sampler for current shard
        self.dataset = TokenDataset(self.tokens, self.B, self.T)
        self.sampler = DistributedSampler(
            self.dataset, num_replicas=self.num_processes, rank=self.process_rank, shuffle=True
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,  # We handle batching in the dataset
            sampler=self.sampler,
            num_workers=0,  # Since we're already loading data in parallel
        )
        self.iterator = iter(self.dataloader)

    def set_epoch(self, epoch: int):
        """Set the epoch number for proper shuffling"""
        self.sampler.set_epoch(epoch)

    def advance(self):
        """Advance to next data shard"""
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.tokens = _load_data_shard(self.files[self.current_shard])

        # Create new dataset and sampler for the new shard
        self.dataset = TokenDataset(self.tokens, self.B, self.T)
        self.sampler = DistributedSampler(
            self.dataset, num_replicas=self.num_processes, rank=self.process_rank, shuffle=True
        )
        self.dataloader = DataLoader(self.dataset, batch_size=1, sampler=self.sampler, num_workers=0)
        self.iterator = iter(self.dataloader)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            x, y = next(self.iterator)
            return x.squeeze(0), y.squeeze(0)  # Remove batch dimension added by DataLoader
        except StopIteration:
            self.advance()
            x, y = next(self.iterator)
            return x.squeeze(0), y.squeeze(0)

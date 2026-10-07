"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import os
import random
from collections import defaultdict
import torch
from src.utils import CfgNode as CN, print0
import torch.distributed as dist


class Trainer:
    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = "auto"
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, optimizer, train_loader, local_rank, grad_accum_steps, iter_num=0):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.callbacks = defaultdict(list)

        self.device = self.local_rank = local_rank
        self.model = self.model.to(self.device)

        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
        self.ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

        # Initialize training state
        self.iter_num = iter_num
        self.log_freq = config.log_freq
        self.save_freq = config.save_freq
        self.grad_accum_steps = grad_accum_steps
        self.ddp = True

        # Set initial learning rate and verify state if resuming
        if iter_num > 0:
            self.verify_training_state(optimizer.state_dict(), iter_num)

    def verify_training_state(self, state_dict, iter_num):
        """Verify training state after loading checkpoint"""
        # Check learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        expected_lr = self.get_lr(iter_num)
        if abs(current_lr - expected_lr) > 1e-6:
            print0(f"Warning: LR mismatch. Current: {current_lr}, Expected: {expected_lr}")
            # Fix learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = expected_lr

        # Verify optimizer state device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def save_training_state(self, model, snapshot_dir, current_iter):
        """Save complete training state including RNG states"""
        ckpt_path = os.path.join(snapshot_dir, f"ckpt_iter_{current_iter}.pth")

        torch_rng_state = torch.get_rng_state().cpu()  # Make sure it's on CPU
        cuda_rng_states = [state.cpu() for state in torch.cuda.get_rng_state_all()]

        training_state = {
            "iteration": current_iter,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.loss.item(),
            "random_state": random.getstate(),
            "torch_random_state": torch_rng_state,
            "cuda_random_state": cuda_rng_states,
            "lr": self.get_lr(current_iter),
        }
        torch.save(training_state, ckpt_path)
        return ckpt_path

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def get_lr(self, it):
        """Learning rate decay scheduler (cosine with warmup)"""
        min_lr = self.config.learning_rate * self.config.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return self.config.learning_rate * (it + 1) / self.config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.max_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config.warmup_iters) / (self.config.max_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (self.config.learning_rate - min_lr)

    def run(self, current_time, iter_num):
        model, config = self.model, self.config
        self.iter_num = iter_num

        saved_snapshots = []
        snapshot_dir = f"{config.ckpt_dir}_{current_time}/"
        os.makedirs(snapshot_dir, exist_ok=True)

        # Setup the optimizer with correct initial state
        self.optimizer = model.module.configure_optimizers(config)
        if iter_num > 0:
            self.verify_training_state(self.optimizer.state_dict(), iter_num)

        while True:
            # Training section
            model.train()
            self.optimizer.zero_grad(set_to_none=True)
            self.lossf = 0.0

            for micro_step in range(self.grad_accum_steps):
                self.train_loader.set_epoch(self.iter_num)
                x, y = self.train_loader.next_batch()
                x = x.to(self.device)
                y = y.to(self.device)

                if self.ddp:
                    model.require_backward_grad_sync = micro_step == self.grad_accum_steps - 1

                # Forward pass
                with self.ctx:
                    _, loss, _ = model(x, targets=y, attention_mask=None, iter_num=self.iter_num)
                    loss = loss / self.grad_accum_steps
                    self.lossf += loss.detach()

                # Backward pass
                loss.backward(retain_graph=False)

            if self.ddp:
                dist.all_reduce(self.lossf, op=dist.ReduceOp.AVG)
            self.loss = self.lossf

            # Gradient clipping and optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            lr = self.get_lr(self.iter_num)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.optimizer.step()

            self.trigger_callbacks("on_batch_end")
            self.iter_num += 1

            # Save checkpoint if needed
            if self.iter_num % self.save_freq == 0 and dist.get_rank() == 0:
                ckpt_path = self.save_training_state(model, snapshot_dir, self.iter_num)
                saved_snapshots.append(ckpt_path)
                print0(f"Checkpoint saved at iteration {self.iter_num}")

                # Remove older checkpoints if needed
                if len(saved_snapshots) > config.max_ckpts_to_keep:
                    oldest_ckpt = saved_snapshots.pop(0)
                    if os.path.exists(oldest_ckpt):
                        os.remove(oldest_ckpt)
                        print0(f"Removed old checkpoint: {oldest_ckpt}")

            # Check termination condition
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

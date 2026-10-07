import hydra
import os
import torch
from src.model_memory_llama import CustomLlamaConfig, LlamaMem
from src.dataloader import DistributedDataLoader
from datetime import datetime
from src.trainer import Trainer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoConfig

from src.utils import print0, set_seed


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def load_model_state(model, optimizer_state_dict, rank, train_config):
    """Helper function to load and verify model state"""
    if optimizer_state_dict:
        try:
            optimizer = model.configure_optimizers(train_config)
            optimizer.load_state_dict(optimizer_state_dict)
            # Ensure states are on correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(rank)
            print0("Successfully loaded optimizer state")
            return optimizer
        except Exception as e:
            print0(f"Warning: Failed to load optimizer state: {str(e)}")
            return model.configure_optimizers(train_config)
    return model.configure_optimizers(train_config)


def verify_model_state(model, rank):
    """Verify model parameters are on correct device"""
    for name, param in model.named_parameters():
        if not param.is_cuda:
            raise RuntimeError(f"Parameter {name} is not on CUDA device {rank}")
    print0(f"Verified all model parameters are on device {rank}")


@hydra.main(config_path="configs/", config_name="train", version_base=None)
def main(cfg):
    print0(f"=====>Script arguments:\n{cfg}")
    ddp_setup()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up reproducibility
    set_seed(cfg.seed)

    """Model configs"""
    model_config = LlamaMem.get_default_config()
    model_config.model_type = cfg.model.model_type
    model_config.model_name = cfg.model.model_name
    model_config.vocab_size = cfg.model.vocab_size
    model_config.block_size = cfg.model.sequence_length
    model_config.max_length = cfg.model.max_length
    model_config.memory_slots = cfg.model.memory_slots
    model_config.n_layer = cfg.model.n_layer
    model_config.n_head = cfg.model.n_head
    model_config.n_embd = cfg.model.n_embd
    model_config.use_memory = cfg.model.use_memory
    model_config.log_freq = cfg.train.log_freq
    model_config.beta_coeff = cfg.model.beta_coeff
    model_config.seq_length = cfg.model.sequence_length
    model_config.num_mem_heads = cfg.model.num_mem_heads
    model_config.batch_size = cfg.train.batch_size

    """Training configs"""
    train_config = Trainer.get_default_config()
    train_config.max_iters = cfg.train.max_iters
    train_config.num_workers = 0
    train_config.ckpt_dir = cfg.train.ckpt_dir
    train_config.max_ckpts_to_keep = cfg.train.max_ckpts_to_keep
    train_config.log_freq = cfg.train.log_freq
    train_config.save_freq = cfg.train.save_freq
    train_config.dtype = cfg.train.dtype
    train_config.learning_rate = cfg.train.learning_rate
    train_config.learning_rate_decay_frac = cfg.train.lr_decay_frac
    train_config.warmup_iters = cfg.train.warmup_iters
    train_config.batch_size = cfg.train.batch_size
    train_config.num_mem_heads = cfg.model.num_mem_heads
    train_config.memory_slots = cfg.model.memory_slots

    # Set up distributed training parameters
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    B = cfg.train.batch_size
    T = cfg.model.sequence_length

    """Set up training and validation dataset"""
    train_loader = DistributedDataLoader(cfg.input_bin, B, T, rank, world_size)
    valid_dataset = DistributedDataLoader(cfg.input_val_bin, B, T, rank, world_size)

    iter_num = 0
    optimizer_state_dict = None

    try:
        if cfg.model.model_type == "llama":
            base_config = AutoConfig.from_pretrained(cfg.model.model_name).to_dict()
            config = CustomLlamaConfig(
                use_memory=model_config.use_memory,
                memory_slots=model_config.memory_slots,
                num_mem_heads=model_config.num_mem_heads,
                **base_config,
            )

            tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
            tokenizer.pad_token = tokenizer.eos_token

            if cfg.pretrain.snapshot_path:
                print0(f"Loading model from checkpoint: {cfg.pretrain.snapshot_path}")
                model, optimizer_state_dict, iter_num = LlamaMem.from_ckpt(
                    cfg.pretrain.snapshot_path,
                    config=config,
                    tokenizer=tokenizer,
                    rank=rank,
                    load_memory=cfg.pretrain.load_mem,
                    resume_training=True,
                )
                print0(f"Successfully loaded checkpoint at iteration {iter_num}")
            else:
                print0(f"Initializing new model from {cfg.model.model_name}")
                model = LlamaMem.from_config(
                    config=config,
                    tokenizer=tokenizer,
                )

    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {str(e)}")

    # Move model to device and verify
    model = model.to(rank)
    verify_model_state(model, rank)

    # Configure optimizer with state restoration
    optimizer = load_model_state(model, optimizer_state_dict, rank, train_config)

    # Wrap model in DDP
    model = DDP(model, find_unused_parameters=False)

    # Define trainer
    trainer = Trainer(
        train_config,
        model,
        optimizer,
        train_loader=train_loader,
        local_rank=rank,
        grad_accum_steps=1,
        iter_num=iter_num,
    )

    def batch_end_callback(trainer):
        if trainer.iter_num % trainer.log_freq == 0:
            # validation loss
            valid_loader = valid_dataset
            valid_loader.reset()

            trainer.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i in range(100):
                    valid_loader.set_epoch(i)
                    x, y = valid_loader.next_batch()
                    x = x.to(trainer.device)
                    y = y.to(trainer.device)

                    _, loss, _ = model(x, y)
                    val_loss += loss.item()
                val_loss /= 100

                # Aggregate losses from all nodes
                val_loss_tensor = torch.tensor(val_loss, device=trainer.device)
                train_loss_tensor = torch.tensor(trainer.loss.item(), device=trainer.device)

                # Sum losses across all nodes
                torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(train_loss_tensor, op=torch.distributed.ReduceOp.SUM)

                # Calculate mean across nodes
                world_size = torch.distributed.get_world_size()
                val_loss = val_loss_tensor.item() / world_size
                train_loss = train_loss_tensor.item() / world_size

            if int(os.environ["RANK"]) == 0:
                print0(f"iter {trainer.iter_num}: train loss {train_loss:.5f}, valid loss {val_loss:.5f}")

    trainer.set_callback("on_batch_end", batch_end_callback)

    trainer.run(current_time, iter_num)

    destroy_process_group()


if __name__ == "__main__":
    main()

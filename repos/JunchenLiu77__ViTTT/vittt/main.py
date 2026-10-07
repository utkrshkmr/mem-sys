# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, load_pretrained

import wandb

import warnings
warnings.filterwarnings('ignore')

def parse_option():
    parser = argparse.ArgumentParser('ViT^3 training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--pretrained', type=str, help='Finetune 384 initial checkpoint.', default='')
    parser.add_argument('--find-unused-params', action='store_true', default=False)
    
    # training hyperparameters
    parser.add_argument('--lr', type=float, help='base learning rate (before linear scaling)')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--warmup-epochs', type=int, help='number of warmup epochs')
    parser.add_argument('--cooldown-epochs', type=int, help='number of cooldown epochs')
    parser.add_argument('--weight-decay', type=float, help='weight decay')
    parser.add_argument('--min-lr', type=float, help='minimum learning rate')
    parser.add_argument('--clip-grad', type=float, help='gradient clipping max norm')
    parser.add_argument('--drop-path-rate', type=float, help='drop path rate')
    
    # wandb arguments
    parser.add_argument('--wandb-project', type=str, default='ViTTT', help='wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='wandb run name')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main():
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    args, config = parse_option()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    # Override config with command line arguments
    config.defrost()
    if args.lr is not None:
        config.TRAIN.BASE_LR = args.lr
    if args.epochs is not None:
        config.TRAIN.EPOCHS = args.epochs
    if args.warmup_epochs is not None:
        config.TRAIN.WARMUP_EPOCHS = args.warmup_epochs
    if args.cooldown_epochs is not None:
        config.TRAIN.COOLDOWN_EPOCHS = args.cooldown_epochs
    if args.weight_decay is not None:
        config.TRAIN.WEIGHT_DECAY = args.weight_decay
    if args.min_lr is not None:
        config.TRAIN.MIN_LR = args.min_lr
    if args.clip_grad is not None:
        config.TRAIN.CLIP_GRAD = args.clip_grad
    if args.drop_path_rate is not None:
        config.MODEL.DROP_PATH_RATE = args.drop_path_rate
    config.freeze()

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.LOCAL_RANK = local_rank
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    # Initialize wandb (only on rank 0)
    if dist.get_rank() == 0:
        wandb_run_name = args.wandb_run_name or f"{config.MODEL.NAME}_{config.TAG}" if config.TAG else config.MODEL.NAME
        wandb_id_file = os.path.join(config.OUTPUT, "wandb_id.txt")
        
        # Try to load existing wandb run id for resuming
        wandb_id = None
        if os.path.exists(wandb_id_file):
            with open(wandb_id_file, "r") as f:
                wandb_id = f.read().strip()
            logger.info(f"Found existing wandb run id: {wandb_id}, resuming...")
        
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,  # Use existing id if available, otherwise wandb generates a new one
            config={
                "model": config.MODEL.NAME,
                "model_type": config.MODEL.TYPE,
                "epochs": config.TRAIN.EPOCHS,
                "batch_size": config.DATA.BATCH_SIZE,
                "learning_rate": config.TRAIN.BASE_LR,
                "weight_decay": config.TRAIN.WEIGHT_DECAY,
                "img_size": config.DATA.IMG_SIZE,
                "drop_path_rate": config.MODEL.DROP_PATH_RATE,
                "optimizer": config.TRAIN.OPTIMIZER.NAME,
                "lr_scheduler": config.TRAIN.LR_SCHEDULER.NAME,
                "warmup_epochs": config.TRAIN.WARMUP_EPOCHS,
                "amp": config.AMP,
                "world_size": dist.get_world_size(),
            },
            dir=config.OUTPUT,
            resume="allow",
        )
        
        # Save the wandb run id for future resuming
        with open(wandb_id_file, "w") as f:
            f.write(wandb.run.id)
        logger.info(f"wandb initialized: {wandb.run.url} (id: {wandb.run.id})")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    
    _, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=True, find_unused_parameters=args.find_unused_params)
    model_without_ddp = model.module
    
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    total_epochs = config.TRAIN.EPOCHS + config.TRAIN.COOLDOWN_EPOCHS

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if args.pretrained != '':
        load_pretrained(args.pretrained, model_without_ddp, logger)

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, logger, epoch=config.TRAIN.START_EPOCH, args=args)
        max_accuracy = max(max_accuracy, acc1)
        torch.cuda.empty_cache()
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            if dist.get_rank() == 0:
                wandb.log({"eval/acc1": acc1, "eval/acc5": acc5, "eval/loss": loss})
                wandb.finish()
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, total_epochs):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, logger, total_epochs, args)
        acc1, acc5, loss = validate(config, data_loader_val, model, logger, epoch, args)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

        # Log validation metrics to wandb
        if dist.get_rank() == 0:
            wandb.log({
                "val/acc1": acc1,
                "val/acc5": acc5,
                "val/loss": loss,
                "val/max_acc1": max(max_accuracy, acc1),
                "epoch": epoch + 1,
            })

        if dist.get_rank() == 0 and ((epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == (total_epochs)):
            save_checkpoint(config, epoch + 1, model_without_ddp, max(max_accuracy, acc1), optimizer, lr_scheduler, logger)

        if dist.get_rank() == 0 and ((epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == (total_epochs)) and acc1 >= max_accuracy:
            save_checkpoint(config, epoch + 1, model_without_ddp, max(max_accuracy, acc1), optimizer, lr_scheduler, logger, name='max_acc')
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    
    # Log final summary to wandb
    if dist.get_rank() == 0:
        wandb.log({"final/max_acc1": max_accuracy, "final/training_time_hours": total_time / 3600})
        wandb.finish()


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, logger, total_epochs, args):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    scaler = GradScaler()

    for idx, (samples, targets) in enumerate(data_loader):

        optimizer.zero_grad()
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        if config.AMP: 
            with autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = get_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()

        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch + 1}/{total_epochs}][{idx + 1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            
            # Log training metrics to wandb
            if dist.get_rank() == 0:
                global_step = epoch * num_steps + idx
                wandb.log({
                    "train/loss": loss_meter.val,
                    "train/loss_avg": loss_meter.avg,
                    "train/grad_norm": norm_meter.val,
                    "train/grad_norm_avg": norm_meter.avg,
                    "train/lr": lr,
                    "train/memory_MB": memory_used,
                    "train/batch_time": batch_time.val,
                    "global_step": global_step,
                    "epoch": epoch + 1,
                })

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    
    # Log epoch summary to wandb
    if dist.get_rank() == 0:
        wandb.log({
            "train/epoch_loss": loss_meter.avg,
            "train/epoch_grad_norm": norm_meter.avg,
            "train/epoch_time_seconds": epoch_time,
            "epoch": epoch + 1,
        })


@torch.no_grad()
def validate(config, data_loader, model, logger, epoch=None, args=None):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{(idx + 1)}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for _, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    
    main()

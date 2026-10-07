# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

uv run python -m torch.distributed.launch \
    --nproc_per_node=2 \
    main.py \
    --cfg cfgs/vittt_b_no_query.yaml \
    --data-path $DATA_PATH \
    --output output/vittt_b_2gpu_bs512_60epoch_no_query \
    --batch-size 256 \
    --epochs 60 \
    --warmup-epochs 4 \
    --wandb-run-name vittt_b_2gpu_bs512_60epoch_no_query \
    --amp
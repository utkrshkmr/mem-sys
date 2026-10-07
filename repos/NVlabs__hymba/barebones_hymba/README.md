# Barebones Hymba

This is a minimal implementation of Hymba model. 

> NOTE: This version is only for demonstration purposes and does not include all the features of the full model. For full implementation, please refer to [`HF/Hymba-1.5B-Instruct`](https://huggingface.co/nvidia/Hymba-1.5B-Instruct/blob/main/modeling_hymba.py)


This minimal implementation includes:
1. Parallel fused attention and mamba heads 
2. Meta tokens and how to use them with sliding window attention 
3. Mix of local and global attention 

and does not include:
1. Cache management for generation during inference
2. Cross-layer kv reusing 

## Run Barebones Hymba

```bash
python test_barebones_hymba.py
```

Feel free to change the `HymbaConfig` to build your own Hymba model. 
```python
class HymbaConfig:
    num_layers = 32
    global_layer_list = [5,11,18,25,31]
    hidden_size = 1536
    vocab_size = 151936
    num_meta_tokens = 256
    mamba_expand = 2
    num_attention_heads = 12
    num_key_value_heads = 2
    conv_kernel_size = 3
    time_step_rank = 8
    ssm_state_size = 16
    attention_window_size = 2048
    mlp_hidden_act = "silu"
    intermediate_size = 4608
    modify_attention_mask = True # set this to False if you want to use flashattention for simplicity
    seq_length = 4096
    use_positional_embedding = True
    rope_base = 10000
```


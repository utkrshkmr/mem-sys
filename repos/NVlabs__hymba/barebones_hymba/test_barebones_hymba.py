import torch
from torch import nn
from transformers.activations import ACT2FN
from barebones_hymba_block import HymbaBlock, HymbaRMSNorm


class HymbaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.config = config
        self.act_fn_name = config.mlp_hidden_act
        self.act_fn = ACT2FN[self.act_fn_name]
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        if self.act_fn_name == "silu":
            self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)


    def forward(self, x):
        if self.act_fn_name == "silu":
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        elif self.act_fn_name == "relu2":
            return self.down_proj(self.act_fn(self.up_proj(x)))
        else:
            raise NotImplementedError(f"No such hidden_act: {self.act_fn_name}")
        


class SimplifiedHymbaDecoder(nn.Module):
    def __init__(self, config, is_global=False, modify_attention_mask=False):
        super().__init__()
        self.hymba_block = HymbaBlock(
            mamba_expand=config.mamba_expand,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            conv_kernel_size=config.conv_kernel_size,
            time_step_rank=config.time_step_rank,
            ssm_state_size=config.ssm_state_size,
            attention_window_size=None if is_global else config.attention_window_size,
            modify_attention_mask=modify_attention_mask,
            num_meta_tokens=config.num_meta_tokens,
            seq_length=config.seq_length,
            use_positional_embedding=config.use_positional_embedding,
            rope_base=config.rope_base,
        )
        self.mlp = HymbaMLP(config)

        self.input_layernorm = HymbaRMSNorm(config.hidden_size, eps=1e-6)

        self.pre_mlp_layernorm = HymbaRMSNorm(config.hidden_size, eps=1e-6)


    def forward(self, hidden_states):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.hymba_block(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


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




class SimplifiedHymba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = []
        for layer_idx in range(config.num_layers):
            self.layers.append(SimplifiedHymbaDecoder(config, is_global=layer_idx in config.global_layer_list, modify_attention_mask=config.modify_attention_mask and layer_idx not in config.global_layer_list))

        self.layers = nn.ModuleList(self.layers)
        self.meta_token = nn.Parameter(torch.randn(config.num_meta_tokens, config.hidden_size))


    def forward(self, hidden_states):
        # prepend meta tokens 
        hidden_states = torch.cat([self.meta_token.expand(hidden_states.size(0), -1, -1), hidden_states], dim=1)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)

        # remove meta tokens before returning
        hidden_states = hidden_states[:, config.num_meta_tokens:]
        return hidden_states


if __name__ == "__main__":
    config = HymbaConfig()
    model = SimplifiedHymba(config).to("cuda").to(torch.bfloat16)
    print(model)

    bsz = 1
    seq_len = 4096
    input_tensor = torch.randn(bsz, seq_len, config.hidden_size).to("cuda").to(torch.bfloat16)

    output = model(input_tensor)

    print(output.shape)
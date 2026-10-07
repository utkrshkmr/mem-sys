from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class RepeatLinear(nn.Module):
    """Linear layer that applies a learnable vector 'w' for feature-wise modulation of input.

    The main purpose of this module is to use a learnable vector 'w' to selectively modulate
    input features before applying a linear transformation. By repeating 'w' across the batch,
    the module can control which features are amplified or suppressed, enhancing the gating
    flexibility in the `MemoryModule`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        # Learnable vector 'w' to modulate input features before linear transformation.
        # This vector acts as a feature-wise gate, controlling how each input feature is weighted.
        self.w = nn.Parameter(torch.randn(in_dim).cuda())
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w.unsqueeze(0).repeat(x.size(0), 1, 1)

        # Apply element-wise modulation of 'x' using 'w', followed by ReLU and mean pooling.
        x = torch.relu(w * x)
        x = torch.mean(x, dim=1)
        return self.linear(x)


# Custom linear layer with grouped weights initialization
class GroupLinearLayer(nn.Module):
    def __init__(
        self,
        in_dim,  # Input dimension
        out_dim,  # Output dimension
        a=None,  # Scaling factor for weight and bias initialization
    ) -> None:
        super().__init__()  # Initialize the parent nn.Module
        if a is None:
            a = 1.0 / math.sqrt(
                out_dim
            )  # Default scaling factor based on output dimension
        self.linear = nn.Linear(in_dim, out_dim)  # Define a linear transformation layer
        self.linear.weight.data.uniform_(
            -a, a
        )  # Initialize weights uniformly in [-a, a]
        self.linear.bias.data.uniform_(-a, a)  # Initialize biases uniformly in [-a, a]

    def forward(self, x):
        x = self.linear(x)  # Apply the linear transformation
        return x  # Return the transformed input


class MemoryModule(nn.Module):
    """Relational memory core with multi-head attention and gating mechanisms.

    This module's primary role is to manage memory interactions through multi-head attention
    and gating. It selectively updates memory states by attending over inputs and applying
    gating to control the retention and forgetting of information.

    Components:
    - Multi-head Attention: Attends over memory using a multi-head attention mechanism.
    - Gating: Uses gates to modulate memory updates based on input and memory.
    """

    def __init__(
        self,
        mem_slots: int,  # Number of memory slots
        head_size: int,  # Size of each attention head
        hidden_dim: int,  # Dimension of the hidden state
        attn_drop: float = 0.9,  # Dropout rate for attention
        num_heads: int = 1,  # Number of attention heads
        num_blocks: int = 1,  # Number of attention blocks
        forget_bias: float = 1.0,  # Bias for forget gate
        input_bias: float = 0.0,  # Bias for input gate
        attention_mlp_layers: int = 2,  # Number of MLP layers in attention
        use_topk: bool = False,  # Whether to use top-k attention
        topk: int = 3,  # Number of top elements to keep if using top-k
    ) -> None:
        super().__init__()

        self.mem_slots = mem_slots
        self.head_size = head_size
        self.hidden_dim = hidden_dim
        self.n_heads = num_heads
        self.use_topk = use_topk
        self.topk = topk
        self.attn_drop = nn.Dropout(attn_drop)

        if num_blocks < 1:
            msg = f"num blocks must be >= 1. Got: {num_blocks}"
            raise ValueError(msg)
        self.num_blocks = num_blocks
        self.num_atten_mlp_layers = attention_mlp_layers

        self.query_proj = nn.Linear(self.hidden_dim, self.mem_slots)
        self.key_proj = nn.Linear(self.mem_slots, self.mem_slots)
        self.value_proj = nn.Linear(self.mem_slots, self.mem_slots)

        # Define MLP layers for processing attended memory
        self.attention_mlp = nn.ModuleList(
            [nn.Linear(self.mem_slots, self.mem_slots)] * self.num_atten_mlp_layers
        )
        self.attended_memory_layernorm = nn.LayerNorm(self.mem_slots)
        self.attended_memory_layernorm2 = nn.LayerNorm(self.mem_slots)

        # params for gating
        self.num_gates = 2 * self.calculate_gate_size()

        # Initialize input gate projector with RepeatLinear
        self.input_gate_projector = RepeatLinear(
            in_dim=self.mem_slots, out_dim=self.num_gates
        )
        # Initialize memory gate projector with GroupLinearLayer
        self.memory_gate_projector = GroupLinearLayer(
            in_dim=self.mem_slots, out_dim=self.num_gates
        )

        # Define bias parameters for forget and input gates
        self.forget_bias = nn.Parameter(torch.tensor(forget_bias, dtype=torch.float32))
        self.input_bias = nn.Parameter(torch.tensor(input_bias, dtype=torch.float32))

    def multi_head_attention(
        self,
        ipts: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply multi-head attention over memory and inputs, potentially using top-k masking.

        Multi-head attention enables the model to learn relationships across different memory slots
        and attend over them with context provided by the input.
        """
        b, t, c1 = ipts.size()
        _, m, c2 = memory.size()

        # Adjust the input sequence length to match memory length
        if t < m:
            # Upsample input using linear interpolation to match memory slots
            ipts = F.interpolate(ipts.transpose(1, 2), size=m, mode="linear").transpose(
                1, 2
            )
        elif t > m:
            # Downsample input using adaptive average pooling to match memory slots
            ipts = F.adaptive_avg_pool1d(ipts.transpose(1, 2), m).transpose(1, 2)

        """Perform multi-head attention"""
        q = self.query_proj(ipts)  # Project inputs to queries: Shape (B, T, hidden_dim)
        # Project memory to keys: Shape (B, M, mem_slots)
        k = self.key_proj(memory)
        # Project memory to values: Shape (B, M, mem_slots)
        v = self.value_proj(memory)

        # Reshape and transpose for multi-head attention
        q = q.reshape(b, m, self.n_heads, -1).transpose(
            1, 2
        )  # Shape: (B, n_heads, T, head_dim)
        k = k.reshape(k.size(0), k.size(1), self.n_heads, -1).transpose(
            1, 2
        )  # Shape: (B, n_heads, M, head_dim)
        v = v.reshape(v.size(0), v.size(1), self.n_heads, -1).transpose(
            1, 2
        )  # Shape: (B, n_heads, M, head_dim)

        # Compute scaled dot-product attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, M, M)

        if m != t:
            raise ValueError(f"Memory length M {m} must be equal sequence length T {t} for causal masking.")

        causal_mask = ~torch.tril(torch.ones((t, m), dtype=torch.bool, device=att.device))  # (T, M)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, M)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)  # (B, n_heads, T, M)
            combined_mask = causal_mask | attention_mask  # (B, n_heads, T, M)
        else:
            combined_mask = causal_mask

        attn_bias = combined_mask.to(dtype=torch.float32).masked_fill(combined_mask, float("-inf"))

        att = att + attn_bias
        att = F.softmax(att, dim=-1)  # (B, n_heads, T, M)
        att = self.attn_drop(att)

        if self.use_topk:
            # If top-k attention is enabled, retain only the top-k attention scores
            topk = torch.topk(att, dim=-1, k=self.topk)
            mask = torch.zeros_like(att).to(att.device)
            mask.scatter_(3, topk.indices, 1)
            att = att * mask

        output = att @ v  # (B, n_heads, T, head_dim)
        return output.transpose(1, 2).contiguous().view(b, t, self.n_heads * v.size(-1))

    def attend_over_memory(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attend over memory for each block, applying normalization and MLP processing.

        This function iterates over multiple attention blocks, allowing the memory state
        to be refined and updated progressively.
        """
        for _ in range(self.num_blocks):
            attended_memory = self.multi_head_attention(inputs, memory, attention_mask)
            memory = self.attended_memory_layernorm(memory + attended_memory)

            # Pass the normalized memory through MLP layers with ReLU activation
            attention_mlp = (
                # memory.clone()
                memory
            )  # Clone memory to avoid in-place modifications
            for i, _ in enumerate(self.attention_mlp):
                attention_mlp = self.attention_mlp[i](
                    attention_mlp
                )  # Apply each MLP layer
                attention_mlp = F.relu(attention_mlp)  # Apply ReLU activation
            # Add residual connection and apply second layer normalization
            memory = self.attended_memory_layernorm2(memory + attention_mlp)
        return memory  # Return the updated memory

    def forward(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process inputs and memory states, applying multi-head attention and gates.

        Each step performs:
        - Attention over memory to update states.
        - Gating to control memory retention or update based on inputs.
        - Returns the final inputs (with memory) and the updated memory state.
        """
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)

        # Attend over memory to obtain the next memory state
        next_memory = self.attend_over_memory(inputs, memory, attention_mask)

        # Create input and forget gates based on current inputs and memory
        input_gate, forget_gate = self.create_gates(inputs, memory)
        # Apply input gate with tanh activation
        next_memory = input_gate * torch.tanh(next_memory)
        # Apply forget gate to retain part of the previous memory
        next_memory += forget_gate * memory

        # Return the updated inputs and the next memory state
        return inputs + next_memory, next_memory

    def calculate_gate_size(self) -> int:
        """Determine gate size based on gating style (unit or memory-based)."""
        return self.mem_slots  # One gate per memory slot

    def create_gates(
        self, inputs: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create input and forget gates using inputs and memory.

        •   Purpose: Generates input and forget gates based on the current inputs and memory.
        •	Process:
            •	Memory Activation: Applies tanh activation to the memory to bound its values.
            •	Gate Projection: Projects inputs and memory into gate values using the respective projector layers.
            •	Gate Combination: Combines the projected inputs and memory to form the final gate values.
            •	Gate Splitting: Splits the combined gates into separate input and forget gates.
            •	Activation and Bias Application: Applies sigmoid activation with biases to constrain gate values.
            •	Logging: Initializes an attention log for debugging or analysis purposes.
            •	Gate Output: Returns the input and forget gates for use in modulating the memory.
        """
        memory = torch.tanh(memory)  # Apply tanh activation to memory for gating
        shape_dim = 3

        if len(inputs.shape) == shape_dim:
            # Project inputs to gate values using RepeatLinear
            gate_inputs = self.input_gate_projector(
                inputs
            )  # Shape: (Batch, Seq, num_gates)
            # Add a dimension: Shape (Batch, 1, Seq, num_gates)
            gate_inputs = gate_inputs.unsqueeze(1)

            # Project memory to gate values using GroupLinearLayer
            gate_memory = self.memory_gate_projector(
                memory
            )  # Shape: (Batch, mem_slots, num_gates)
        else:
            # Raise an error if input shape is not as expected (Batch, Seq, Features)
            msg = f"input shape of create_gate function is {inputs.shape}, expects 3"
            raise ValueError(msg)

        # Combine gate inputs and memory projections
        gates = gate_memory + gate_inputs  # Broadcasting addition

        # Split the combined gates into input and forget gates
        gates = torch.split(
            gates, split_size_or_sections=int(gates.shape[2] / 2), dim=2
        )  # Split along the gate dimension
        input_gate, forget_gate = gates  # Unpack the split gates

        # Ensure input and forget gates have the same number of features
        if input_gate.shape[2] != forget_gate.shape[2]:
            raise ValueError

        # Apply sigmoid activation with biases to gates to constrain them between 0 and 1
        input_gate = torch.sigmoid(
            input_gate + self.input_bias
        )  # Shape: (Batch, 1, Seq, num_gates/2)
        forget_gate = torch.sigmoid(
            forget_gate + self.forget_bias
        )  # Shape: (Batch, 1, Seq, num_gates/2)

        return input_gate, forget_gate  # Return the input and forget gates

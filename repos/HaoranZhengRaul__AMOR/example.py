"""
Minimal AMOR usage example: instantiate, forward, generate.

Run with:
    python example.py
"""

import torch

from amor import AMOR
from amor_decode import AMORDecodeWrapper


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tiny config so the example runs on a laptop. For the published variants,
    # use d_model=768, n_layer=12, d_ff=1216 (180M scale) and matching depths.
    model = AMOR(
        backbone='mamba2',          # 'mamba2' or 'gdn'
        n_amor_blocks=3,            # 1 or 3
        residual_mode='classic',    # 'classic' or 'h'
        vocab_size=1024,
        d_model=128,
        n_layer=2,
        d_ff=256,
        n_heads=4,
        head_dim=32,
        max_seq_len=512,
    ).to(device).eval()

    # Forward pass on dummy tokens.
    input_ids = torch.randint(0, 1024, (1, 32), device=device)
    with torch.no_grad():
        logits = model(input_ids)
        print(f"forward: input {tuple(input_ids.shape)} -> logits {tuple(logits.shape)}")

        logits, details = model(input_ids, return_details=True)
        print(f"  fire_rate={details['fire_rate']:.3f} alpha={details['alpha']:.3f}")

    # Autoregressive generation with caches (prefill + decode steps).
    wrapper = AMORDecodeWrapper(model)
    prompt = torch.randint(0, 1024, (1, 16), device=device)
    with torch.no_grad():
        logits, cache = wrapper.prefill(prompt)
        print(f"prefill:  prompt {tuple(prompt.shape)} -> logits {tuple(logits.shape)}")

        next_token = logits[:, -1].argmax(-1, keepdim=True)
        for step in range(8):
            step_idx = prompt.shape[1] + step
            logits, cache, fire_info = wrapper.decode_step(next_token, cache, step_idx)
            fired = sum(1 for b in fire_info if b['fired'])
            print(f"  step {step:2d}  fired {fired}/{len(fire_info)} blocks  next_token={next_token.item()}")
            next_token = logits[:, -1].argmax(-1, keepdim=True)


if __name__ == '__main__':
    main()

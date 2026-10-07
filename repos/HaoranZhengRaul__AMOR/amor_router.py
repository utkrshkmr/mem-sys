"""Fitted entropy routers for AMOR inference."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from weight_io import file_sha256


class EntropyRouter(nn.Module):
    """Pointwise fp32 entropy estimate: width-512 SiLU MLP plus linear skip."""

    def __init__(self, d_in, r=512, act="silu", chunk=4096):
        super().__init__()
        if act != "silu":
            raise ValueError("This release supports the fitted SiLU router.")
        self.d_in, self.r, self.act, self.chunk = int(d_in), int(r), act, int(chunk)
        self.fc1 = nn.Linear(self.d_in, self.r, dtype=torch.float32)
        self.fc2 = nn.Linear(self.r, 1, dtype=torch.float32)
        self.skip = nn.Linear(self.d_in, 1, bias=False, dtype=torch.float32)
        self.register_buffer("fit_meta", torch.tensor([self.r, 3, -1, -1], dtype=torch.int64))

    def _apply(self, fn, *args, **kwargs):
        # Preserve fp32 values, including when the surrounding model is cast.
        def apply_fp32(tensor):
            result = fn(tensor)
            if tensor.is_floating_point() and result.dtype != torch.float32:
                return tensor.to(device=result.device, dtype=torch.float32)
            return result
        return super()._apply(apply_fp32, *args, **kwargs)

    def _core(self, features):
        return (self.fc2(F.silu(self.fc1(features))) + self.skip(features)).squeeze(-1)

    def forward(self, features):
        if features.shape[-1] != self.d_in:
            raise ValueError(f"Router expects {self.d_in} input channels.")
        with torch.autocast(device_type=features.device.type, enabled=False):
            flat = features.reshape(-1, self.d_in)
            if self.chunk and flat.shape[0] > self.chunk:
                output = torch.cat([
                    self._core(flat[start:start + self.chunk].float())
                    for start in range(0, flat.shape[0], self.chunk)
                ])
            else:
                output = self._core(flat.float())
        return output.reshape(features.shape[:-1])


def detach_routers(model):
    """Restore the native entropy gate before training or entropy-gate inference."""
    for block in model.amor_blocks:
        block.gate.router = None


def attach_routers(model, directory):
    """Attach the matching fitted routers after strict base-weight loading."""
    if model.training:
        raise ValueError("Call model.eval() before attaching inference routers.")
    directory = Path(directory)
    config = json.loads((directory / "router_config.json").read_text())
    if config.get("format_version") != 1 or config.get("variant") != "distill":
        raise ValueError("Unsupported router configuration.")
    if getattr(model, "_release_weight_sha256", None) != config["base_weight_sha256"]:
        raise ValueError("Router/base weights mismatch; load the base with load_model().")
    expected = {
        "backbone": model.backbone, "residual_mode": model.residual_mode,
        "d_model": model.d_model, "n_blocks": len(model.amor_blocks),
    }
    if config["model"] != expected:
        raise ValueError("Router architecture does not match this model.")
    weights = directory / "router.safetensors"
    if file_sha256(weights) != config["router_sha256"]:
        raise ValueError("Router weights checksum mismatch.")
    states = load_file(str(weights), device="cpu")
    routers, consumed = [], set()
    device = next(model.parameters()).device
    # Build and validate the entire set before modifying any gate.
    with torch.random.fork_rng(devices=[]):
        for i, block in enumerate(model.amor_blocks):
            if getattr(block.gate, "router", None) is not None:
                raise ValueError("Routers are already attached; detach them first.")
            router = EntropyRouter(model.d_model, r=config["hidden_dim"], act=config["activation"])
            prefix = f"blocks.{i}."
            state = {key[len(prefix):]: value for key, value in states.items()
                     if key.startswith(prefix)}
            consumed.update(key for key in states if key.startswith(prefix))
            router.load_state_dict(state, strict=True)
            if router.fit_meta[:3].tolist() != [config["hidden_dim"], 0, i]:
                raise ValueError(f"Router {i} fitting metadata does not match.")
            if not all(torch.isfinite(p).all() for p in router.parameters()):
                raise ValueError(f"Router {i} contains non-finite parameters.")
            routers.append(router.to(device).eval().requires_grad_(False))
    if consumed != set(states):
        raise ValueError("Unexpected tensors in router weights.")
    for block, router in zip(model.amor_blocks, routers):
        block.gate.router = router
    return model

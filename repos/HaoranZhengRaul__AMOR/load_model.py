"""Strict AMOR loading with optional fitted inference routers."""
import json
from pathlib import Path

import torch

from amor import AMOR
from amor_router import attach_routers
from weight_io import load_weight_state


def load_model(directory=None, device="cpu", dtype=torch.float32, use_router=False):
    directory = Path(directory) if directory else Path(__file__).resolve().parent
    config = json.loads((directory / "config.json").read_text())
    state, weight_hash = load_weight_state(directory)
    # Load fp32 values before applying the caller's explicit inference dtype.
    model = AMOR(**config["model_kwargs"]).float()
    if "lm_head.weight" in state and not torch.equal(state["lm_head.weight"], state["embed.weight"]):
        raise ValueError("The supplied LM head differs from the tied embedding.")
    state["lm_head.weight"] = state["embed.weight"]
    model.load_state_dict(state, strict=True)
    if model.lm_head.weight is not model.embed.weight:
        raise RuntimeError("Embedding/head weight tying was lost.")
    model = model.to(device=device, dtype=dtype).eval()
    model._release_weight_sha256 = weight_hash
    if use_router:
        attach_routers(model, directory)
    return model

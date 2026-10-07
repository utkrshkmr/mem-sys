"""Strict model loading preserves fp32 checkpoint values."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from amor import AMOR
from load_model import load_model
from weight_io import file_sha256


class LoaderTests(unittest.TestCase):
    def test_lower_global_default_does_not_round_loaded_weights(self):
        kwargs = dict(backbone="mamba2", n_amor_blocks=3, residual_mode="classic",
                      vocab_size=256, d_model=64, n_layer=1, d_ff=128,
                      n_heads=1, head_dim=32, d_state=16, d_conv=4, expand=2,
                      n_groups=1, max_seq_len=64)
        reference = AMOR(**kwargs).float().eval()
        state = {key: value.clone() for key, value in reference.state_dict().items()
                 if key != "lm_head.weight"}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); weights = root / "model.safetensors"
            save_file(state, str(weights))
            (root / "config.json").write_text(json.dumps({"model_kwargs": kwargs}))
            (root / "manifest.json").write_text(json.dumps({
                weights.name: {"bytes": weights.stat().st_size,
                               "sha256": file_sha256(weights)}}))
            previous = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                loaded = load_model(root)
            finally:
                torch.set_default_dtype(previous)
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

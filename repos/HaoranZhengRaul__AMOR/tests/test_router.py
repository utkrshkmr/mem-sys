"""Router attachment and numerical-contract tests; run with unittest."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file

from amor_router import EntropyRouter, attach_routers, detach_routers


class ModelStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.residual_mode, self.d_model = "mamba2", "classic", 8
        self.anchor = nn.Parameter(torch.zeros(1))
        self.amor_blocks = nn.ModuleList([nn.Module() for _ in range(3)])
        for block in self.amor_blocks:
            block.gate = nn.Module()
            block.gate.router = None
        self._release_weight_sha256 = "matching-base"


class RouterTests(unittest.TestCase):
    def package(self, directory):
        tensors = {}
        for i in range(3):
            router = EntropyRouter(8, r=4)
            router.fit_meta.copy_(torch.tensor([4, 0, i, 42]))
            for key, value in router.state_dict().items():
                tensors[f"blocks.{i}.{key}"] = value.clone()
        save_file(tensors, str(directory / "router.safetensors"))
        cfg = {"format_version": 1, "variant": "distill", "hidden_dim": 4,
               "activation": "silu", "base_weight_sha256": "matching-base",
               "model": {"backbone": "mamba2", "residual_mode": "classic",
                         "d_model": 8, "n_blocks": 3},
               "router_sha256": hashlib.sha256((directory / "router.safetensors").read_bytes()).hexdigest()}
        (directory / "router_config.json").write_text(json.dumps(cfg))
        return cfg, tensors

    def test_fp32_values_survive_cast_and_autocast(self):
        router = EntropyRouter(8, r=4).eval()
        original = {k: v.clone() for k, v in router.state_dict().items()}
        router.to(dtype=torch.bfloat16)
        for key, value in router.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        x = torch.randn(2, 9, 8)
        expected = router(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = router(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_initialization_ignores_a_lower_global_default_dtype(self):
        reference = EntropyRouter(8, r=4).state_dict()
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            router = EntropyRouter(8, r=4)
            router.load_state_dict(reference, strict=True)
            for name, value in router.state_dict().items():
                torch.testing.assert_close(value, reference[name], rtol=0, atol=0)
        finally:
            torch.set_default_dtype(previous)

    def test_chunking_and_batch_composition_preserve_predictions(self):
        router = EntropyRouter(8, r=4).eval()
        x = torch.randn(2, 4097, 8)
        with torch.no_grad():
            chunked = router(x)
            router.chunk = 0
            full = router(x)
            separate = torch.stack([router(sequence) for sequence in x])
        torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(full, separate, rtol=1e-5, atol=1e-6)

    def test_attach_is_rng_neutral_and_detach_restores_keys(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); self.package(folder)
            model = ModelStub().eval(); keys = set(model.state_dict())
            rng = torch.get_rng_state().clone()
            attach_routers(model, folder)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(not p.requires_grad for b in model.amor_blocks
                                for p in b.gate.router.parameters()))
            detach_routers(model)
            self.assertEqual(keys, set(model.state_dict()))

    def test_rejects_training_and_wrong_base(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); self.package(folder)
            model = ModelStub()
            with self.assertRaises(ValueError): attach_routers(model, folder)
            model.eval(); model._release_weight_sha256 = "wrong-base"
            with self.assertRaises(ValueError): attach_routers(model, folder)
            self.assertTrue(all(b.gate.router is None for b in model.amor_blocks))

    def test_late_invalid_block_cannot_partially_attach(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); cfg, tensors = self.package(folder)
            tensors["blocks.2.fit_meta"][1] = 1
            save_file(tensors, str(folder / "router.safetensors"))
            cfg["router_sha256"] = hashlib.sha256((folder / "router.safetensors").read_bytes()).hexdigest()
            (folder / "router_config.json").write_text(json.dumps(cfg))
            model = ModelStub().eval()
            with self.assertRaises(ValueError): attach_routers(model, folder)
            self.assertTrue(all(b.gate.router is None for b in model.amor_blocks))


if __name__ == "__main__":
    unittest.main()

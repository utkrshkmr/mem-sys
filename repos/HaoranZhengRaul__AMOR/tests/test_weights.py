"""Weight integrity and standard sharded-loader tests."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.shard_weights import shard_file
from weight_io import file_sha256, load_weight_state, weight_identity


class WeightTests(unittest.TestCase):
    def source(self, folder):
        tensors = {"a": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                   "b": torch.tensor([1.25, -2.5, 0.0])}
        file = folder / "model.safetensors"
        save_file(tensors, str(file))
        return tensors, file

    def test_single_file_load(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); tensors, source = self.source(root)
            manifest = {source.name: {"bytes": source.stat().st_size, "sha256": file_sha256(source)}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            loaded, identity = load_weight_state(root)
            self.assertEqual(identity, file_sha256(source))
            for key in tensors: torch.testing.assert_close(loaded[key], tensors[key], rtol=0, atol=0)

    def test_sharding_preserves_values_and_binds_all_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); tensors, source = self.source(root)
            out = root / "shards"; report = shard_file(source, out, target_bytes=16)
            self.assertTrue(report["all_tensor_bytes_identical"])
            (out / "manifest.json").write_text(json.dumps(report["files"]))
            loaded, identity = load_weight_state(out)
            self.assertEqual(identity, weight_identity(report["files"]))
            self.assertEqual(set(loaded), set(tensors))
            for key in tensors: torch.testing.assert_close(loaded[key], tensors[key], rtol=0, atol=0)

    def test_bad_index_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); _, source = self.source(root)
            out = root / "shards"; report = shard_file(source, out, target_bytes=16)
            index_path = out / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            values = list(index["weight_map"].values())
            index["weight_map"]["a"], index["weight_map"]["b"] = values[1], values[0]
            index_path.write_text(json.dumps(index))
            report["files"][index_path.name] = {"bytes": index_path.stat().st_size,
                                               "sha256": file_sha256(index_path)}
            (out / "manifest.json").write_text(json.dumps(report["files"]))
            with self.assertRaises(ValueError): load_weight_state(out)

    def test_corrupt_shard_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); _, source = self.source(root)
            out = root / "shards"; report = shard_file(source, out, target_bytes=16)
            (out / "manifest.json").write_text(json.dumps(report["files"]))
            shard = next(out.glob("*.safetensors"))
            with shard.open("ab") as stream: stream.write(b"corrupt")
            with self.assertRaises(ValueError): load_weight_state(out)


if __name__ == "__main__":
    unittest.main()

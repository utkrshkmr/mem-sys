"""Verified single-file and standard sharded safetensors loading."""
import hashlib
import json
import re
from pathlib import Path

INDEX = "model.safetensors.index.json"
SHARD = re.compile(r"model-[0-9]+-of-[0-9]+[.]safetensors")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_files(manifest):
    if INDEX in manifest:
        shards = sorted(name for name in manifest if SHARD.fullmatch(name))
        if not shards:
            raise ValueError("The weight index has no shards in the manifest.")
        return [INDEX] + shards
    if "model.safetensors" not in manifest:
        raise ValueError("The package has no model weights.")
    return ["model.safetensors"]


def weight_identity(manifest):
    names = weight_files(manifest)
    for name in names:
        if not re.fullmatch(r"[a-f0-9]{64}", manifest[name]["sha256"]):
            raise ValueError("Invalid weight checksum metadata.")
    if names == ["model.safetensors"]:
        return manifest[names[0]]["sha256"]
    hashes = {name: manifest[name]["sha256"] for name in names}
    payload = json.dumps({"format": "safetensors-sharded-v1", "files": hashes},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_weight_state(directory):
    from safetensors.torch import load_file

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    names = weight_files(manifest)
    # Verify every file before parsing tensors or allocating the model.
    for name in names:
        path = directory / name
        record = manifest[name]
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError("Model weights checksum mismatch: " + name)
    if names == ["model.safetensors"]:
        return load_file(str(directory / names[0]), device="cpu"), weight_identity(manifest)
    index = json.loads((directory / INDEX).read_text())
    mapping = index["weight_map"]
    if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                for k, v in mapping.items()):
        raise ValueError("Invalid weight index.")
    if set(mapping.values()) != set(names[1:]):
        raise ValueError("Index and manifest list different shards.")
    state = {}
    for name in names[1:]:
        tensors = load_file(str(directory / name), device="cpu")
        expected = {key for key, filename in mapping.items() if filename == name}
        if set(tensors) != expected:
            raise ValueError("Shard tensors do not match the index: " + name)
        state.update(tensors)
    return state, weight_identity(manifest)

"""Split safetensors at tensor boundaries and verify every tensor's bytes."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weight_io import file_sha256


def shard_file(source, output, target_bytes=900_000_000):
    source, output = Path(source), Path(output)
    if target_bytes <= 0:
        raise ValueError("Shard target must be positive.")
    output.mkdir(parents=True, exist_ok=False)
    with source.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size > 100 * 1024 * 1024:
            raise ValueError("Unexpectedly large safetensors header.")
        header = json.loads(stream.read(header_size))
    data_start = 8 + header_size
    entries = sorted(((key, value) for key, value in header.items() if key != "__metadata__"),
                     key=lambda item: item[1]["data_offsets"][0])
    if not entries:
        raise ValueError("The source contains no tensors.")
    groups, current, current_size, previous_end = [], [], 0, 0
    for name, descriptor in entries:
        start, end = descriptor["data_offsets"]
        if start != previous_end or end < start:
            raise ValueError("Invalid or non-contiguous source tensor offsets.")
        previous_end = end
        size = end - start
        if current and current_size + size > target_bytes:
            groups.append(current); current, current_size = [], 0
        current.append((name, descriptor)); current_size += size
        if size > target_bytes:
            groups.append(current); current, current_size = [], 0
    if current:
        groups.append(current)
    if data_start + previous_end != source.stat().st_size:
        raise ValueError("Source data size differs from its header.")
    mapping, files = {}, {}
    for number, group in enumerate(groups, 1):
        filename = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
        new_header = {"__metadata__": header.get("__metadata__", {})}
        offset = 0
        for name, descriptor in group:
            size = descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
            new_header[name] = {"dtype": descriptor["dtype"], "shape": descriptor["shape"],
                                "data_offsets": [offset, offset + size]}
            offset += size; mapping[name] = filename
        raw = json.dumps(new_header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        originals = {}
        destination = output / filename
        with destination.open("xb") as target, source.open("rb") as stream:
            target.write(struct.pack("<Q", len(raw))); target.write(raw)
            for name, descriptor in group:
                start, end = descriptor["data_offsets"]; stream.seek(data_start + start)
                remaining, digest = end - start, hashlib.sha256()
                while remaining:
                    chunk = stream.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Source ended during tensor copy.")
                    target.write(chunk); digest.update(chunk); remaining -= len(chunk)
                originals[name] = digest.hexdigest()
        with destination.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            exported = json.loads(stream.read(length))
            for name, descriptor in group:
                if exported[name]["shape"] != descriptor["shape"] or exported[name]["dtype"] != descriptor["dtype"]:
                    raise ValueError("Export changed tensor metadata.")
                start, end = exported[name]["data_offsets"]; stream.seek(8 + length + start)
                remaining, digest = end - start, hashlib.sha256()
                while remaining:
                    chunk = stream.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Export ended during tensor verification.")
                    digest.update(chunk); remaining -= len(chunk)
                if digest.hexdigest() != originals[name]:
                    raise ValueError("Export changed tensor bytes.")
        files[filename] = {"bytes": destination.stat().st_size, "sha256": file_sha256(destination)}
    index = output / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": previous_end},
                                "weight_map": mapping}, indent=2) + "\n")
    files[index.name] = {"bytes": index.stat().st_size, "sha256": file_sha256(index)}
    return {"tensor_count": len(entries), "all_tensor_bytes_identical": True,
            "files": files, "max_shard_bytes": max(record["bytes"] for name, record in files.items()
                                                    if name.endswith(".safetensors"))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-bytes", type=int, default=900_000_000)
    args = parser.parse_args()
    print(json.dumps(shard_file(args.source, args.output, args.target_bytes), indent=2))

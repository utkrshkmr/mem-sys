"""Download one anonymous AMOR package as individually verified files."""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class SameHostRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        target = urlsplit(newurl)
        if (target.scheme != "https" or target.hostname != "anonymous-hf.com"
                or target.username or target.password or target.port not in (None, 443)):
            raise ValueError("The anonymous download tried to leave the proxy host.")
        return super().redirect_request(request, fp, code, message, headers, newurl)


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or any(ord(char) < 32 for char in name)):
        raise ValueError("Unsafe package filename.")
    return Path(*path.parts)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Anonymous model viewer URL")
    parser.add_argument("--output", type=Path, default=Path("model"))
    args = parser.parse_args()
    parsed = urlsplit(args.url)
    match = re.fullmatch(r"/a/([A-Za-z0-9_-]+)/?", parsed.path)
    if parsed.scheme != "https" or parsed.hostname != "anonymous-hf.com" or not match:
        parser.error("Use the https://anonymous-hf.com/a/.../ viewer URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("Use the plain anonymous viewer URL.")
    prefix = "https://anonymous-hf.com/api/a/" + match.group(1)
    opener = build_opener(SameHostRedirect())
    opener.addheaders = [("User-Agent", "AMOR-release-downloader/1.0")]

    def get_json(url):
        with opener.open(Request(url), timeout=60) as response:
            data = response.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("Unexpectedly large package metadata.")
        return json.loads(data)

    info = get_json(prefix + "/info/")
    if info.get("status") != "active" or info.get("identity_revealed"):
        raise ValueError("This anonymous link is not active. Request a current link.")
    manifest = get_json(prefix + "/resolve/manifest.json")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for name, record in manifest.items():
        relative = safe_relative(name)
        target = (output / relative).resolve()
        if not target.is_relative_to(output):
            raise ValueError("A package filename escapes the output folder.")
        size = record["bytes"]
        if not isinstance(size, int) or size < 0 or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"]):
            raise ValueError("Invalid file checksum metadata.")
        if target.exists():
            if target.is_file() and target.stat().st_size == size and digest(target) == record["sha256"]:
                print("Verified existing", name, flush=True)
                continue
            raise FileExistsError(f"{target} exists but differs from this package.")
        files.append((name, target, record))
    if shutil.disk_usage(output).free < sum(row[2]["bytes"] for row in files) + 256 * 1024 * 1024:
        raise OSError("Insufficient free space for this model package.")
    for name, target, record in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        value, size = hashlib.sha256(), 0
        print("Downloading", name, flush=True)
        created = False
        try:
            with partial.open("xb") as stream:
                created = True
                with opener.open(Request(prefix + "/resolve/" + quote(name, safe="/")), timeout=60) as response:
                    while chunk := response.read(4 * 1024 * 1024):
                        size += len(chunk)
                        if size > record["bytes"]:
                            raise ValueError("Response exceeds the expected file size.")
                        value.update(chunk);stream.write(chunk)
            if size != record["bytes"] or value.hexdigest() != record["sha256"]:
                raise ValueError("Downloaded file failed size/checksum verification: " + name)
            if target.exists():
                raise FileExistsError("Output appeared during download: " + str(target))
            os.replace(partial, target)
        finally:
            if created and partial.exists():
                partial.unlink()
        print("Verified", name, flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Verified model package:", output)


if __name__ == "__main__":
    main()

"""Safety checks for the anonymous release downloader."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request

import download_release as download


class DownloaderTests(unittest.TestCase):
    def test_rejects_unsafe_paths(self):
        for name in ("", "../file", "/absolute", "a/../../file", "a\\b", "a\nfile"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                download.safe_relative(name)

    def test_preserves_normal_relative_paths(self):
        self.assertEqual(download.safe_relative("assets/architecture.png"),
                         Path("assets/architecture.png"))

    def test_rejects_external_or_insecure_redirects(self):
        handler = download.SameHostRedirect()
        request = Request("https://anonymous-hf.com/api/a/example/resolve/file")
        for url in ("https://huggingface.co/file", "http://anonymous-hf.com/file",
                    "https://anonymous-hf.com:8443/file", "https://user@anonymous-hf.com/file"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                handler.redirect_request(request, None, 302, "Found", {}, url)

    def run_download(self, folder, corrupt=False):
        body = b"verified example"
        manifest = {"example.txt": {"bytes": len(body),
                                   "sha256": hashlib.sha256(body).hexdigest()}}
        class Opener:
            def open(self, request, timeout=60):
                url = request.full_url
                if url.endswith("/info/"):
                    return io.BytesIO(b'{"status":"active"}')
                if url.endswith("/manifest.json"):
                    return io.BytesIO(json.dumps(manifest).encode())
                return io.BytesIO(body + b"x" if corrupt else body)
        argv = ["download_release.py", "https://anonymous-hf.com/a/example/",
                "--output", str(folder)]
        with patch.object(sys, "argv", argv), patch.object(download, "build_opener", return_value=Opener()):
            with contextlib.redirect_stdout(io.StringIO()):
                download.main()

    def test_streamed_file_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.run_download(root)
            self.assertEqual((root / "example.txt").read_bytes(), b"verified example")
            self.assertFalse((root / "example.txt.partial").exists())
            self.run_download(root)

    def test_corrupt_download_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError): self.run_download(root, corrupt=True)
            self.assertFalse((root / "example.txt").exists())
            self.assertFalse((root / "example.txt.partial").exists())

    def test_existing_different_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);(root / "example.txt").write_bytes(b"original")
            with self.assertRaises(FileExistsError):self.run_download(root)
            self.assertEqual((root / "example.txt").read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()

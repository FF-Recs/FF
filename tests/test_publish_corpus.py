import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_corpus", ROOT / "scripts/publish-corpus.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublicationBoundaries(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".workdir/corpus-release"
        scratch.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="test-", dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def release(self, *, payload=None, change=None, duplicate=False):
        payloads = {name: (name + "\n").encode() for name in publisher.PAYLOADS}
        if payload is not None:
            payloads["features.safetensors"] = payload
        manifest = {
            "format_version": 5,
            "feature_model": "binary_presence_and_signed_log_popularity",
            "files": {name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                      for name, value in payloads.items()},
        }
        if change is not None:
            change(manifest)
        raw = json.dumps(manifest, separators=(",", ":")).encode()
        if duplicate:
            raw = raw.replace(b'"format_version":5', b'"format_version":5,"format_version":5')
        path = self.root / hashlib.sha256(raw).hexdigest()
        path.mkdir()
        (path / "manifest.json").write_bytes(raw)
        for name, value in payloads.items():
            (path / name).write_bytes(value)
        return path

    def validate(self, path):
        with contextlib.ExitStack() as stack:
            return publisher.validate_release(path, stack)

    def test_upload_stream_retains_verified_inode_after_path_replacement(self):
        path = self.release()
        name = "features.safetensors"
        expected = (path / name).read_bytes()
        with contextlib.ExitStack() as stack:
            _, streams, _ = publisher.validate_release(path, stack)
            (path / name).unlink()
            (path / name).write_bytes(b"unverified replacement")
            self.assertEqual(streams[name].read(), expected)

    def test_same_size_payload_corruption_is_rejected(self):
        path = self.release()
        target = path / "features.safetensors"
        value = target.read_bytes()
        target.write_bytes(bytes([value[0] ^ 1]) + value[1:])
        with self.assertRaises(publisher.PublicationError):
            self.validate(path)

    def test_manifest_directory_identity_binds_exact_bytes(self):
        path = self.release()
        with (path / "manifest.json").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaises(publisher.PublicationError):
            self.validate(path)

    def test_extra_hidden_file_cannot_become_a_release_asset(self):
        path = self.release()
        (path / ".envrc").write_text("synthetic private configuration")
        with self.assertRaises(publisher.PublicationError):
            self.validate(path)

    def test_allowed_name_cannot_symlink_to_an_external_file(self):
        path = self.release()
        target = path / "aliases.parquet"
        external = self.root / "external"
        external.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(external)
        with self.assertRaises((publisher.PublicationError, OSError)):
            self.validate(path)

    def test_duplicate_manifest_keys_are_rejected_even_when_values_match(self):
        path = self.release(duplicate=True)
        with self.assertRaises(publisher.PublicationError):
            self.validate(path)

    def test_asset_size_bound_is_strict_before_transport(self):
        limit = 16 * 1024
        below = self.release(payload=b"a" * (limit - 1))
        boundary = self.release(payload=b"a" * limit)
        with patch.object(publisher, "ASSET_LIMIT", limit):
            with contextlib.ExitStack() as stack:
                _, streams, sizes = publisher.validate_release(below, stack)
                self.assertEqual(sizes["features.safetensors"], limit - 1)
                self.assertEqual(hashlib.sha256(streams["features.safetensors"].read()).hexdigest(), hashlib.sha256(b"a" * (limit - 1)).hexdigest())
            with self.assertRaises(publisher.PublicationError):
                self.validate(boundary)

    def test_declared_size_is_not_coerced_from_boolean(self):
        def change(manifest):
            manifest["files"]["features.safetensors"]["bytes"] = True
        path = self.release(payload=b"a", change=change)
        with self.assertRaises(publisher.PublicationError):
            self.validate(path)

    def test_supported_format_is_required_before_transport(self):
        def change(manifest):
            manifest["format_version"] = 6
        path = self.release(change=change)
        with self.assertRaises(publisher.PublicationError):
            self.validate(path)


if __name__ == "__main__":
    unittest.main()

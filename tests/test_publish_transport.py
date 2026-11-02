import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import io
import json
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_corpus_transport", ROOT / "scripts/publish-corpus.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)

TOKEN = "synthetic-transport-test-token-not-a-credential"


class StrictUploadHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, format, *arguments):
        pass

    def read_chunked_json(self):
        body = bytearray()
        while True:
            line = self.rfile.readline(4097)
            if len(line) > 4096 or not line.endswith(b"\r\n"):
                raise ValueError("Invalid chunk header")
            size = int(line[:-2].split(b";", 1)[0], 16)
            if size < 0 or len(body) + size > 65536:
                raise ValueError("Chunked body exceeds test server bound")
            if size == 0:
                trailer_bytes = 0
                while True:
                    trailer = self.rfile.readline(4097)
                    trailer_bytes += len(trailer)
                    if trailer_bytes > 8192 or not trailer.endswith(b"\r\n"):
                        raise ValueError("Invalid chunk trailers")
                    if trailer == b"\r\n":
                        return bytes(body)
                    if b":" not in trailer:
                        raise ValueError("Invalid chunk trailer")
            chunk = self.rfile.read(size)
            if len(chunk) != size or self.rfile.read(2) != b"\r\n":
                raise ValueError("Truncated chunk")
            body.extend(chunk)

    def do_POST(self):
        lengths = self.headers.get_all("Content-Length", [])
        transfer_encoding = self.headers.get("Transfer-Encoding")
        status = self.server.response_status
        response = self.server.response_body
        body = b""
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip()
        if transfer_encoding:
            if content_type != "application/json" or lengths or transfer_encoding.lower() != "chunked":
                status, response = 411, {"message": "A declared Content-Length without chunking is required"}
            else:
                try:
                    body = self.read_chunked_json()
                except (ValueError, TimeoutError, OSError):
                    status, response = 400, {"message": "Invalid chunked JSON request"}
        elif len(lengths) != 1:
            status, response = 411, {"message": "A declared Content-Length is required"}
        else:
            try:
                length = int(lengths[0])
                if not 0 <= length <= 65536:
                    raise ValueError("Body exceeds test server bound")
            except ValueError:
                status, response = 400, {"message": "Invalid Content-Length"}
            else:
                try:
                    body = self.rfile.read(length)
                except (TimeoutError, OSError):
                    status, response = 400, {"message": "Declared body did not arrive"}
                else:
                    if len(body) != length:
                        status, response = 400, {"message": "Truncated request body"}
        # Deliberately retain only protocol fields, never Authorization headers.
        self.server.observations.put({
            "method": self.command,
            "lengths": lengths,
            "transfer_encoding": transfer_encoding,
            "content_type": self.headers.get("Content-Type"),
            "body": body,
            "status": status,
        })
        encoded = json.dumps(response).encode()
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    do_PATCH = do_POST


class PublicationTransport(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            shutil.which("gh"),
            "The installed GitHub CLI (gh) is required for publisher transport coverage.",
        )
        scratch = ROOT / ".workdir/corpus-release"
        scratch.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="test-", dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        config = self.root / "gh-config"
        config.mkdir()
        self.client = publisher.GitHub(str(config))
        for name in list(self.client.environment):
            if name.lower().endswith("_proxy") or name == "GH_HTTP_UNIX_SOCKET":
                self.client.environment.pop(name)
        self.client.environment.update(GH_TOKEN=TOKEN, NO_PROXY="127.0.0.1")
        self.server = HTTPServer(("127.0.0.1", 0), StrictUploadHandler)
        self.server.observations = queue.Queue()
        self.server.response_status = 200
        self.server.response_body = {"ok": True}
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="publisher-loopback-server",
            daemon=True,
        )
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/assets"

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.assertFalse(self.thread.is_alive(), "Loopback server did not terminate")

    def received(self, expected, content_type):
        observation = self.server.observations.get(timeout=5)
        self.assertEqual(observation["status"], self.server.response_status)
        if content_type == "application/json" and observation["transfer_encoding"]:
            self.assertEqual(observation["lengths"], [])
            self.assertEqual(observation["transfer_encoding"].lower(), "chunked")
        else:
            self.assertEqual(observation["lengths"], [str(len(expected))])
            self.assertIsNone(observation["transfer_encoding"])
        self.assertEqual((observation["content_type"] or "").split(";", 1)[0].strip(), content_type)
        self.assertEqual(observation["body"], expected)
        self.assertTrue(self.server.observations.empty(), "Unexpected extra HTTP request")
        return observation

    def release(self, payload):
        payloads = {name: (name + "\n").encode() for name in publisher.PAYLOADS}
        payloads["features.safetensors"] = payload
        manifest = {
            "format_version": 5,
            "feature_model": "binary_presence_and_signed_log_popularity",
            "files": {
                name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                for name, value in payloads.items()
            },
        }
        raw = json.dumps(manifest, separators=(",", ":")).encode()
        path = self.root / hashlib.sha256(raw).hexdigest()
        path.mkdir()
        (path / "manifest.json").write_bytes(raw)
        for name, value in payloads.items():
            (path / name).write_bytes(value)
        return path

    def test_binary_upload_has_declared_length_and_exact_bytes(self):
        payload = bytes(range(256)) * 4 + b"\x00\xff\r\nnot-json"
        path = self.root / "binary-upload"
        path.write_bytes(payload)
        with path.open("rb") as stream:
            result = self.client.request("upload binary", "POST", self.endpoint, stream=stream)
        self.assertEqual(result, {"ok": True})
        self.received(payload, "application/octet-stream")

    def test_upload_length_counts_only_bytes_after_current_offset(self):
        prefix = b"already consumed\x00\xff"
        payload = b"remaining\x00\xff\r\nbytes"
        path = self.root / "offset-upload"
        path.write_bytes(prefix + payload)
        with path.open("rb") as stream:
            stream.seek(len(prefix))
            result = self.client.request("upload remaining bytes", "POST", self.endpoint, stream=stream)
        self.assertEqual(result, {"ok": True})
        self.received(payload, "application/octet-stream")

    def test_upload_uses_validated_inode_after_filename_replacement(self):
        payload = b"verified original inode\x00\xff"
        path = self.release(payload)
        with contextlib.ExitStack() as stack:
            _, streams, _ = publisher.validate_release(path, stack)
            target = path / "features.safetensors"
            target.unlink()
            target.write_bytes(b"unverified replacement of a different size")
            result = self.client.request(
                "upload validated inode", "POST", self.endpoint,
                stream=streams["features.safetensors"],
            )
        self.assertEqual(result, {"ok": True})
        self.received(payload, "application/octet-stream")

    def test_metadata_request_preserves_json_body(self):
        metadata = {"draft": True, "name": "corpus metadata \u2603", "make_latest": "false"}
        result = self.client.request("create metadata", "POST", self.endpoint, body=metadata)
        self.assertEqual(result, {"ok": True})
        observation = self.received(json.dumps(metadata).encode(), "application/json")
        self.assertEqual(json.loads(observation["body"]), metadata)

    def test_http_failure_has_bounded_safe_detail_and_retains_patch_uncertainty(self):
        self.server.response_status = 422
        self.server.response_body = {
            "message": "Upload rejected " + TOKEN + "\x1b[31m\r\n\b\x00\u202e" + " oversized" * 1500,
        }
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(publisher.PublicationError) as caught:
                self.client.request("publish metadata", "PATCH", self.endpoint, body={"draft": False})
        diagnostic = str(caught.exception)
        self.assertIn("publish metadata", diagnostic)
        self.assertRegex(diagnostic, r"\b422\b")
        self.assertIn("Upload rejected", diagnostic)
        self.assertNotIn(TOKEN, diagnostic)
        self.assertNotIn(TOKEN, stderr.getvalue())
        self.assertTrue(all(character.isprintable() for character in diagnostic))
        self.assertLess(len(diagnostic), 2000)
        self.assertRegex(diagnostic, r"(?i)outcome.*unknown|may.*occur")
        observation = self.received(json.dumps({"draft": False}).encode(), "application/json")
        self.assertEqual(observation["method"], "PATCH")


if __name__ == "__main__":
    unittest.main()

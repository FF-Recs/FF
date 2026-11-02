#!/usr/bin/env python3
"""Publish one verified corpus: python3 scripts/publish-corpus.py RELEASE_PATH SOURCE_SHA.

Requires GH_TOKEN and the GitHub CLI on PATH. The repository is fixed. Validation
is entirely local before the first request; failures never delete releases or
assets. A failed write can leave this invocation's own unpublished draft.
"""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

REPOSITORY = "FF-Recs/FF"
API = f"repos/{REPOSITORY}"
PAYLOADS = {
    "aliases.parquet",
    "features.safetensors",
    "postings.safetensors",
    "posts.safetensors",
    "vocab.parquet",
}
ASSETS = PAYLOADS | {"manifest.json"}
ASSET_LIMIT = 2 * 1024**3  # GitHub requires each asset to be strictly smaller.
MANIFEST_LIMIT = 16 * 1024**2
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")


class PublicationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise PublicationError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Manifest contains a duplicate JSON key.")
        result[key] = value
    return result


def open_asset(path, stack):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    stream = stack.enter_context(os.fdopen(descriptor, "rb"))
    metadata = os.fstat(stream.fileno())
    require(stat.S_ISREG(metadata.st_mode), f"Not a regular asset: {path.name}")
    require(0 < metadata.st_size < ASSET_LIMIT, f"Asset must be nonempty and <2 GiB: {path.name}")
    return stream, metadata.st_size


def validate_release(path, stack):
    require(not path.is_symlink() and path.is_dir(), "Release path must be a real directory, not a symlink.")
    path = path.resolve(strict=True)
    require(SHA256.fullmatch(path.name), "Release directory name must be the full lowercase manifest SHA256.")
    require({child.name for child in path.iterdir()} == ASSETS, "Release must contain exactly manifest.json and the five runtime payloads.")
    streams = {}
    sizes = {}
    for name in sorted(ASSETS):
        streams[name], sizes[name] = open_asset(path / name, stack)
    require(sizes["manifest.json"] <= MANIFEST_LIMIT, "Manifest exceeds the 16 MiB validation bound.")
    raw_manifest = streams["manifest.json"].read(MANIFEST_LIMIT + 1)
    require(hashlib.sha256(raw_manifest).hexdigest() == path.name, "Directory identity does not match exact manifest bytes.")
    manifest = json.loads(raw_manifest, object_pairs_hook=unique_object)
    require(isinstance(manifest, dict), "Manifest must be a JSON object.")
    require(type(manifest.get("format_version")) is int and manifest["format_version"] == 5, "Only corpus format 5 can be published.")
    require(manifest.get("feature_model") == "binary_presence_and_signed_log_popularity", "Unsupported corpus feature model.")
    declarations = manifest.get("files")
    require(isinstance(declarations, dict) and set(declarations) == PAYLOADS, "Manifest must declare exactly the five allowed payloads.")
    buffer = bytearray(1024 * 1024)
    for name in sorted(PAYLOADS):
        declaration = declarations[name]
        require(isinstance(declaration, dict), f"Invalid descriptor: {name}")
        require({"bytes", "sha256"} <= set(declaration) <= {"bytes", "sha256", "tensors"}, f"Unexpected descriptor fields: {name}")
        require(type(declaration["bytes"]) is int and declaration["bytes"] == sizes[name], f"Declared size does not match: {name}")
        require(isinstance(declaration["sha256"], str) and SHA256.fullmatch(declaration["sha256"]), f"Invalid declared SHA256: {name}")
        digest = hashlib.sha256()
        stream = streams[name]
        while count := stream.readinto(buffer):
            digest.update(memoryview(buffer)[:count])
        require(digest.hexdigest() == declaration["sha256"], f"Payload SHA256 mismatch: {name}")
    for stream in streams.values():
        stream.seek(0)
    return path.name, streams, sizes


class GitHub:
    def __init__(self, config_directory):
        self.environment = os.environ.copy()
        # Prevent interactive auth, host overrides, debug traces or user gh config.
        for name in ("GH_HOST", "GH_REPO", "GH_DEBUG", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"):
            self.environment.pop(name, None)
        self.environment.update(GH_CONFIG_DIR=config_directory, GH_PROMPT_DISABLED="1", GH_PAGER="cat", NO_COLOR="1")

    def request(self, operation, method, endpoint, *, body=None, stream=None, paginate=False):
        print(f"Publication phase: {operation}", file=sys.stderr, flush=True)
        arguments = ["gh", "api", "--hostname", "github.com", "--method", method, endpoint]
        if paginate:
            arguments += ["--paginate", "--slurp"]
        if body is not None or stream is not None:
            arguments += ["--input", "-"]
        if stream is not None:
            remaining = os.fstat(stream.fileno()).st_size - stream.tell()
            # gh treats this header as the request length, avoiding chunked uploads.
            arguments += [
                "--header", "Content-Type: application/octet-stream",
                "--header", f"Content-Length: {remaining}",
            ]
        else:
            arguments += ["--header", "Accept: application/vnd.github+json"]
        # Direct argv, never a shell; file descriptors retain the validated inodes.
        result = subprocess.run(
            arguments,
            input=json.dumps(body).encode() if body is not None else None,
            stdin=stream,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.environment,
            check=False,
        )
        # Never echo raw CLI errors or response bodies: either can contain auth.
        failure_detail = "publication outcome may be unknown; inspect the owned release ID" if method == "PATCH" else "any created draft is retained"
        if result.returncode != 0:
            details = [f"exit {result.returncode}"]
            status = re.search(rb"\bHTTP ([1-5][0-9]{2})\b", result.stderr[-4096:])
            if status:
                details.append(f"HTTP {status[1].decode('ascii')}")
            # Bound parsing as well as displayed text; malformed provider errors
            # must not replace the original request failure with a parser error.
            response = None
            if len(result.stdout) <= 64 * 1024:
                try:
                    response = json.loads(result.stdout)
                except (ValueError, UnicodeError, RecursionError):
                    pass
            message = response.get("message") if isinstance(response, dict) else None
            if isinstance(message, str):
                token = self.environment.get("GH_TOKEN")
                if token:
                    message = message.replace(token, "[REDACTED]")
                message = " ".join("".join(character if character.isprintable() else " " for character in message).split())
                if message:
                    details.append(f"GitHub message: {message[:240]}")
            raise PublicationError(f"GitHub request failed during {operation} ({'; '.join(details)}); {failure_detail}.")
        try:
            return json.loads(result.stdout)
        except (ValueError, UnicodeError) as error:
            raise PublicationError(f"Invalid GitHub JSON during {operation}; {failure_detail}.") from error


def check_remote(release, release_id, tag, revision, sizes, *, draft):
    require(isinstance(release, dict), "GitHub returned an invalid release object.")
    require(type(release.get("id")) is int and release["id"] == release_id, "GitHub release identity changed.")
    require(release.get("tag_name") == tag, "GitHub release tag changed.")
    require(release.get("target_commitish") == revision, "GitHub release target does not match the exact source revision.")
    require(release.get("draft") is draft, "GitHub release draft state is unexpected.")
    assets = release.get("assets")
    require(isinstance(assets, list), "GitHub release assets are missing.")
    require(len(assets) == len(sizes), "Remote asset count does not match the six local assets.")
    names = set()
    for asset in assets:
        require(isinstance(asset, dict), "Invalid remote asset record.")
        name = asset.get("name")
        require(isinstance(name, str) and name in sizes and name not in names, "Unexpected or duplicate remote asset name.")
        require(type(asset.get("size")) is int and asset["size"] == sizes[name], f"Remote asset size mismatch: {name}")
        require(asset.get("state") == "uploaded", f"Remote asset is not fully uploaded: {name}")
        names.add(name)
    require(names == set(sizes), "Remote asset names differ from the local allowlist.")


def publish(path, revision):
    require(SOURCE_SHA.fullmatch(revision), "Source revision must be exactly 40 hexadecimal characters.")
    revision = revision.lower()
    with contextlib.ExitStack() as stack:
        corpus_hash, streams, sizes = validate_release(path, stack)
        require(bool(os.environ.get("GH_TOKEN")), "GH_TOKEN is required for publication.")
        scratch_root = Path(__file__).resolve().parent.parent / ".workdir" / "corpus-release"
        scratch_root.mkdir(parents=True, exist_ok=True)
        config = stack.enter_context(tempfile.TemporaryDirectory(prefix="publish-", dir=scratch_root))
        github = GitHub(config)
        tag = f"corpus-{corpus_hash}"
        # Include drafts and every page; never repurpose an existing release.
        pages = github.request("check tag availability", "GET", f"{API}/releases?per_page=100", paginate=True)
        require(isinstance(pages, list), "Invalid release listing.")
        for page in pages:
            require(isinstance(page, list), "Invalid paginated release listing.")
            for release in page:
                require(isinstance(release, dict), "Invalid release listing entry.")
                require(release.get("tag_name") != tag, "A release already exists for this manifest; refusing to overwrite it.")
        # target_commitish does not override an existing Git tag. Resolve the
        # actual commit, including annotated tags, before creating any draft.
        references = github.request("check existing tag reference", "GET", f"{API}/git/matching-refs/tags/{tag}")
        require(isinstance(references, list), "Invalid matching tag reference listing.")
        exact_references = []
        for reference in references:
            require(isinstance(reference, dict), "Invalid matching tag reference entry.")
            if reference.get("ref") == f"refs/tags/{tag}":
                exact_references.append(reference)
        require(len(exact_references) <= 1, "Duplicate exact tag references returned by GitHub.")
        if exact_references:
            commit = github.request("resolve existing tag commit", "GET", f"{API}/commits/{tag}")
            require(isinstance(commit, dict) and commit.get("sha") == revision, "Existing tag resolves to a different source revision; refusing publication.")
        created = github.request("create new draft", "POST", f"{API}/releases", body={
            "tag_name": tag,
            "target_commitish": revision,
            "name": f"Corpus {corpus_hash}",
            "body": f"Corpus manifest SHA256: `{corpus_hash}`\nSource revision: `{revision}`",
            "draft": True,
            "prerelease": False,
            "make_latest": "false",
        })
        require(isinstance(created, dict) and type(created.get("id")) is int and created["id"] > 0, "GitHub did not return a valid owned draft ID.")
        release_id = created["id"]
        print(f"Owned draft ID: {release_id}; prepublication failures retain this draft.", file=sys.stderr, flush=True)
        check_remote(created, release_id, tag, revision, {}, draft=True)
        for name in sorted(ASSETS):
            uploaded = github.request(
                f"upload {name}", "POST",
                f"https://uploads.github.com/{API}/releases/{release_id}/assets?name={name}",
                stream=streams[name],
            )
            require(isinstance(uploaded, dict) and uploaded.get("name") == name and type(uploaded.get("size")) is int and uploaded["size"] == sizes[name] and uploaded.get("state") == "uploaded", f"Uploaded asset response does not match: {name}")
        remote = github.request("verify final draft assets", "GET", f"{API}/releases/{release_id}")
        check_remote(remote, release_id, tag, revision, sizes, draft=True)
        try:
            published = github.request("publish verified draft", "PATCH", f"{API}/releases/{release_id}", body={"draft": False, "make_latest": "false"})
            check_remote(published, release_id, tag, revision, sizes, draft=False)
            commit = github.request("verify published tag commit", "GET", f"{API}/commits/{tag}")
            require(isinstance(commit, dict) and commit.get("sha") == revision, "Published tag does not resolve to the exact source revision.")
        except (PublicationError, OSError, ValueError, UnicodeError) as error:
            raise PublicationError(f"Publication may already have occurred; inspect owned release ID {release_id}: {error}") from error
        print(json.dumps({"repository": REPOSITORY, "tag": tag, "release_id": release_id, "source_revision": revision, "manifest_sha256": corpus_hash}))


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/publish-corpus.py RELEASE_PATH SOURCE_SHA", file=sys.stderr)
        return 2
    try:
        publish(Path(sys.argv[1]), sys.argv[2])
    except (PublicationError, OSError, ValueError, UnicodeError) as error:
        print(f"Publication stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

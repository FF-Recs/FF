# FF corpus builder

Headless Rust pipeline for acquiring dated compressed CSV exports, generating FF
format-5 corpora and publishing verified corpus assets to GitHub Releases.

## Build and check locally

Install the pinned Rust toolchain, a C/C++ compiler, Python 3 and the GitHub CLI.
On Ubuntu, install `build-essential`, `pkg-config` and CA certificates.

```bash
export CARGO_TARGET_DIR="$PWD/.workdir/cargo-target"
python3 -B -m unittest discover -s tests -v
cargo test --release --locked --workspace
"$CARGO_TARGET_DIR/release/ff-corpus-builder" --help
```

The shared `ff-core` crate supplies corpus readers and verification.
`ff-test-support` supplies synthetic fixtures for the tests.
Integration tests build the release CLI binaries used for generation.
Actor tests use real time for native worker readiness and completion, and virtual
time for timer assertions. Publisher transport tests exercise the real `gh`
executable against loopback HTTP servers with synthetic credentials.

## Generate and publish

Use the **Corpus release** workflow in `FF-Recs/FF`. It runs manually on a standard
Ubuntu GitHub runner; choose the source ref and provide:

- `export_date`: an explicit `YYYY-MM-DD` shared by all four current upstream
  catalog entries.
- `user_agent`: an identifiable upstream User-Agent, including your account/contact.
- `publish`: enable after publication authorization and the source/license/metadata
  redistribution review.

The runner downloads only posts, tags, tag aliases and tag implications from the
fixed upstream HTTPS origins. It checks compressed sizes/hashes, gzip/CSV integrity
and date consistency. The existing builder converts those inputs and verifies the
result through the real corpus consumers before release upload.

Release assets are `manifest.json`, `aliases.parquet`, `features.safetensors`,
`postings.safetensors`, `posts.safetensors` and `vocab.parquet`.

A release uses `corpus-<SHA256 of manifest.json>` and the exact source commit.
Publication validates payload hashes, sizes and asset names, uploads a new draft,
and publishes after its remote inventory matches. Each asset must be smaller
than 2 GiB. Existing releases retain their manifest identity.
Raw assets stream from validated file descriptors with an explicit HTTP length.
Failures report HTTP status and bounded, credential-redacted provider messages.

The publication step receives the job's GitHub token. GitHub permissions are
job-scoped; this manually dispatched job requires `contents: write`.

A failed publication retains its draft for inspection. Preserve useful assets
and obtain owner approval before clearing a failed draft. Runner-local inputs
and generated files last for the hosted job; retain a separate copy for reuse.

## Build cache and verification

The driver shares isolated build prerequisites across runs in
`.workdir/cargo-home`, `.workdir/rustup-home` and `.workdir/cargo-target`.
Inputs, metrics and generated releases live in per-run
`.workdir/corpus-release/run-*` directories.

Actions restores and saves the Cargo registry, pinned Rust toolchain and compiled
outputs. Cache keys cover platform, toolchain, dependency/build configuration and
source revision; compatible entries supply reusable dependencies. Cache saving
runs before publication and also after a generation failure. GitHub manages cache
retention; an evicted entry produces a cold build.

Every invocation runs the publisher and Rust suites, then uses the CLI binaries
produced by integration tests. Each generated corpus passes consumer verification.
Workflow logs report phase timings, maximum resident memory and sampled disk usage.

## Corpus activation

Obtain the expected manifest hash through a trusted channel, then verify
compatibility and every payload before activation.

Public distribution requires a code-license decision and review of dependency
and upstream metadata obligations.

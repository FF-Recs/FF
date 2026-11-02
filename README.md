# FF corpus builder

Acquire dated CSV exports, build verified FF format-5 corpora and publish them to
GitHub Releases.

## Generate a corpus

Open [Corpus release](https://github.com/FF-Recs/FF/actions/workflows/corpus-release.yml)
in GitHub Actions, choose a source ref and select **Run workflow**.

- `export_date`: `YYYY-MM-DD`, shared by posts, tags, aliases and implications in
  the [current export catalog](https://e621.net/db_exports.json).
- `user_agent`: an identifiable User-Agent with your username or contact.
- `publish`: enable after licensing and metadata redistribution review.

The workflow downloads, builds and verifies the corpus. With publication enabled,
assets appear in [Releases](https://github.com/FF-Recs/FF/releases) under
`corpus-<manifest SHA256>`. Inspect retained drafts and tags before retrying a
failed publication.

## Build and test locally

Install the pinned Rust toolchain, a C/C++ compiler, Python 3 and the GitHub CLI.
On Ubuntu, install `build-essential`, `pkg-config` and CA certificates.

```bash
export CARGO_TARGET_DIR="$PWD/.workdir/cargo-target"
python3 -B -m unittest discover -s tests -v
cargo test --release --locked --workspace
"$CARGO_TARGET_DIR/release/ff-corpus-builder" --help
```

- [Workflow](.github/workflows/corpus-release.yml) and [build driver](scripts/build-corpus-release.sh)
- [Corpus normalization policy](corpus-builder/src/contract.rs)


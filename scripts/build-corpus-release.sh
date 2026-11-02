#!/usr/bin/env bash
# Usage: bash scripts/build-corpus-release.sh YYYY-MM-DD 'Identifiable User-Agent'
# Retains its owned run (including inputs and resource reports) for inspection.
# In Actions, exports release_path and run_path through GITHUB_OUTPUT.
set -euo pipefail

if [[ $# != 2 || ! "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ || -z "$2" || "$2" == *$'\n'* || "$2" == *$'\r'* ]]; then
  printf 'Usage: bash scripts/build-corpus-release.sh YYYY-MM-DD "Identifiable User-Agent"\n' >&2
  exit 2
fi
export_date="$1"
user_agent="$2"
repository="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$repository"
mkdir -p -- "$repository/.workdir/corpus-release"
run="$(mktemp -d "$repository/.workdir/corpus-release/run-XXXXXXXX")"
mkdir -- "$run/metrics" "$run/inputs" "$run/releases" "$run/work"
# Build prerequisites are shared across runs, separate from user state and data.
export CARGO_HOME="$repository/.workdir/cargo-home"
export RUSTUP_HOME="$repository/.workdir/rustup-home"
export CARGO_TARGET_DIR="$repository/.workdir/cargo-target"
export CARGO_BUILD_JOBS=2
mkdir -p -- "$CARGO_HOME" "$RUSTUP_HOME" "$CARGO_TARGET_DIR"

snapshot() {
  printf '\nDisk snapshot sample (not an exact peak): %s\n' "$1" >&2
  df -h -- "$run" "$CARGO_TARGET_DIR" >&2
  du -sh -- "$run" "$CARGO_HOME" "$RUSTUP_HOME" "$CARGO_TARGET_DIR" >&2
}
finish() {
  status=$?
  trap - EXIT
  snapshot "exit status $status" || true
  printf 'Owned run retained: %s\n' "$run" >&2
  exit "$status"
}
trap finish EXIT

timed_phase() {
  local phase="$1" status=0
  shift
  printf '\n=== %s ===\n' "$phase" >&2
  snapshot "before $phase"
  /usr/bin/time -v -o "$run/metrics/$phase.time.txt" -- "$@" || status=$?
  cat -- "$run/metrics/$phase.time.txt" >&2
  snapshot "after $phase"
  return "$status"
}

# No GitHub token is required or accepted by any generation command.
unset GH_TOKEN GITHUB_TOKEN
timed_phase publisher-tests python3 -B -m unittest discover -s tests -v
# The committed rust-toolchain.toml pins this same toolchain for direct Cargo use.
if ! rustup run 1.95.0 rustc --version; then
  timed_phase toolchain rustup toolchain install 1.95.0 --profile minimal
fi
timed_phase tests rustup run 1.95.0 cargo test --release --locked -p ff-core -p ff-corpus-builder
builder="$CARGO_TARGET_DIR/release/ff-corpus-builder"
core="$CARGO_TARGET_DIR/release/ff-core"
if [[ ! -x "$builder" || ! -x "$core" ]]; then
  printf 'Tests did not produce both required release CLI binaries.\n' >&2
  exit 1
fi

# Only the builder resolves approved upstream endpoints and validates the date.
timed_phase download "$builder" download "$export_date" "$run/inputs" "$user_agent" > "$run/download-report.json"
snapshot_path="$(jq -er 'select(.file_count == 4) | .snapshot_path | select(type == "string" and length > 0)' "$run/download-report.json")"
snapshot_path="$(realpath -e -- "$snapshot_path")"
if [[ "$(dirname -- "$snapshot_path")" != "$run/inputs" || ! "$(basename -- "$snapshot_path")" =~ ^[0-9a-f]{64}$ || ! -d "$snapshot_path" ]]; then
  printf 'Downloader returned a path outside this run input snapshot.\n' >&2
  exit 1
fi

timed_phase generate "$builder" build \
  --posts "$snapshot_path/posts.csv.gz" --tags "$snapshot_path/tags.csv.gz" \
  --aliases "$snapshot_path/tag_aliases.csv.gz" \
  --implications "$snapshot_path/tag_implications.csv.gz" \
  --work "$run/work" --releases "$run/releases" --report "$run/build-report.json"
release_path="$(jq -er '.release_path | select(type == "string" and length > 0)' "$run/build-report.json")"
release_path="$(realpath -e -- "$release_path")"
if [[ "$(dirname -- "$release_path")" != "$run/releases" || ! "$(basename -- "$release_path")" =~ ^[0-9a-f]{64}$ || ! -d "$release_path" ]]; then
  printf 'Builder returned a path outside this run releases.\n' >&2
  exit 1
fi
# An explicit path is mandatory: generation never selects an application corpus.
timed_phase verify "$core" --verify-corpus --corpus-path "$release_path"
printf '\nVerified release: %s\n' "$release_path" >&2
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'release_path=%s\nrun_path=%s\n' "$release_path" "$run" >> "$GITHUB_OUTPUT"
fi
jq -n --arg release_path "$release_path" --arg run_path "$run" \
  '{release_path: $release_path, run_path: $run_path}'

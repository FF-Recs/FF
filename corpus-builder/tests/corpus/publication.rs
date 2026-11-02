use super::support::{fixture, gzip};
use ff_core::corpus::verify_corpus;
use ff_corpus_builder::{build, build::sha256, equivalence};
use std::fs;

#[test]
fn publication_is_deterministic_new_only_provenance_bound_and_cleans_failed_runs() {
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let options = fixture(root.path());
    let first = build(&options).unwrap();
    let manifest_before = sha256(&first.release_path.join("manifest.json")).unwrap();
    assert_eq!(
        first.release_path.file_name().unwrap().to_str().unwrap(),
        manifest_before
    );
    let mut other = options.clone();
    other.releases = root.path().join("other-releases");
    let second = build(&other).unwrap();
    assert_eq!(first.corpus_id, second.corpus_id);
    assert_eq!(
        sha256(&second.release_path.join("manifest.json")).unwrap(),
        manifest_before
    );
    let comparison = equivalence::compare(&first.release_path, &second.release_path).unwrap();
    assert_eq!(comparison["status"], "equivalent");
    assert!(build(&options).is_err());
    assert_eq!(
        sha256(&first.release_path.join("manifest.json")).unwrap(),
        manifest_before
    );
    assert_eq!(fs::read_dir(&options.releases).unwrap().count(), 1);
    assert_eq!(fs::read_dir(&options.work).unwrap().count(), 0);
    // A new unused input tag changes source provenance, not selected logical payloads.
    let mut decoder = flate2::read::MultiGzDecoder::new(fs::File::open(&options.tags).unwrap());
    let mut tags = String::new();
    std::io::Read::read_to_string(&mut decoder, &mut tags).unwrap();
    tags.push_str("999999,unused_tag,0\n");
    gzip(&options.tags, &tags);
    let third = build(&options).unwrap();
    assert_eq!(third.corpus_id, first.corpus_id);
    assert_ne!(third.release_path, first.release_path);
    // Unknown active content fails before publication and cleans only its new run.
    gzip(
        &options.posts,
        "id,md5,rating,file_ext,is_deleted,tag_string,score\n1,00000000000000000000000000000001,s,jpg,f,unknown_tag,0\n",
    );
    assert!(build(&options).is_err());
    assert_eq!(fs::read_dir(&options.releases).unwrap().count(), 2);
    assert_eq!(fs::read_dir(&options.work).unwrap().count(), 0);
    assert_eq!(
        sha256(&first.release_path.join("manifest.json")).unwrap(),
        manifest_before
    );
}

#[test]
fn unsupported_neighboring_inputs_do_not_change_identity_or_cross_publication_boundary() {
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let options = fixture(root.path());
    let baseline = build(&options).unwrap();
    // Unsupported exports and unrelated neighbor files are not discovered,
    // parsed, hashed as provenance, or copied into the release.
    gzip(
        &root.path().join("post_replacements.csv.gz"),
        "not a supported export",
    );
    fs::write(root.path().join("private-notes.txt"), b"not corpus data").unwrap();
    let mut other = options.clone();
    other.releases = root.path().join("other-releases");
    let result = build(&other).unwrap();
    verify_corpus(&result.release_path).unwrap();
    assert_eq!(result.corpus_id, baseline.corpus_id);
    assert_eq!(
        sha256(&result.release_path.join("manifest.json")).unwrap(),
        sha256(&baseline.release_path.join("manifest.json")).unwrap()
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(result.release_path.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(
        manifest["sources"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "posts.csv.gz",
            "tag_aliases.csv.gz",
            "tag_implications.csv.gz",
            "tags.csv.gz"
        ]
    );
    let mut published = fs::read_dir(&result.release_path)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect::<Vec<_>>();
    published.sort_unstable();
    assert_eq!(
        published,
        [
            "aliases.parquet",
            "features.safetensors",
            "manifest.json",
            "postings.safetensors",
            "posts.safetensors",
            "vocab.parquet"
        ]
    );
}

#[test]
fn cli_report_preflight_preserves_existing_file_without_starting_a_build() {
    use std::process::Command;
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let options = fixture(root.path());
    fs::create_dir_all(options.work.join("foreign-run")).unwrap();
    fs::create_dir_all(options.releases.join(".build-foreign")).unwrap();
    fs::write(options.work.join("foreign-run/keep"), b"foreign work").unwrap();
    fs::write(
        options.releases.join(".build-foreign/keep"),
        b"foreign release",
    )
    .unwrap();
    let report = root.path().join("report.json");
    let sentinel = b"caller-owned report, not builder JSON\n";
    fs::write(&report, sentinel).unwrap();
    let invoke = || {
        Command::new(env!("CARGO_BIN_EXE_ff-corpus-builder"))
            .arg("build")
            .arg("--posts")
            .arg(&options.posts)
            .arg("--tags")
            .arg(&options.tags)
            .arg("--aliases")
            .arg(&options.aliases)
            .arg("--implications")
            .arg(&options.implications)
            .arg("--work")
            .arg(&options.work)
            .arg("--releases")
            .arg(&options.releases)
            .arg("--report")
            .arg(&report)
            .output()
            .unwrap()
    };
    let rejected = invoke();
    assert!(!rejected.status.success());
    assert!(rejected.stdout.is_empty());
    assert_eq!(fs::read(&report).unwrap(), sentinel);
    assert_eq!(fs::read_dir(&options.work).unwrap().count(), 1);
    assert_eq!(fs::read_dir(&options.releases).unwrap().count(), 1);
    // Prove valid inputs and CLI syntax by succeeding after the sole obstruction
    // is removed, then preserve the successful new report on another invocation.
    fs::remove_file(&report).unwrap();
    let accepted = invoke();
    assert!(
        accepted.status.success(),
        "{}",
        String::from_utf8_lossy(&accepted.stderr)
    );
    let bytes = fs::read(&report).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let release = std::path::PathBuf::from(parsed["release_path"].as_str().unwrap());
    verify_corpus(&release).unwrap();
    assert_eq!(parsed["post_count"], 61);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&accepted.stdout).unwrap(),
        parsed
    );
    assert!(!invoke().status.success());
    assert_eq!(fs::read(&report).unwrap(), bytes);
    assert_eq!(fs::read_dir(&options.work).unwrap().count(), 1);
    assert_eq!(fs::read_dir(&options.releases).unwrap().count(), 2);
    assert_eq!(
        fs::read(options.work.join("foreign-run/keep")).unwrap(),
        b"foreign work"
    );
    assert_eq!(
        fs::read(options.releases.join(".build-foreign/keep")).unwrap(),
        b"foreign release"
    );
}

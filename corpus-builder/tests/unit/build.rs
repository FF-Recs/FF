use super::*;

fn spool_fixture(path: &Path, rows: &[&[i32]], extensions: &[u8]) {
    let mut file = writer(path).unwrap();
    file.write_all(SPOOL_MAGIC).unwrap();
    file.write_all(&u32::try_from(rows.len()).unwrap().to_le_bytes())
        .unwrap();
    for (row, tags) in rows.iter().enumerate() {
        let mut header = [0u8; 38];
        header[..8].copy_from_slice(&i64::try_from(row + 1).unwrap().to_le_bytes());
        header[24] = extensions[row];
        header[25] = 1;
        header[26..34].copy_from_slice(&0.0f64.to_le_bytes());
        header[34..].copy_from_slice(&u32::try_from(tags.len()).unwrap().to_le_bytes());
        file.write_all(&header).unwrap();
        file.write_all(bytemuck::cast_slice(tags)).unwrap();
    }
    finish(file).unwrap();
}

fn entry(index: i32, category: i32, occurrences: i64) -> VocabularyEntry {
    VocabularyEntry {
        index,
        source_id: i64::from(index) + 1,
        name: format!("tag_{index}"),
        category,
        occurrences,
    }
}

#[test]
fn source_rotation_after_extraction_rejects_publication_and_preserves_foreign_work() {
    use flate2::{Compression, write::GzEncoder};

    fn gzip(path: &Path, text: &str) {
        let mut encoder = GzEncoder::new(File::create(path).unwrap(), Compression::fast());
        encoder.write_all(text.as_bytes()).unwrap();
        encoder.finish().unwrap();
    }
    fn rotate(options: &BuildOptions, stage: &Path) {
        // Reaching this observer proves extraction, serialization and real
        // consumer validation already completed with the initial source.
        ff_core::corpus::verify_corpus(stage).unwrap();
        let manifest: Value =
            serde_json::from_reader(File::open(stage.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(
            manifest["sources"]["tags.csv.gz"],
            sha256(&options.tags).unwrap()
        );
        gzip(
            &options.tags,
            "id,name,category\n1,fox,0\n2,new_unused_tag,0\n",
        );
        assert_ne!(
            manifest["sources"]["tags.csv.gz"],
            sha256(&options.tags).unwrap()
        );
    }
    struct ResetObserver;
    impl Drop for ResetObserver {
        fn drop(&mut self) {
            BEFORE_INPUT_RECHECK.with(|observer| observer.set(None));
        }
    }

    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let options = BuildOptions {
        posts: root.path().join("posts.gz"),
        tags: root.path().join("tags.gz"),
        aliases: root.path().join("aliases.gz"),
        implications: root.path().join("implications.gz"),
        work: root.path().join("work"),
        releases: root.path().join("releases"),
    };
    gzip(&options.tags, "id,name,category\n1,fox,0\n");
    gzip(&options.aliases, "antecedent_name,consequent_name,status\n");
    gzip(
        &options.implications,
        "antecedent_name,consequent_name,status\n",
    );
    let mut posts = String::from("id,md5,rating,file_ext,is_deleted,tag_string,score\n");
    for id in 1..=50 {
        posts.push_str(&format!("{id},{id:032x},s,jpg,f,fox,0\n"));
    }
    gzip(&options.posts, &posts);
    fs::create_dir_all(options.work.join("foreign-run")).unwrap();
    fs::create_dir_all(options.releases.join(".build-foreign")).unwrap();
    fs::write(options.work.join("foreign-run/keep"), b"foreign work").unwrap();
    fs::write(
        options.releases.join(".build-foreign/keep"),
        b"foreign stage",
    )
    .unwrap();
    let before = sha256(&options.tags).unwrap();
    BEFORE_INPUT_RECHECK.with(|observer| {
        assert!(observer.replace(Some(rotate)).is_none());
    });
    let reset = ResetObserver;
    let result = build(&options);
    drop(reset);
    assert_ne!(sha256(&options.tags).unwrap(), before);
    assert!(result.is_err());
    assert_eq!(fs::read_dir(&options.work).unwrap().count(), 1);
    assert_eq!(fs::read_dir(&options.releases).unwrap().count(), 1);
    assert_eq!(
        fs::read(options.work.join("foreign-run/keep")).unwrap(),
        b"foreign work"
    );
    assert_eq!(
        fs::read(options.releases.join(".build-foreign/keep")).unwrap(),
        b"foreign stage"
    );
    // The rotated source is valid: without the observer a fresh build
    // succeeds. Rejection above is not a parse/schema failure.
    let accepted = build(&options).unwrap();
    ff_core::corpus::verify_corpus(&accepted.release_path).unwrap();
    assert_eq!(accepted.post_count, 50);
    assert_eq!(accepted.tag_count, 1);
    assert_eq!(fs::read_dir(&options.work).unwrap().count(), 1);
    assert_eq!(fs::read_dir(&options.releases).unwrap().count(), 2);
}

#[test]
fn replay_rejects_count_drift_and_short_or_trailing_rows_and_removes_staged_files() {
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let spool = root.path().join("spool");
    // Equal total counts cannot hide a changed distribution across tags.
    spool_fixture(&spool, &[&[0], &[0]], &[1, 1]);
    let vocabulary = [entry(0, 0, 1), entry(1, 0, 1)];
    assert!(indexed_tensors(root.path(), &spool, 2, &vocabulary, &[0, 1]).is_err());
    let vocabulary = [entry(0, 0, 1), entry(1, 1, 1)];
    assert!(indexed_tensors(root.path(), &spool, 2, &vocabulary, &[0, 1]).is_err());
    for file in [
        "features.safetensors",
        "posts.safetensors",
        "postings.safetensors",
    ] {
        assert!(!root.path().join(file).exists());
    }
    std::fs::remove_file(&spool).unwrap();
    spool_fixture(&spool, &[&[0]], &[1]);
    // Too many planned links are rejected independently of file allocation.
    assert!(indexed_tensors(root.path(), &spool, 1, &[entry(0, 0, 2)], &[0]).is_err());
    // A row cannot overrun a named segment even when its taxonomy tag is valid.
    assert!(indexed_tensors(root.path(), &spool, 1, &[entry(0, 0, 0)], &[0]).is_err());
    // EOF is checked at both boundaries, not inferred from preallocated length.
    let original = std::fs::read(&spool).unwrap();
    std::fs::write(&spool, &original[..original.len() - 1]).unwrap();
    assert!(indexed_tensors(root.path(), &spool, 1, &[entry(0, 0, 1)], &[0]).is_err());
    let mut trailing = original;
    trailing.push(0);
    std::fs::write(&spool, trailing).unwrap();
    assert!(indexed_tensors(root.path(), &spool, 1, &[entry(0, 0, 1)], &[0]).is_err());
    for file in [
        "features.safetensors",
        "posts.safetensors",
        "postings.safetensors",
    ] {
        assert!(!root.path().join(file).exists());
    }
}

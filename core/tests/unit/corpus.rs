use super::*;
use std::io::Write;

fn scratch() -> tempfile::TempDir {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.workdir/ff-rs/corpus");
    std::fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}
fn write_tensor(path: &Path, header: &str, payload: &[u8]) {
    let padded_length = (header.len() + 7) / 8 * 8;
    let mut file = File::create(path).unwrap();
    file.write_all(&(padded_length as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&vec![b' '; padded_length - header.len()])
        .unwrap();
    file.write_all(payload).unwrap();
}
fn header(descriptor: &str) -> String {
    format!(
        r#"{{"__metadata__":{{"ff_format_version":"5","ff_group":"features"}},"values":{descriptor}}}"#
    )
}
#[test]
fn opens_posts_with_zero_copy_popularity_media_and_all_artist_links() {
    let temp = scratch();
    fn ints(values: &[i32]) -> Vec<u8> {
        bytemuck::cast_slice(values).to_vec()
    }
    fn vector(name: &str, values: &[i32]) -> (String, String, Vec<usize>, Vec<u8>) {
        (name.into(), "I32".into(), vec![values.len()], ints(values))
    }
    fn group(
        root: &Path,
        name: &str,
        mut tensors: Vec<(String, String, Vec<usize>, Vec<u8>)>,
    ) -> serde_json::Value {
        let mut header = serde_json::Map::new();
        header.insert(
            "__metadata__".into(),
            serde_json::json!({"ff_format_version":"5","ff_group":name}),
        );
        tensors.sort_by_key(|(_, dtype, _, _)| {
            std::cmp::Reverse(match dtype.as_str() {
                "I64" | "F64" => 8,
                "I32" | "F32" => 4,
                "U8" => 1,
                _ => unreachable!(),
            })
        });
        let mut data = Vec::new();
        let mut names = Vec::new();
        for (tensor, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend(bytes);
            header.insert(
                tensor.clone(),
                serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}),
            );
            names.push(tensor);
        }
        let path = root.join(format!("{name}.safetensors"));
        write_tensor(&path, &serde_json::to_string(&header).unwrap(), &data);
        let bytes = std::fs::read(&path).unwrap();
        serde_json::json!({"bytes":bytes.len(),"sha256":hex::encode(Sha256::digest(&bytes)),"tensors":names})
    }
    let features = group(
        temp.path(),
        "features",
        vec![
            vector("post_tag_offsets", &[0, 3, 5, 5, 5, 5, 5, 5]),
            vector("post_tag_indices", &[0, 1, 2, 1, 2]),
        ],
    );
    let post_tensors = vec![
        (
            "post_ids".into(),
            "I64".into(),
            vec![7],
            bytemuck::cast_slice(&[10i64, 20, 30, 40, 50, 60, 70]).to_vec(),
        ),
        (
            "popularity".into(),
            "F64".into(),
            vec![7],
            bytemuck::cast_slice(&[-3.0f64, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).to_vec(),
        ),
        (
            "md5_bytes".into(),
            "U8".into(),
            vec![7, 16],
            (0u8..112).collect(),
        ),
        (
            "file_extension_codes".into(),
            "U8".into(),
            vec![7],
            vec![1, 2, 3, 4, 1, 6, 7],
        ),
        (
            "rating_codes".into(),
            "U8".into(),
            vec![7],
            vec![1, 2, 3, 1, 2, 3, 1],
        ),
    ];
    let posts = group(temp.path(), "posts", post_tensors.clone());
    let postings = group(
        temp.path(),
        "postings",
        vec![
            vector("tag_post_offsets", &[0, 1, 3, 5]),
            vector("tag_post_indices", &[0, 0, 1, 0, 1]),
            vector("artist_vocab_indices", &[0, 1]),
            vector("ordinal_metadata_vocab_indices", &[2]),
            vector("post_artist_offsets", &[0, 2, 3, 3, 3, 3, 3, 3]),
            vector("post_artist_indices", &[0, 1, 1]),
        ],
    );
    let empty = serde_json::json!({"bytes":0,"sha256":hex::encode(Sha256::digest([]))});
    let mut manifest = serde_json::json!({"format_version":5,"corpus_id":"real-layout-test","post_count":7,"tag_count":3,"nonzero_count":5,"feature_model":"binary_presence_and_signed_log_popularity","files":{
        "features.safetensors":features,"posts.safetensors":posts,"postings.safetensors":postings,"vocab.parquet":empty,"aliases.parquet":empty,
    }});
    std::fs::write(
        temp.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let corpus = Corpus::open(temp.path()).unwrap();
    assert_eq!(
        (corpus.post_count, corpus.tag_count, corpus.artist_count),
        (7, 3, 2)
    );
    assert!(matches!(&corpus.post_indices, Tensor::Mapped { .. }));
    assert!(matches!(&corpus.tag_indices, Tensor::Mapped { .. }));
    assert!(matches!(&corpus.ids, Tensor::Mapped { .. }));
    assert!(matches!(&corpus.md5s, Tensor::Mapped { .. }));
    assert!(matches!(&corpus.extensions, Tensor::Mapped { .. }));
    assert!(matches!(&corpus.ratings, Tensor::Mapped { .. }));
    assert!(matches!(&corpus.artist_indices, Tensor::Mapped { .. }));
    assert!(matches!(&corpus.popularities, Tensor::Mapped { .. }));
    assert_eq!(corpus.popularities(), [-3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(corpus.popularity(2), 0.0);
    assert_eq!(
        (0..7).map(|row| corpus.extension(row)).collect::<Vec<_>>(),
        ["jpg", "png", "gif", "webm", "jpg", "mp4", "webp"]
    );
    assert_eq!(corpus.md5(0), "000102030405060708090a0b0c0d0e0f");
    assert_eq!(corpus.rating_code(2), 3);
    assert_eq!(corpus.post_artists(0), [0, 1]);
    assert_eq!(corpus.post_artists(1), [1]);
    assert!(corpus.post_artists(2).is_empty());
    assert_eq!(corpus.tag_posts(2), [0, 1]);
    assert_eq!(corpus.find_post(70), Some(6));
    assert_eq!(
        corpus.manifest_sha256,
        streaming_sha256(&temp.path().join("manifest.json")).unwrap()
    );
    drop(corpus);
    let mut invalid_posts = Vec::new();
    let mut missing = post_tensors.clone();
    missing.retain(|(name, _, _, _)| name != "popularity");
    invalid_posts.push((missing, "Missing tensor 'popularity'"));
    let mut wrong_shape = post_tensors.clone();
    wrong_shape[1].2 = vec![1, 7];
    invalid_posts.push((wrong_shape, "Wrong shape for popularity"));
    let mut wrong_dtype = post_tensors.clone();
    wrong_dtype[1].1 = "F32".into();
    wrong_dtype[1].3 = bytemuck::cast_slice(&[0.0f32; 7]).to_vec();
    invalid_posts.push((wrong_dtype, "Wrong dtype for popularity"));
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut nonfinite = post_tensors.clone();
        nonfinite[1].3 = bytemuck::cast_slice(&[value; 7]).to_vec();
        invalid_posts.push((
            nonfinite,
            "Popularity must contain one finite value per post",
        ));
    }
    for (tensors, expected) in invalid_posts {
        manifest["files"]["posts.safetensors"] = group(temp.path(), "posts", tensors);
        std::fs::write(
            temp.path().join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let error = Corpus::open(temp.path()).err().unwrap().to_string();
        assert!(error.contains(expected), "{error}");
    }
    for version in [3, 4] {
        manifest["format_version"] = serde_json::json!(version);
        std::fs::write(
            temp.path().join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(read_manifest(temp.path()).is_err());
    }
}
#[test]
fn tensor_borrow_is_bounded_to_payload_and_retains_mapping() {
    let temp = scratch();
    let path = temp.path().join("tensor");
    write_tensor(
        &path,
        &header(r#"{"dtype":"I32","shape":[2],"data_offsets":[0,8]}"#),
        bytemuck::cast_slice(&[10i32, 20]),
    );
    let tensor = {
        let file = TensorFile::open(&path, "features", Some(&["values".into()])).unwrap();
        assert!(file.tensor::<i64>("values", Dtype::I64, None).is_err());
        assert!(
            file.tensor::<i32>("values", Dtype::I32, Some(&[3]))
                .is_err()
        );
        file.tensor::<i32>("values", Dtype::I32, Some(&[2]))
            .unwrap()
    };
    assert_eq!(tensor.slice(), [10, 20]);
    assert!(matches!(&tensor, Tensor::Mapped { .. }));
}
#[test]
fn descriptors_reject_overflow_truncation_overlap_and_duplicate_keys() {
    let temp = scratch();
    let path = temp.path().join("tensor");
    for descriptor in [
        r#"{"dtype":"I32","shape":[2],"data_offsets":[0,12]}"#.to_owned(),
        r#"{"dtype":"I32","shape":[2],"data_offsets":[1,9]}"#.to_owned(),
        r#"{"dtype":"I32","shape":[2],"data_offsets":[8,0]}"#.to_owned(),
        r#"{"dtype":"I32","dtype":"I32","shape":[2],"data_offsets":[0,8]}"#.to_owned(),
        r#"{"dtype":"I32","shape":[2],"shape":[2],"data_offsets":[0,8]}"#.to_owned(),
        r#"{"dtype":"I32","shape":[2],"data_offsets":[0,8],"data_offsets":[0,8]}"#.to_owned(),
        r#"{"dtype":"I32","shape":[2],"data_offsets":[0,8],"extra":false}"#.to_owned(),
        r#"{"dtype":"F16","shape":[4],"data_offsets":[0,8]}"#.to_owned(),
        r#"{"dtype":"I32","shape":null,"data_offsets":[0,8]}"#.to_owned(),
        r#"{"dtype":"I32","shape":[2],"data_offsets":[0,8,8]}"#.to_owned(),
        r#"{"dtype":"I32","shape":[-1],"data_offsets":[0,8]}"#.to_owned(),
        format!(
            r#"{{"dtype":"I64","shape":[{},8],"data_offsets":[0,8]}}"#,
            usize::MAX
        ),
    ] {
        write_tensor(&path, &header(&descriptor), &[0; 8]);
        assert!(
            TensorFile::open(&path, "features", None).is_err(),
            "{descriptor}"
        );
    }
    let repeated = r#"{"__metadata__":{"ff_format_version":"5","ff_group":"features"},"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"I32","shape":[1],"data_offsets":[4,8]}}"#;
    write_tensor(&path, repeated, &[0; 8]);
    assert!(TensorFile::open(&path, "features", None).is_err());
    let overlap = repeated.replacen(
        r#""a":{"dtype":"I32","shape":[1],"data_offsets":[4,8]}"#,
        r#""b":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}"#,
        1,
    );
    write_tensor(&path, &overlap, &[0; 8]);
    assert!(TensorFile::open(&path, "features", None).is_err());
    let gap = r#"{"__metadata__":{"ff_format_version":"5","ff_group":"features"},"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"I32","shape":[1],"data_offsets":[8,12]}}"#;
    write_tensor(&path, gap, &[0; 12]);
    assert!(TensorFile::open(&path, "features", None).is_err());
    write_tensor(
        &path,
        &header(r#"{"dtype":"I32","shape":[1],"data_offsets":[0,4]}"#),
        &[0; 8],
    );
    assert!(
        TensorFile::open(&path, "features", None).is_err(),
        "trailing bytes"
    );
    std::fs::write(&path, u64::MAX.to_le_bytes()).unwrap();
    assert!(TensorFile::open(&path, "features", None).is_err());
}
#[test]
fn metadata_and_declared_names_remain_strict_consumer_boundaries() {
    let temp = scratch();
    let path = temp.path().join("tensor");
    for text in [
        r#"{"__metadata__":{"ff_format_version":"5","ff_group":"features","ff_group":"features"},"values":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}"#,
        r#"{"__metadata__":{"ff_format_version":"5","ff_group":"features"},"__metadata__":{"ff_format_version":"5","ff_group":"features"},"values":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}"#,
        r#"{"values":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}"#,
        r#"{"__metadata__":null,"values":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}"#,
        r#"{"__metadata__":{"ff_format_version":5,"ff_group":"features"},"values":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}"#,
        r#"{"__metadata__":{"ff_format_version":"4","ff_group":"features"},"values":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}"#,
        r#"{"__metadata__":{"ff_format_version":"5","ff_group":"posts"},"values":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}"#,
        r#"{"__metadata__":{"ff_format_version":"5","ff_group":"features"},"values":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}{}"#,
    ] {
        write_tensor(&path, text, &[0; 8]);
        assert!(TensorFile::open(&path, "features", None).is_err(), "{text}");
    }
    write_tensor(
        &path,
        &header(r#"{"dtype":"I32","shape":[2],"data_offsets":[0,8]}"#),
        &[0; 8],
    );
    for declared in [
        vec![],
        vec!["other".into()],
        vec!["values".into(), "values".into()],
    ] {
        assert!(TensorFile::open(&path, "features", Some(&declared)).is_err());
    }
}
#[test]
fn admission_bounds_upstream_final_length_arithmetic_and_header_resources() {
    let temp = scratch();
    let path = temp.path().join("tensor");
    // Each descriptor's shape-to-bits multiplication fits, but the cumulative
    // final endpoint plus framing would overflow read_metadata's usize sum.
    let width = usize::MAX / 8;
    let descriptors = (0..8)
        .map(|i| {
            format!(
                r#""t{i}":{{"dtype":"U8","shape":[{width}],"data_offsets":[{},{}]}}"#,
                i * width,
                (i + 1) * width
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let text = format!(
        r#"{{"__metadata__":{{"ff_format_version":"5","ff_group":"features"}},{descriptors}}}"#
    );
    write_tensor(&path, &text, &[]);
    assert!(TensorFile::open(&path, "features", None).is_err());
    for length in [0, MAX_HEADER as u64 + 1, u64::MAX] {
        std::fs::write(&path, [length.to_le_bytes().as_slice(), b"{}"].concat()).unwrap();
        assert!(TensorFile::open(&path, "features", None).is_err());
    }
    std::fs::write(&path, [64u64.to_le_bytes().as_slice(), b"{}"].concat()).unwrap();
    assert!(TensorFile::open(&path, "features", None).is_err());
}
#[test]
fn scalar_float_and_empty_tensor_preserve_official_shapes_and_mapped_bytes() {
    let temp = scratch();
    let path = temp.path().join("tensor");
    let text = r#"{"__metadata__":{"ff_format_version":"5","ff_group":"features"},"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"empty":{"dtype":"I32","shape":[0],"data_offsets":[0,0]}}"#;
    write_tensor(&path, text, &1.25f32.to_le_bytes());
    let (scalar, empty) = {
        let file = TensorFile::open(&path, "features", None).unwrap();
        assert!(file.tensor::<f32>("scalar", Dtype::F32, None).is_err());
        (
            file.tensor::<f32>("scalar", Dtype::F32, Some(&[])).unwrap(),
            file.tensor::<i32>("empty", Dtype::I32, Some(&[0])).unwrap(),
        )
    };
    assert_eq!(scalar.slice(), [1.25]);
    assert!(empty.slice().is_empty());
    assert!(matches!(&scalar, Tensor::Mapped { .. }));
    assert!(matches!(&empty, Tensor::Mapped { .. }));
}
#[test]
fn unaligned_incidence_is_rejected_without_a_copying_fallback() {
    let temp = scratch();
    let path = temp.path().join("tensor");
    let text = r#"{"__metadata__":{"ff_format_version":"5","ff_group":"features"},"prefix":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"post_tag_indices":{"dtype":"I32","shape":[1],"data_offsets":[1,5]}}"#;
    let mut payload = vec![1];
    payload.extend_from_slice(&7i32.to_le_bytes());
    write_tensor(&path, text, &payload);
    // The container is valid, but this typed FF incidence view is not representable.
    let file = TensorFile::open(&path, "features", None).unwrap();
    assert!(
        file.tensor::<i32>("post_tag_indices", Dtype::I32, Some(&[1]))
            .is_err()
    );
}
#[test]
fn csr_rejects_invalid_boundaries_duplicates_and_unsorted_data() {
    for offsets in [&[0, -1, 2][..], &[0, 3, 2], &[1, 1, 2], &[0, 1, 9]] {
        assert!(validate_csr(offsets, &[0, 1], 2, "test").is_err());
    }
    assert!(validate_csr(&[0, 2], &[1, 1], 2, "test").is_err());
    assert!(validate_csr(&[0, 2], &[1, 0], 2, "test").is_err());
    assert!(validate_csr(&[0, 1], &[2], 2, "test").is_err());
    assert!(validate_csr(&[0, 0, 0], &[], 0, "test").is_ok());
}
#[test]
fn corpus_preserves_indices_media_and_binary_lookup() {
    let corpus = Corpus::fixture(
        &[10, 20, 30],
        &[vec![0, 1, 2], vec![1, 2], vec![]],
        3,
        &[0, 1],
        &[2],
    );
    assert_eq!(corpus.post_tags(1), [1, 2]);
    assert_eq!(corpus.tag_posts(2), [0, 1]);
    assert_eq!(corpus.post_artists(1), [1]);
    assert_eq!(corpus.post_artists(0), [0, 1]);
    assert!(corpus.post_artists(2).is_empty());
    assert_eq!(corpus.find_post(20), Some(1));
    assert_eq!(corpus.find_post(21), None);
    assert_eq!(corpus.md5(0), "0".repeat(32));
    assert_eq!(corpus.extension(0), "jpg");
    assert_eq!(corpus.rating_code(0), 1);
}
#[test]
fn invalid_media_fail_before_exposure() {
    let mut corpus = Corpus::fixture(&[10], &[vec![]], 0, &[], &[]);
    for invalid in [0, 5, 8, u8::MAX] {
        corpus.extensions = Tensor::Owned(vec![invalid]);
        assert!(corpus.validate().is_err(), "extension code {invalid}");
    }
    corpus.extensions = Tensor::Owned(vec![1]);
    corpus.ratings = Tensor::Owned(vec![4]);
    assert!(corpus.validate().is_err());
}
#[test]
fn post_artist_membership_must_be_present_in_post_tags() {
    let mut corpus = Corpus::fixture(&[10], &[vec![]], 1, &[0], &[]);
    corpus.artist_offsets = Tensor::Owned(vec![0, 1]);
    corpus.artist_indices = Tensor::Owned(vec![0]);
    assert!(corpus.validate().is_err());
}
#[test]
fn post_ids_must_be_positive_unique_and_sorted() {
    let mut corpus = Corpus::fixture(&[10, 20], &[vec![], vec![]], 0, &[], &[]);
    for ids in [[0, 20], [-10, 20], [10, 10], [20, 10]] {
        corpus.ids = Tensor::Owned(ids.to_vec());
        assert!(corpus.validate().is_err(), "{ids:?}");
    }
}
#[test]
fn pin_requires_exact_manifest_hash_and_single_property() {
    let temp = scratch();
    let manifest = b"{\"arbitrary\":\"exact bytes\"}\n";
    let digest = hex::encode(Sha256::digest(manifest));
    let release = temp.path().join("data/corpora").join(&digest);
    std::fs::create_dir_all(&release).unwrap();
    std::fs::write(release.join("manifest.json"), manifest).unwrap();
    let pin = temp.path().join("corpus-release.json");
    std::fs::write(&pin, format!(r#"{{"manifest_sha256":"{digest}"}}"#)).unwrap();
    assert_eq!(resolve_corpus(temp.path()).unwrap(), release);
    std::fs::write(release.join("manifest.json"), b"changed").unwrap();
    assert!(resolve_corpus(temp.path()).is_err());
    std::fs::write(release.join("manifest.json"), manifest).unwrap();
    std::fs::write(
        &pin,
        format!(r#"{{"manifest_sha256":"{digest}","other":0}}"#),
    )
    .unwrap();
    assert!(resolve_corpus(temp.path()).is_err());
    std::fs::write(&pin, r#"{"manifest_sha256":"../outside"}"#).unwrap();
    assert!(resolve_corpus(temp.path()).is_err());
}
#[test]
fn explicit_verification_checks_all_declared_payload_checksums_and_sizes() {
    let temp = scratch();
    let hash = hex::encode(Sha256::digest(b"payload"));
    let mut files = serde_json::Map::new();
    for name in REQUIRED_FILES.iter().copied().chain(["additional.bin"]) {
        std::fs::write(temp.path().join(name), b"payload").unwrap();
        files.insert(name.into(), serde_json::json!({"bytes":7,"sha256":hash}));
    }
    let manifest = serde_json::json!({"format_version":5,"corpus_id":"test","post_count":0,"tag_count":0,"nonzero_count":0,"feature_model":"binary_presence_and_signed_log_popularity","files":files});
    std::fs::write(
        temp.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    std::fs::write(temp.path().join("additional.bin"), b"corrupt").unwrap();
    let error = verify_corpus(temp.path()).err().unwrap().to_string();
    assert!(
        error.contains("checksum mismatch: additional.bin"),
        "{error}"
    );
    std::fs::write(temp.path().join("additional.bin"), b"short").unwrap();
    let error = verify_corpus(temp.path()).err().unwrap().to_string();
    assert!(
        error.contains("size mismatch for additional.bin"),
        "{error}"
    );
    assert!(validate_payload_name("../outside").is_err());
    assert!(validate_payload_name("/absolute").is_err());
    assert!(validate_payload_name("sub/file").is_err());
    assert!(validate_payload_name("sub\\\\file").is_err());
}
#[test]
fn manifest_file_map_still_rejects_duplicate_payload_names() {
    let temp = scratch();
    let digest = "0".repeat(64);
    let text = format!(
        r#"{{"format_version":5,"corpus_id":"duplicate-test","post_count":0,"tag_count":0,"nonzero_count":0,"feature_model":"binary_presence_and_signed_log_popularity","files":{{"features.safetensors":{{"bytes":0,"sha256":"{digest}"}},"features.safetensors":{{"bytes":0,"sha256":"{digest}"}}}}}}"#
    );
    std::fs::write(temp.path().join("manifest.json"), text).unwrap();
    let error = read_manifest(temp.path()).err().unwrap();
    assert!(
        format!("{error:#}").contains("Duplicate key 'features.safetensors'"),
        "{error:#}"
    );
}

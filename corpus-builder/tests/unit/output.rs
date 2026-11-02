use super::*;
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::RowAccessor,
};
use safetensors::{Dtype, SafeTensors};

fn scratch() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap()
}

type Fixture<'a> = (&'static str, Dtype, Vec<usize>, &'a [u8]);

fn write_fixture(path: &Path, group: &str, entries: &[Fixture<'_>]) -> Result<()> {
    let layouts = entries
        .iter()
        .map(|(name, dtype, shape, _)| TensorLayout {
            name,
            dtype: *dtype,
            shape: shape.clone(),
        })
        .collect();
    let mut output = StagedTensors::create(path, group, layouts)?;
    {
        let mut segments = output.segments()?;
        for (name, _, _, bytes) in entries {
            segments.get_mut(name).unwrap().append(bytes)?;
        }
    }
    output.finish()?;
    Ok(())
}

fn assert_fixture(path: &Path, entries: &[Fixture<'_>]) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let (header_length, metadata) = SafeTensors::read_metadata(&bytes).unwrap();
    assert_eq!(header_length % 8, 0);
    assert_eq!(tensors.len(), entries.len());
    let mut ranges = Vec::new();
    for (name, dtype, shape, expected) in entries {
        let tensor = tensors.tensor(name).unwrap();
        assert_eq!(tensor.dtype(), *dtype, "{name}");
        assert_eq!(tensor.shape(), shape, "{name}");
        assert_eq!(tensor.data(), *expected, "{name}");
        let info = metadata.info(name).unwrap();
        let (start, end) = info.data_offsets;
        assert_eq!(
            (8 + header_length + start) % (dtype.bitsize() / 8),
            0,
            "{name}"
        );
        ranges.push((start, end));
    }
    ranges.sort_unstable();
    let mut end = 0;
    for (start, next) in ranges {
        assert_eq!(start, end, "Tensor payload has a gap or overlap");
        end = next;
    }
    assert_eq!(8 + header_length + end, bytes.len());
    bytes
}

#[test]
fn official_reader_decodes_all_schema_fields_with_odd_post_counts() {
    let temp = scratch();
    let ids = [1i64, 256, i64::MAX].map(i64::to_le_bytes).concat();
    let popularity = [-2.5f64, 0.0, 40.0].map(f64::to_le_bytes).concat();
    let md5s: Vec<u8> = (0..48).collect();
    let offsets = [0i32, 0, 2, 3].map(i32::to_le_bytes).concat();
    let indices = [0i32, 1, 1].map(i32::to_le_bytes).concat();
    let tag_offsets = [0i32, 1, 3].map(i32::to_le_bytes).concat();
    let tag_posts = [1i32, 1, 2].map(i32::to_le_bytes).concat();
    let artist_vocab = [0i32, 1].map(i32::to_le_bytes).concat();
    let groups: [(&str, Vec<Fixture<'_>>); 3] = [
        (
            "features",
            vec![
                ("post_tag_offsets", Dtype::I32, vec![4], &offsets),
                ("post_tag_indices", Dtype::I32, vec![3], &indices),
            ],
        ),
        (
            "posts",
            vec![
                ("post_ids", Dtype::I64, vec![3], &ids),
                ("popularity", Dtype::F64, vec![3], &popularity),
                ("md5_bytes", Dtype::U8, vec![3, 16], &md5s),
                ("file_extension_codes", Dtype::U8, vec![3], &[1, 6, 7]),
                ("rating_codes", Dtype::U8, vec![3], &[1, 2, 3]),
            ],
        ),
        (
            "postings",
            vec![
                ("tag_post_offsets", Dtype::I32, vec![3], &tag_offsets),
                ("tag_post_indices", Dtype::I32, vec![3], &tag_posts),
                ("artist_vocab_indices", Dtype::I32, vec![2], &artist_vocab),
                ("ordinal_metadata_vocab_indices", Dtype::I32, vec![0], &[]),
                ("post_artist_offsets", Dtype::I32, vec![4], &offsets),
                ("post_artist_indices", Dtype::I32, vec![3], &indices),
            ],
        ),
    ];
    for (group, entries) in groups {
        let path = temp.path().join(group);
        write_fixture(&path, group, &entries).unwrap();
        let bytes = assert_fixture(&path, &entries);
        assert!(write_fixture(&path, group, &entries).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        let repeated = temp.path().join(format!("{group}-repeated"));
        write_fixture(&repeated, group, &entries).unwrap();
        assert_eq!(std::fs::read(repeated).unwrap(), bytes);
    }
}

#[test]
fn tensor_names_types_shapes_and_integer_bounds_are_checked_before_creation() {
    let temp = scratch();
    let path = temp.path().join("output");
    let invalid = [
        TensorLayout {
            name: "overflow",
            dtype: Dtype::I64,
            shape: vec![i64::MAX as usize],
        },
        TensorLayout {
            name: "shape_overflow",
            dtype: Dtype::U8,
            shape: vec![usize::MAX, 2],
        },
        TensorLayout {
            name: "unsupported",
            dtype: Dtype::U16,
            shape: vec![0],
        },
        TensorLayout {
            name: "__metadata__",
            dtype: Dtype::U8,
            shape: vec![0],
        },
        TensorLayout {
            name: " ",
            dtype: Dtype::U8,
            shape: vec![0],
        },
    ];
    for entry in invalid {
        assert!(StagedTensors::create(&path, "posts", vec![entry]).is_err());
        assert!(!path.exists());
    }
    let duplicate = (0..2)
        .map(|_| TensorLayout {
            name: "same",
            dtype: Dtype::U8,
            shape: vec![0],
        })
        .collect();
    assert!(StagedTensors::create(&path, "posts", duplicate).is_err());
    assert!(!path.exists());
}

#[test]
fn partial_overfilled_and_indexed_wrong_count_outputs_are_removed() {
    let temp = scratch();
    let path = temp.path().join("output");
    assert!(write_fixture(&path, "posts", &[("short", Dtype::I32, vec![2], &[0; 4])]).is_err());
    assert!(!path.exists());
    assert!(write_fixture(&path, "posts", &[("long", Dtype::U8, vec![0], &[0])]).is_err());
    assert!(!path.exists());
    for actual in [0, 2] {
        let mut output = StagedTensors::create(
            &path,
            "postings",
            vec![TensorLayout {
                name: "tag_post_indices",
                dtype: Dtype::I32,
                shape: vec![1],
            }],
        )
        .unwrap();
        {
            let mut segments = output.segments().unwrap();
            assert!(
                segments
                    .get_mut("tag_post_indices")
                    .unwrap()
                    .fill_i32(|values| {
                        values[0] = 7;
                        Ok(actual)
                    })
                    .is_err()
            );
        }
        assert!(output.finish().is_err());
        assert!(!path.exists());
    }
    {
        let mut output = StagedTensors::create(
            &path,
            "posts",
            vec![TensorLayout {
                name: "post_ids",
                dtype: Dtype::I64,
                shape: vec![1],
            }],
        )
        .unwrap();
        let mut segments = output.segments().unwrap();
        segments
            .get_mut("post_ids")
            .unwrap()
            .append(&[0; 4])
            .unwrap();
        // Dropping an unfinished mapping is also failure cleanup.
    }
    assert!(!path.exists());
}

#[test]
fn official_reader_accepts_empty_and_scalar_boundaries_without_gaps() {
    let temp = scratch();
    let path = temp.path().join("output");
    let scalar = (-1.25f32).to_le_bytes();
    let entries = [
        ("empty_start", Dtype::I64, vec![0, 8], &[][..]),
        ("scalar", Dtype::F32, vec![], &scalar[..]),
        ("empty_end", Dtype::I32, vec![3, 0], &[][..]),
    ];
    write_fixture(&path, "features", &entries).unwrap();
    assert_fixture(&path, &entries);
}

#[test]
fn parquet_columns_types_and_entry_order_are_preserved() {
    let temp = scratch();
    let path = temp.path().join("vocab.parquet");
    let entries = [
        VocabularyEntry {
            index: 1,
            source_id: i64::MAX,
            name: "狐\nname".into(),
            category: 5,
            occurrences: 9000000000,
        },
        VocabularyEntry {
            index: 0,
            source_id: 9,
            name: "artist".into(),
            category: 1,
            occurrences: 50,
        },
    ];
    write_vocabulary(&path, &entries).unwrap();
    let reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    let schema = reader.metadata().file_metadata().schema_descr();
    assert_eq!(
        schema
            .columns()
            .iter()
            .map(|column| column.name())
            .collect::<Vec<_>>(),
        ["idx", "tag_id", "name", "category", "occ"]
    );
    assert!(
        schema
            .columns()
            .iter()
            .all(|column| column.max_def_level() == 0)
    );
    use parquet::basic::LogicalType;
    assert_eq!(
        schema
            .columns()
            .iter()
            .map(|column| column.logical_type())
            .collect::<Vec<_>>(),
        [
            Some(LogicalType::Integer {
                bit_width: 32,
                is_signed: true
            }),
            Some(LogicalType::Integer {
                bit_width: 64,
                is_signed: true
            }),
            Some(LogicalType::String),
            Some(LogicalType::Integer {
                bit_width: 32,
                is_signed: true
            }),
            Some(LogicalType::Integer {
                bit_width: 64,
                is_signed: true
            })
        ]
    );
    let rows = reader
        .get_row_iter(None)
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(rows.len(), entries.len());
    for (row, entry) in rows.iter().zip(&entries) {
        assert_eq!(row.get_int(0).unwrap(), entry.index);
        assert_eq!(row.get_long(1).unwrap(), entry.source_id);
        assert_eq!(row.get_string(2).unwrap(), &entry.name);
        assert_eq!(row.get_int(3).unwrap(), entry.category);
        assert_eq!(row.get_long(4).unwrap(), entry.occurrences);
    }
    assert!(write_vocabulary(&path, &entries).is_err());
    let path = temp.path().join("aliases.parquet");
    write_aliases(
        &path,
        &[AliasEntry {
            alias: "old狐".into(),
            canonical_index: 1,
        }],
    )
    .unwrap();
    let reader = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let schema = reader.metadata().file_metadata().schema_descr();
    assert_eq!(
        schema
            .columns()
            .iter()
            .map(|column| column.name())
            .collect::<Vec<_>>(),
        ["alias", "canonical_idx"]
    );
    assert!(
        schema
            .columns()
            .iter()
            .all(|column| column.max_def_level() == 0)
    );
    assert_eq!(
        schema
            .columns()
            .iter()
            .map(|column| column.logical_type())
            .collect::<Vec<_>>(),
        [
            Some(LogicalType::String),
            Some(LogicalType::Integer {
                bit_width: 32,
                is_signed: true
            })
        ]
    );
    let row = reader.get_row_iter(None).unwrap().next().unwrap().unwrap();
    assert_eq!(row.get_string(0).unwrap(), "old狐");
    assert_eq!(row.get_int(1).unwrap(), 1);
}

#[test]
fn parquet_writer_rollover_preserves_every_vocabulary_and_alias_entry() {
    const ENTRY_COUNT: usize = 65_537;
    let temp = scratch();
    let vocabulary_path = temp.path().join("vocab.parquet");
    let aliases_path = temp.path().join("aliases.parquet");
    // Each column changes across rows 65535/65536, including the final partial
    // batch. Unique strings and wide integers expose stale reusable buffers.
    let vocabulary: Vec<_> = (0..ENTRY_COUNT)
        .map(|index| VocabularyEntry {
            index: index as i32,
            source_id: 1_000_000_000_000 + 17 * index as i64,
            name: format!("tag_{index:05}"),
            category: [0, 1, 3, 4, 5, 7][index % 6],
            occurrences: 9_000_000_000 + 31 * index as i64,
        })
        .collect();
    let aliases: Vec<_> = (0..ENTRY_COUNT)
        .map(|index| AliasEntry {
            alias: format!("alias_{index:05}"),
            canonical_index: (ENTRY_COUNT - 1 - index) as i32,
        })
        .collect();
    write_vocabulary(&vocabulary_path, &vocabulary).unwrap();
    write_aliases(&aliases_path, &aliases).unwrap();

    let reader = SerializedFileReader::new(File::open(&vocabulary_path).unwrap()).unwrap();
    let mut rows = reader.get_row_iter(None).unwrap();
    for entry in &vocabulary {
        let row = rows.next().unwrap().unwrap();
        assert_eq!(row.get_int(0).unwrap(), entry.index);
        assert_eq!(row.get_long(1).unwrap(), entry.source_id);
        assert_eq!(row.get_string(2).unwrap(), &entry.name);
        assert_eq!(row.get_int(3).unwrap(), entry.category);
        assert_eq!(row.get_long(4).unwrap(), entry.occurrences);
    }
    assert!(rows.next().is_none());

    let reader = SerializedFileReader::new(File::open(&aliases_path).unwrap()).unwrap();
    let mut rows = reader.get_row_iter(None).unwrap();
    for entry in &aliases {
        let row = rows.next().unwrap().unwrap();
        assert_eq!(row.get_string(0).unwrap(), &entry.alias);
        assert_eq!(row.get_int(1).unwrap(), entry.canonical_index);
    }
    assert!(rows.next().is_none());

    // Match the catalog's focused fixture pattern: declare the required tensor
    // payloads, but only generate/open Parquet, not a complete corpus.
    use sha2::{Digest, Sha256};
    let mut files = serde_json::Map::new();
    for name in [
        "features.safetensors",
        "posts.safetensors",
        "postings.safetensors",
    ] {
        files.insert(
            name.into(),
            serde_json::json!({"bytes": 0, "sha256": hex::encode(Sha256::digest([]))}),
        );
    }
    for path in [&vocabulary_path, &aliases_path] {
        let bytes = std::fs::read(path).unwrap();
        files.insert(
            path.file_name().unwrap().to_str().unwrap().into(),
            serde_json::json!({
                "bytes": bytes.len(),
                "sha256": hex::encode(Sha256::digest(&bytes)),
            }),
        );
    }
    let manifest = serde_json::json!({
        "format_version": 5,
        "corpus_id": "parquet-writer-rollover",
        "post_count": 0,
        "tag_count": ENTRY_COUNT,
        "nonzero_count": 0,
        "feature_model": "binary_presence_and_signed_log_popularity",
        "files": files,
    });
    std::fs::write(
        temp.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let catalog = ff_core::catalog::TagCatalog::load(temp.path()).unwrap();
    assert_eq!(catalog.len(), ENTRY_COUNT);
    for entry in &vocabulary {
        let index = entry.index as usize;
        assert_eq!(catalog.resolve(&entry.name), Some(index));
        let tag = catalog.tag(index);
        assert_eq!(tag.name, entry.name);
        assert_eq!(tag.category, entry.category);
        assert_eq!(tag.post_count, entry.occurrences);
    }
    for entry in &aliases {
        assert_eq!(
            catalog.resolve(&entry.alias),
            Some(entry.canonical_index as usize)
        );
    }
}

#[test]
fn empty_parquet_outputs_are_readable() {
    let temp = scratch();
    for (name, vocabulary) in [("vocab", true), ("aliases", false)] {
        let path = temp.path().join(name);
        if vocabulary {
            write_vocabulary(&path, &[]).unwrap();
        } else {
            write_aliases(&path, &[]).unwrap();
        }
        let reader = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
        assert_eq!(reader.metadata().file_metadata().num_rows(), 0);
        assert!(reader.get_row_iter(None).unwrap().next().is_none());
    }
}

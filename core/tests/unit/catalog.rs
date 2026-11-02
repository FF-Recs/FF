use super::*;
fn tag(name: &str, count: i64) -> Tag {
    Tag {
        name: name.into(),
        category: 0,
        post_count: count,
    }
}
#[test]
fn terminal_alias_normalization_preserves_global_indices() {
    let catalog = TagCatalog::fixture(
        vec![tag("zebra", 1), tag("blue_fox", 10)],
        vec![("old_name".into(), 1)],
    )
    .unwrap();
    assert_eq!(catalog.resolve(" +OLD-NAME "), Some(1));
    assert_eq!(catalog.resolve("-BLUE-FOX"), Some(1));
    assert_eq!(catalog.tag(0).name, "zebra");
    assert_eq!(catalog.resolve("missing"), None);
}
#[test]
fn alias_conflicts_and_bad_indices_fail() {
    assert!(TagCatalog::fixture(vec![tag("a", 1), tag("b", 1)], vec![("a".into(), 1)]).is_err());
    assert!(TagCatalog::fixture(vec![tag("a", 1)], vec![("b".into(), 1)]).is_err());
    assert!(TagCatalog::fixture(vec![tag("A", 1), tag("a", 1)], vec![]).is_err());
}
#[test]
fn search_ranking_is_bounded_and_not_alias_expansion() {
    let catalog = TagCatalog::fixture(
        vec![
            tag("fox", 1),
            tag("fox_tail", 2),
            tag("fox_ears", 50),
            tag("arctic_fox", 100),
        ],
        vec![("dog".into(), 0)],
    )
    .unwrap();
    let names = |limit| {
        catalog
            .search("FOX", limit)
            .into_iter()
            .map(|tag| tag.name)
            .collect::<Vec<_>>()
    };
    assert_eq!(names(4), ["fox", "fox_ears", "fox_tail", "arctic_fox"]);
    assert_eq!(names(2), ["fox", "fox_ears"]);
    assert!(catalog.search("dog", 4).is_empty());
    assert!(catalog.search("", 4).is_empty());
    assert!(catalog.search("fox", 0).is_empty());
}
#[test]
fn parquet_field_types_are_not_coerced_or_defaulted() {
    let row = Row::new(vec![
        ("idx".into(), Field::Long(1)),
        ("name".into(), Field::Null),
    ]);
    assert!(integer(&row, "idx").is_err());
    assert!(string(&row, "name").is_err());
    assert!(long(&row, "missing").is_err());
}
#[test]
fn parquet_loading_preserves_indices_across_row_groups_and_terminal_aliases() {
    use parquet::{
        data_type::{ByteArray, ByteArrayType, Int32Type, Int64Type},
        file::{properties::WriterProperties, writer::SerializedFileWriter},
        schema::parser::parse_message_type,
    };
    use sha2::{Digest, Sha256};
    use std::sync::Arc;
    let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.workdir/ff-rs/corpus");
    std::fs::create_dir_all(&scratch).unwrap();
    let temp = tempfile::tempdir_in(scratch).unwrap();
    let schema = Arc::new(parse_message_type("message vocabulary { REQUIRED INT32 idx; REQUIRED BINARY name (UTF8); REQUIRED INT32 category; REQUIRED INT64 occ; }").unwrap());
    let mut writer = SerializedFileWriter::new(
        File::create(temp.path().join("vocab.parquet")).unwrap(),
        schema,
        Arc::new(WriterProperties::default()),
    )
    .unwrap();
    // On-disk row order is not vocabulary order and row groups need not follow indices.
    for (index, name, category, count) in [(1, "blue_fox", 5, 100i64), (0, "artist_name", 1, 50)] {
        let mut group = writer.next_row_group().unwrap();
        let mut column = group.next_column().unwrap().unwrap();
        column
            .typed::<Int32Type>()
            .write_batch(&[index], None, None)
            .unwrap();
        column.close().unwrap();
        let mut column = group.next_column().unwrap().unwrap();
        column
            .typed::<ByteArrayType>()
            .write_batch(&[ByteArray::from(name)], None, None)
            .unwrap();
        column.close().unwrap();
        let mut column = group.next_column().unwrap().unwrap();
        column
            .typed::<Int32Type>()
            .write_batch(&[category], None, None)
            .unwrap();
        column.close().unwrap();
        let mut column = group.next_column().unwrap().unwrap();
        column
            .typed::<Int64Type>()
            .write_batch(&[count], None, None)
            .unwrap();
        column.close().unwrap();
        group.close().unwrap();
    }
    writer.close().unwrap();
    let schema = Arc::new(
        parse_message_type(
            "message aliases { REQUIRED BINARY alias (UTF8); REQUIRED INT32 canonical_idx; }",
        )
        .unwrap(),
    );
    let mut writer = SerializedFileWriter::new(
        File::create(temp.path().join("aliases.parquet")).unwrap(),
        schema,
        Arc::new(WriterProperties::default()),
    )
    .unwrap();
    let mut group = writer.next_row_group().unwrap();
    let mut column = group.next_column().unwrap().unwrap();
    column
        .typed::<ByteArrayType>()
        .write_batch(
            &[ByteArray::from("old_fox"), ByteArray::from("ancient_fox")],
            None,
            None,
        )
        .unwrap();
    column.close().unwrap();
    let mut column = group.next_column().unwrap().unwrap();
    column
        .typed::<Int32Type>()
        .write_batch(&[1, 1], None, None)
        .unwrap();
    column.close().unwrap();
    group.close().unwrap();
    writer.close().unwrap();
    let mut files = serde_json::Map::new();
    for name in [
        "features.safetensors",
        "posts.safetensors",
        "postings.safetensors",
        "vocab.parquet",
        "aliases.parquet",
    ] {
        let bytes = std::fs::read(temp.path().join(name)).unwrap_or_default();
        files.insert(
            name.into(),
            serde_json::json!({"bytes":bytes.len(),"sha256":hex::encode(Sha256::digest(&bytes))}),
        );
    }
    let mut manifest = serde_json::json!({"format_version":5,"corpus_id":"test","post_count":0,"tag_count":2,"nonzero_count":0,"feature_model":"binary_presence_and_signed_log_popularity","files":files});
    std::fs::write(
        temp.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let catalog = TagCatalog::load(temp.path()).unwrap();
    assert_eq!(catalog.len(), 2);
    assert_eq!(catalog.tag(0).name, "artist_name");
    assert_eq!(catalog.tag(1).name, "blue_fox");
    assert_eq!(catalog.resolve("ANCIENT-FOX"), Some(1));
    assert_eq!(catalog.resolve("old_fox"), Some(1));
    assert_eq!(catalog.search("blue", 1)[0].category, "species");
    assert_eq!(catalog.search("blue", 1)[0].post_count, 100);
    manifest["tag_count"] = serde_json::json!(3);
    std::fs::write(
        temp.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    assert!(
        TagCatalog::load(temp.path()).is_err(),
        "vocabulary cannot silently omit declared rows"
    );
}

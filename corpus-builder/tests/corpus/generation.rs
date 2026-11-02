use super::support::{fixture, fixture_score, gzip};
use ff_core::{
    catalog::TagCatalog,
    corpus::{Corpus, verify_corpus},
};
use ff_corpus_builder::build;
use std::fs;

#[test]
fn extraction_filters_flash_before_thresholds_closure_and_artist_membership() {
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let options = fixture(root.path());
    let result = build(&options).unwrap();
    verify_corpus(&result.release_path).unwrap();
    let corpus = Corpus::open(&result.release_path).unwrap();
    let catalog = TagCatalog::load(&result.release_path).unwrap();
    assert_eq!(corpus.post_ids(), &(1..=60).chain([62]).collect::<Vec<_>>());
    assert_eq!(result.post_count, 61);
    assert!(corpus.post_ids().binary_search(&61).is_err());
    assert_eq!(
        (54..60)
            .map(|row| corpus.extension(row))
            .collect::<Vec<_>>(),
        ["png", "gif", "webm", "mp4", "webp", "jpg"]
    );
    let post_bytes = fs::read(result.release_path.join("posts.safetensors")).unwrap();
    let post_tensors = safetensors::SafeTensors::deserialize(&post_bytes).unwrap();
    let popularity = post_tensors.tensor("popularity").unwrap();
    assert_eq!(popularity.dtype(), safetensors::Dtype::F64);
    assert_eq!(popularity.shape(), [corpus.post_count]);
    for (row, (&post_id, &value)) in corpus
        .post_ids()
        .iter()
        .zip(corpus.popularities())
        .enumerate()
    {
        // Convert first, then take the magnitude: signed integer abs would overflow
        // for MIN and integer-space transforms would differ above f64 precision.
        let score = fixture_score(post_id) as f64;
        let expected = score.signum() * score.abs().ln_1p();
        assert_eq!(value.to_bits(), expected.to_bits(), "post {post_id}");
        assert_eq!(corpus.popularity(row).to_bits(), expected.to_bits());
        assert_eq!(
            &popularity.data()[row * 8..(row + 1) * 8],
            &expected.to_le_bytes(),
        );
    }
    assert_eq!(
        &post_tensors.tensor("file_extension_codes").unwrap().data()[54..60],
        &[2, 3, 4, 6, 7, 1]
    );
    let fox = catalog.resolve("fox").unwrap();
    let canine = catalog.resolve("canine").unwrap();
    assert_eq!(catalog.resolve("alias_old"), Some(fox));
    assert_eq!(catalog.tag(fox).post_count, 50);
    assert_eq!(catalog.tag(canine).post_count, 50);
    assert!(catalog.resolve("rare_content").is_none());
    assert!(catalog.resolve("rare_artist").is_none());
    // Expectations are literal policy examples, independent of PSEUDO_ARTISTS.
    let pseudo_artists = [
        "conditional_dnp",
        "third-party_edit",
        "sound_warning",
        "ai_assisted",
        "ai_generated_background",
        "unknown_artist",
        "anonymous_artist",
    ];
    for name in pseudo_artists {
        let global = catalog.resolve(name).unwrap();
        assert_eq!(catalog.tag(global).post_count, 10);
        assert!(!corpus.artist_vocab_indices().contains(&(global as i32)));
    }
    assert_eq!(corpus.artist_count, 2);
    assert_eq!(
        catalog.tag(catalog.resolve("artist_a").unwrap()).post_count,
        10
    );
    assert_eq!(
        catalog.tag(catalog.resolve("artist_b").unwrap()).post_count,
        12
    );
    for (row, &id) in corpus.post_ids().iter().enumerate() {
        let mut actual = corpus
            .post_artists(row)
            .iter()
            .map(|&local| {
                catalog
                    .tag(corpus.artist_vocab_indices()[local as usize] as usize)
                    .name
                    .as_str()
            })
            .collect::<Vec<_>>();
        actual.sort_unstable();
        let expected = match id {
            1..=5 => vec!["artist_a"],
            6..=10 => vec!["artist_a", "artist_b"],
            11..=17 => vec!["artist_b"],
            _ => vec![],
        };
        assert_eq!(actual, expected, "post {id}");
    }
    for (category, admitted, rejected) in [
        (3, "copyright50", "copyright49"),
        (4, "character50", "character49"),
        (5, "species50", "species49"),
    ] {
        let tag = catalog.resolve(admitted).unwrap();
        assert_eq!(
            (catalog.tag(tag).category, catalog.tag(tag).post_count),
            (category, 50)
        );
        assert_eq!(corpus.tag_posts(tag), &(0..50).collect::<Vec<_>>());
        assert!(catalog.resolve(rejected).is_none());
    }
    let metadata = corpus.ordinal_metadata_vocab_indices();
    assert_eq!(metadata.len(), 2);
    assert_eq!(
        metadata,
        &(0..catalog.len())
            .filter(|&tag| catalog.tag(tag).category == 7)
            .map(|tag| tag as i32)
            .collect::<Vec<_>>()
    );
    for name in ["supported_metadata", "implied_metadata"] {
        let tag = catalog.resolve(name).unwrap();
        assert_eq!(catalog.tag(tag).category, 7);
        assert_eq!(catalog.tag(tag).post_count, 50);
        assert!(metadata.contains(&(tag as i32)));
        assert!(!corpus.artist_vocab_indices().contains(&(tag as i32)));
        let start = corpus.tag_post_offsets()[tag] as usize;
        let end = corpus.tag_post_offsets()[tag + 1] as usize;
        assert_eq!(
            &corpus.tag_post_indices()[start..end],
            &(0..50).collect::<Vec<_>>()
        );
        for row in 0..corpus.post_count {
            assert_eq!(
                corpus
                    .post_tags(row)
                    .iter()
                    .filter(|&&t| t == tag as i32)
                    .count(),
                usize::from(row < 50)
            );
        }
    }
    let supported = catalog.resolve("supported_metadata").unwrap();
    assert_eq!(catalog.resolve("metadata_alias"), Some(supported));
    assert_eq!(catalog.resolve("metadata_alias_chain"), Some(supported));
    assert!((0..catalog.len()).all(|tag| {
        !["metadata_alias", "metadata_alias_chain"].contains(&catalog.tag(tag).name.as_str())
    }));
    assert!(catalog.resolve("boundary49_metadata").is_none());
    assert!(catalog.resolve("unused_metadata").is_none());
    assert!(catalog.resolve("animated").is_none());
    assert!(catalog.resolve("unsupported_category").is_none());
    for name in [
        "flash_only_content",
        "flash_only_artist",
        "flash_only_metadata",
    ] {
        assert!(catalog.resolve(name).is_none());
    }
    let mut first_names = corpus
        .post_tags(0)
        .iter()
        .map(|&tag| catalog.tag(tag as usize).name.as_str())
        .collect::<Vec<_>>();
    first_names.sort_unstable();
    assert_eq!(
        first_names,
        [
            "ai_assisted",
            "ai_generated_background",
            "anonymous_artist",
            "artist_a",
            "canine",
            "character50",
            "conditional_dnp",
            "copyright50",
            "fox",
            "implied_metadata",
            "sound_warning",
            "species50",
            "supported_metadata",
            "third-party_edit",
            "unknown_artist",
        ]
    );
    assert_eq!(result.active_aliases, 4);
    assert_eq!(result.active_implications, 3);
}

#[test]
fn admitted_post_ids_and_unknown_extensions_remain_strict_and_failures_clean_up() {
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let options = fixture(root.path());
    for rows in [
        "1,00000000000000000000000000000001,s,jpeg,f,,0\n",
        "1,00000000000000000000000000000001,s,SWF,f,,0\n",
        "0,00000000000000000000000000000001,s,jpg,f,,0\n",
        "-1,00000000000000000000000000000001,s,jpg,f,,0\n",
        "invalid,00000000000000000000000000000001,s,jpg,f,,0\n",
        "1,00000000000000000000000000000001,s,jpg,f,,0\n1,00000000000000000000000000000001,s,jpg,f,,0\n",
        "2,00000000000000000000000000000002,s,jpg,f,,0\n1,00000000000000000000000000000001,s,jpg,f,,0\n",
    ] {
        gzip(
            &options.posts,
            &format!("id,md5,rating,file_ext,is_deleted,tag_string,score\n{rows}"),
        );
        assert!(build(&options).is_err());
        assert_eq!(fs::read_dir(&options.releases).unwrap().count(), 0);
        assert_eq!(fs::read_dir(&options.work).unwrap().count(), 0);
    }
}

#[test]
fn admitted_scores_require_checked_int64_values_and_a_required_header() {
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let options = fixture(root.path());
    for score in [
        "",
        "NaN",
        "1.5",
        "9223372036854775808",
        "-9223372036854775809",
    ] {
        gzip(
            &options.posts,
            &format!(
                "id,md5,rating,file_ext,is_deleted,tag_string,score\n1,00000000000000000000000000000001,s,jpg,f,,{score}\n"
            ),
        );
        assert!(build(&options).is_err());
        assert_eq!(fs::read_dir(&options.releases).unwrap().count(), 0);
        assert_eq!(fs::read_dir(&options.work).unwrap().count(), 0);
    }
    gzip(
        &options.posts,
        "id,md5,rating,file_ext,is_deleted,tag_string\n1,00000000000000000000000000000001,s,jpg,f,\n",
    );
    let error = build(&options).err().unwrap();
    assert!(format!("{error:#}").contains("score"), "{error:#}");
    assert_eq!(fs::read_dir(&options.releases).unwrap().count(), 0);
    assert_eq!(fs::read_dir(&options.work).unwrap().count(), 0);
}

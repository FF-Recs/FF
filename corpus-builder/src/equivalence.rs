//! Migration acceptance: compare logical data independently of writer encodings.
use crate::{build::sha256, contract};
use anyhow::{Context, Result, ensure};
use ff_core::{
    catalog::TagCatalog,
    corpus::{Corpus, verify_corpus},
};
use memmap2::Mmap;
use parquet::file::reader::{FileReader, SerializedFileReader};
use safetensors::SafeTensors;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs::File, path::Path, time::Instant};

fn tensor_comparison(reference: &Path, candidate: &Path) -> Result<Value> {
    // Tensor declarations are optional FF provenance. The official container
    // metadata, not manifest presence, is the authority for actual tensor names.
    let a = File::open(reference)?;
    let b = File::open(candidate)?;
    // Both immutable releases were verified before mapping; maps outlive metadata.
    let a = unsafe { Mmap::map(&a) }?;
    let b = unsafe { Mmap::map(&b) }?;
    let (a_length, a_metadata) = SafeTensors::read_metadata(&a)?;
    let (b_length, b_metadata) = SafeTensors::read_metadata(&b)?;
    ensure!(
        a_metadata.metadata() == b_metadata.metadata(),
        "Tensor metadata differs"
    );
    let a_tensors = a_metadata.tensors();
    let b_tensors = b_metadata.tensors();
    ensure!(
        a_tensors.len() == b_tensors.len(),
        "Tensor name sets differ: {}",
        candidate.display()
    );
    let a_start = a_length.checked_add(8).context("Header overflow")?;
    let b_start = b_length.checked_add(8).context("Header overflow")?;
    let mut compared = BTreeMap::new();
    for (name, descriptor) in a_tensors {
        let other = b_tensors
            .get(&name)
            .with_context(|| format!("Missing candidate tensor '{name}'"))?;
        ensure!(
            descriptor.dtype == other.dtype && descriptor.shape == other.shape,
            "Tensor type/shape differs: {name}"
        );
        let left = &a[a_start + descriptor.data_offsets.0..a_start + descriptor.data_offsets.1];
        let right = &b[b_start + other.data_offsets.0..b_start + other.data_offsets.1];
        ensure!(
            left == right,
            "Tensor values differ: {name}; first differing byte {:?}",
            left.iter().zip(right).position(|(l, r)| l != r)
        );
        compared.insert(
            name,
            json!({"dtype":descriptor.dtype,"shape":descriptor.shape,"compared_bytes":left.len()}),
        );
    }
    Ok(json!(compared))
}

fn parquet_comparison(reference: &Path, candidate: &Path) -> Result<Value> {
    let a = SerializedFileReader::new(File::open(reference)?)?;
    let b = SerializedFileReader::new(File::open(candidate)?)?;
    let a_schema = a.metadata().file_metadata().schema_descr();
    let b_schema = b.metadata().file_metadata().schema_descr();
    ensure!(
        a_schema.num_columns() == b_schema.num_columns(),
        "Parquet column count differs"
    );
    for (left, right) in a_schema.columns().iter().zip(b_schema.columns()) {
        ensure!(
            left.path() == right.path()
                && left.physical_type() == right.physical_type()
                && left.logical_type() == right.logical_type(),
            "Parquet column name/type differs: {}",
            candidate.display()
        );
    }
    ensure!(
        a.metadata().file_metadata().num_rows() == b.metadata().file_metadata().num_rows(),
        "Parquet row count differs"
    );
    let mut left = a.get_row_iter(None)?;
    let mut right = b.get_row_iter(None)?;
    let mut rows = 0usize;
    while let Some(row) = left.next() {
        let other = right.next().context("Candidate Parquet ended early")??;
        ensure!(
            row? == other,
            "Parquet row {rows} differs: {}",
            candidate.display()
        );
        rows += 1;
    }
    ensure!(right.next().is_none(), "Candidate Parquet has extra rows");
    Ok(json!({"rows":rows,"columns":a_schema.num_columns(),"ordered_typed_values_equal":true}))
}

fn independent_relations(corpus: &Corpus, catalog: &TagCatalog) -> Result<Value> {
    // Consume each sorted inverted row in forward-post order. No hash approximation,
    // pair sampling, or second giant CSR allocation is needed for exact transpose proof.
    let mut cursors = corpus.tag_post_offsets()[..corpus.tag_count].to_vec();
    let mut artist_by_vocab = vec![-1i32; corpus.tag_count];
    let expected_artist_vocab = (0..catalog.len())
        .filter(|&i| {
            let tag = catalog.tag(i);
            tag.category == 1 && !contract::PSEUDO_ARTISTS.contains(&tag.name.as_str())
        })
        .map(|i| i as i32)
        .collect::<Vec<_>>();
    ensure!(
        corpus.artist_vocab_indices() == expected_artist_vocab,
        "Artist vocabulary policy differs"
    );
    for (local, &global) in expected_artist_vocab.iter().enumerate() {
        artist_by_vocab[global as usize] = local as i32;
    }
    let expected_metadata = (0..catalog.len())
        .filter(|&i| catalog.tag(i).category == 7)
        .map(|i| i as i32)
        .collect::<Vec<_>>();
    ensure!(
        corpus.ordinal_metadata_vocab_indices() == expected_metadata,
        "Metadata vocabulary indices differ"
    );
    let mut links = 0u64;
    let mut artist_links = 0u64;
    for row in 0..corpus.post_count {
        let artists = corpus.post_artists(row);
        let mut artist_position = 0usize;
        for &tag in corpus.post_tags(row) {
            let tag = tag as usize;
            let cursor = &mut cursors[tag];
            ensure!(
                *cursor < corpus.tag_post_offsets()[tag + 1]
                    && corpus.tag_post_indices()[*cursor as usize] == row as i32,
                "Forward/inverted transpose differs at post row {row}, tag {tag}"
            );
            *cursor += 1;
            links += 1;
            let artist = artist_by_vocab[tag];
            if artist >= 0 {
                ensure!(
                    artists.get(artist_position) == Some(&artist),
                    "Post/artist relation differs at row {row}"
                );
                artist_position += 1;
                artist_links += 1;
            }
        }
        ensure!(
            artist_position == artists.len(),
            "Post/artist relation contains extra membership at row {row}"
        );
    }
    ensure!(
        cursors == corpus.tag_post_offsets()[1..],
        "Inverted relation contains extra memberships"
    );
    for tag in 0..corpus.tag_count {
        let count = i64::from(corpus.tag_post_offsets()[tag + 1] - corpus.tag_post_offsets()[tag]);
        ensure!(
            catalog.tag(tag).post_count == count,
            "Vocabulary occurrence count differs at tag {tag}"
        );
        let definition = catalog.tag(tag);
        if definition.category == 7 {
            ensure!(
                count >= contract::MINIMUM_METADATA_POSTS,
                "Metadata tag '{}' has insufficient normalized occurrences: {count}",
                definition.name
            );
        }
    }
    Ok(
        json!({"forward_inverted_exact_transpose":true,"vocabulary_occurrences_equal_actual_memberships":true,
        "artist_relation_exact":true,"metadata_supported":true,"links_checked":links,"artist_links_checked":artist_links}),
    )
}

pub fn compare(reference: &Path, candidate: &Path) -> Result<Value> {
    let started = Instant::now();
    eprintln!("[compare] verifying all reference and candidate payloads");
    verify_corpus(reference)?;
    verify_corpus(candidate)?;
    let a: Value = serde_json::from_reader(File::open(reference.join("manifest.json"))?)?;
    let b: Value = serde_json::from_reader(File::open(candidate.join("manifest.json"))?)?;
    for field in [
        "format_version",
        "post_count",
        "tag_count",
        "nonzero_count",
        "feature_model",
        "normalization",
        "popularity",
        "sources",
    ] {
        ensure!(
            a[field] == b[field],
            "Manifest logical field differs: {field}"
        );
    }
    ensure!(
        a["files"]
            .as_object()
            .context("Manifest files")?
            .keys()
            .eq(b["files"].as_object().context("Manifest files")?.keys()),
        "Declared release file sets differ"
    );
    let mut tensors = BTreeMap::new();
    for filename in [
        "features.safetensors",
        "posts.safetensors",
        "postings.safetensors",
    ] {
        eprintln!("[compare] comparing every logical tensor byte in {filename}");
        tensors.insert(
            filename,
            tensor_comparison(&reference.join(filename), &candidate.join(filename))?,
        );
    }
    let mut parquet = BTreeMap::new();
    for filename in ["vocab.parquet", "aliases.parquet"] {
        parquet.insert(
            filename,
            parquet_comparison(&reference.join(filename), &candidate.join(filename))?,
        );
    }
    // Integrity verification above binds each declaration to the actual file.
    // Only the five core payloads have semantic comparators; all other declared
    // payloads must retain their exact size and digest, regardless of their type.
    let mut extra_payloads = BTreeMap::new();
    for (filename, entry) in a["files"].as_object().context("Manifest files")? {
        if matches!(
            filename.as_str(),
            "features.safetensors"
                | "posts.safetensors"
                | "postings.safetensors"
                | "vocab.parquet"
                | "aliases.parquet"
        ) {
            continue;
        }
        let other = &b["files"][filename];
        ensure!(
            entry["bytes"] == other["bytes"] && entry["sha256"] == other["sha256"],
            "Declared extra payload bytes differ: {filename}"
        );
        extra_payloads.insert(
            filename,
            json!({"bytes":entry["bytes"],"sha256":entry["sha256"],"size_and_sha256_equal":true}),
        );
    }
    eprintln!(
        "[compare] independently checking transpose, occurrence and artist/metadata identities"
    );
    let corpus = Corpus::open(candidate)?;
    let catalog = TagCatalog::load(candidate)?;
    let relations = independent_relations(&corpus, &catalog)?;
    Ok(
        json!({"status":"equivalent","equivalence":"exact ordered logical core contents and exact extra payload bytes; core serialization bytes and release identities may differ",
        "reference_manifest_sha256":sha256(&reference.join("manifest.json"))?,
        "candidate_manifest_sha256":sha256(&candidate.join("manifest.json"))?,
        "reference_corpus_id":a["corpus_id"],"candidate_corpus_id":b["corpus_id"],
        "post_count":corpus.post_count,"tag_count":corpus.tag_count,"nonzero_count":corpus.post_tag_indices().len(),
        "tensors":tensors,"parquet":parquet,"extra_payloads":extra_payloads,"independent_relations":relations,"elapsed_seconds":started.elapsed().as_secs_f64()}),
    )
}

use super::support::{fixture, mutations};
use ff_core::{
    catalog::TagCatalog,
    corpus::{Corpus, verify_corpus},
};
use ff_corpus_builder::{build, build::sha256, equivalence};
use std::{fs, io::Write};

#[test]
fn comparison_accepts_optional_manifest_tensor_declarations() {
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let result = build(&fixture(root.path())).unwrap();
    let stage = root.path().join("optional-stage");
    fs::create_dir(&stage).unwrap();
    for entry in fs::read_dir(&result.release_path).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), stage.join(entry.file_name())).unwrap();
    }
    let manifest_path = stage.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["files"]["features.safetensors"]
        .as_object_mut()
        .unwrap()
        .remove("tensors");
    manifest["files"]["posts.safetensors"]["tensors"] = serde_json::Value::Null;
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let releases = root.path().join("optional-releases");
    fs::create_dir(&releases).unwrap();
    let optional = releases.join(sha256(&manifest_path).unwrap());
    fs::rename(&stage, &optional).unwrap();
    verify_corpus(&optional).unwrap();
    let comparison = equivalence::compare(&result.release_path, &optional).unwrap();
    assert_eq!(comparison["status"], "equivalent");
    assert_ne!(
        comparison["reference_manifest_sha256"],
        comparison["candidate_manifest_sha256"]
    );
}

#[test]
fn comparison_accepts_distinct_aligned_supported_container_layouts() {
    use safetensors::{SafeTensors, tensor::TensorInfo};
    use sha2::{Digest, Sha256};

    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let result = build(&fixture(root.path())).unwrap();
    let original_manifest_hash = sha256(&result.release_path.join("manifest.json")).unwrap();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(result.release_path.join("manifest.json")).unwrap())
            .unwrap();
    let stage = root.path().join("retained-stage");
    fs::create_dir(&stage).unwrap();
    for name in manifest["files"].as_object().unwrap().keys() {
        fs::copy(result.release_path.join(name), stage.join(name)).unwrap();
    }
    let path = stage.join("posts.safetensors");
    let original = fs::read(&path).unwrap();
    let tensors = SafeTensors::deserialize(&original).unwrap();
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".into(),
        serde_json::json!({"ff_format_version":"5","ff_group":"posts"}),
    );
    let mut payload = Vec::new();
    // Keep the I64 IDs and F64 popularity aligned first, but change the physical
    // order of the U8 columns. Both layouts support direct borrows with 61 posts.
    for name in [
        "post_ids",
        "popularity",
        "rating_codes",
        "md5_bytes",
        "file_extension_codes",
    ] {
        let tensor = tensors.tensor(name).unwrap();
        let start = payload.len();
        payload.extend_from_slice(tensor.data());
        header.insert(
            name.into(),
            serde_json::to_value(TensorInfo {
                dtype: tensor.dtype(),
                shape: tensor.shape().to_vec(),
                data_offsets: (start, payload.len()),
            })
            .unwrap(),
        );
    }
    let mut encoded_header = serde_json::to_vec(&header).unwrap();
    encoded_header.resize((encoded_header.len() + 7) & !7, b' ');
    let mut output = fs::File::create(&path).unwrap();
    output
        .write_all(&(encoded_header.len() as u64).to_le_bytes())
        .unwrap();
    output.write_all(&encoded_header).unwrap();
    output.write_all(&payload).unwrap();
    output.sync_all().unwrap();
    drop(output);
    let repacked_hash = sha256(&path).unwrap();
    assert_ne!(
        repacked_hash,
        manifest["files"]["posts.safetensors"]["sha256"]
            .as_str()
            .unwrap()
    );
    manifest["files"]["posts.safetensors"]["bytes"] =
        serde_json::json!(path.metadata().unwrap().len());
    manifest["files"]["posts.safetensors"]["sha256"] = serde_json::json!(repacked_hash);
    // Recompute both identities from the actual changed files, not the original
    // writer's identity, while preserving every source/normalization declaration.
    let mut identity = Sha256::new();
    let mut names = manifest["files"]
        .as_object()
        .unwrap()
        .keys()
        .collect::<Vec<_>>();
    names.sort_unstable();
    for name in names {
        identity.update(name.as_bytes());
        identity.update(sha256(&stage.join(name)).unwrap().as_bytes());
    }
    let corpus_id = hex::encode(identity.finalize())[..20].to_string();
    assert_ne!(corpus_id, result.corpus_id);
    manifest["corpus_id"] = serde_json::json!(corpus_id);
    fs::write(
        stage.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let retained_root = root.path().join("retained-releases");
    fs::create_dir(&retained_root).unwrap();
    let retained = retained_root.join(sha256(&stage.join("manifest.json")).unwrap());
    fs::rename(&stage, &retained).unwrap();

    let repacked = fs::read(retained.join("posts.safetensors")).unwrap();
    assert_ne!(original, repacked);
    let (original_header, original_metadata) = SafeTensors::read_metadata(&original).unwrap();
    let (repacked_header, repacked_metadata) = SafeTensors::read_metadata(&repacked).unwrap();
    for (header_length, metadata) in [
        (original_header, &original_metadata),
        (repacked_header, &repacked_metadata),
    ] {
        for name in ["post_ids", "popularity"] {
            let info = metadata.info(name).unwrap();
            assert_eq!((8 + header_length + info.data_offsets.0) % 8, 0);
        }
    }
    assert!(
        original_metadata
            .info("file_extension_codes")
            .unwrap()
            .data_offsets
            .0
            < original_metadata
                .info("rating_codes")
                .unwrap()
                .data_offsets
                .0
    );
    assert!(
        repacked_metadata
            .info("rating_codes")
            .unwrap()
            .data_offsets
            .0
            < repacked_metadata
                .info("file_extension_codes")
                .unwrap()
                .data_offsets
                .0
    );
    verify_corpus(&retained).unwrap();
    let comparison = equivalence::compare(&result.release_path, &retained).unwrap();
    assert_eq!(comparison["status"], "equivalent");
    assert_eq!(
        comparison["independent_relations"]["artist_relation_exact"],
        true
    );
    assert_eq!(
        comparison["independent_relations"]["artist_links_checked"],
        22
    );
    assert_ne!(
        comparison["reference_manifest_sha256"],
        comparison["candidate_manifest_sha256"]
    );
    assert_eq!(
        sha256(&result.release_path.join("manifest.json")).unwrap(),
        original_manifest_hash
    );
}

#[test]
fn comparison_rejects_hash_valid_but_inconsistent_forward_inverted_relations() {
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let result = build(&fixture(root.path())).unwrap();
    let corpus = Corpus::open(&result.release_path).unwrap();
    let catalog = TagCatalog::load(&result.release_path).unwrap();
    let fox = catalog.resolve("fox").unwrap();
    let last = corpus.tag_post_offsets()[fox + 1] as usize - 1;
    assert_eq!(corpus.tag_post_indices()[last], 49);
    drop(corpus);
    let path = result.release_path.join("postings.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_length]).unwrap();
    let start = 8
        + header_length
        + header["tag_post_indices"]["data_offsets"][0]
            .as_u64()
            .unwrap() as usize;
    // Still sorted and in range, but row50 does not carry fox and row49 does.
    bytes[start + last * 4..start + (last + 1) * 4].copy_from_slice(&50i32.to_le_bytes());
    fs::write(&path, bytes).unwrap();
    let manifest_path = result.release_path.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["files"]["postings.safetensors"]["sha256"] = serde_json::json!(sha256(&path).unwrap());
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    verify_corpus(&result.release_path).unwrap();
    // Identical bytes alone cannot establish the independently required identity.
    assert!(equivalence::compare(&result.release_path, &result.release_path).is_err());
}

#[test]
fn comparison_rejects_a_changed_popularity_value_after_integrity_is_rebound() {
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let result = build(&fixture(root.path())).unwrap();
    let candidate = root.path().join("candidate");
    fs::create_dir(&candidate).unwrap();
    for entry in fs::read_dir(&result.release_path).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), candidate.join(entry.file_name())).unwrap();
    }
    let path = candidate.join("posts.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    let (header_length, metadata) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    let info = metadata.info("popularity").unwrap();
    // Change the final row so comparison must cover the complete column.
    let end = 8 + header_length + info.data_offsets.1;
    bytes[end - 8..end].copy_from_slice(&123.0f64.to_le_bytes());
    fs::write(&path, bytes).unwrap();
    let manifest_path = candidate.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["files"]["posts.safetensors"]["sha256"] = serde_json::json!(sha256(&path).unwrap());
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    verify_corpus(&candidate).unwrap();
    let error = equivalence::compare(&result.release_path, &candidate)
        .err()
        .unwrap();
    assert!(format!("{error:#}").contains("popularity"), "{error:#}");
}

#[test]
fn independent_relation_proofs_reject_each_structurally_valid_corruption() {
    use mutations::*;
    use parquet::record::Field;
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let release = build(&fixture(root.path())).unwrap().release_path;
    verify_corpus(&release).unwrap();
    equivalence::compare(&release, &release).unwrap();
    let catalog = TagCatalog::load(&release).unwrap();
    let artist_a = catalog.resolve("artist_a").unwrap() as i32;
    let metadata = catalog.resolve("supported_metadata").unwrap() as i32;
    for case in [
        "artist_vocab",
        "artist_membership",
        "occurrence",
        "metadata_index",
        "metadata_support",
    ] {
        let stage = root.path().join(case);
        copy_release(&release, &stage);
        match case {
            "artist_vocab" => tensors(&stage.join("postings.safetensors"), |values| {
                let vocab = values["artist_vocab_indices"].i32();
                let omitted = vocab.iter().position(|&v| v == artist_a).unwrap() as i32;
                let offsets = values["post_artist_offsets"].i32();
                let indices = values["post_artist_indices"].i32();
                let mut new_offsets = vec![0];
                let mut new_indices = Vec::new();
                for pair in offsets.windows(2) {
                    new_indices.extend(
                        indices[pair[0] as usize..pair[1] as usize]
                            .iter()
                            .filter(|&&v| v != omitted)
                            .map(|&v| if v > omitted { v - 1 } else { v }),
                    );
                    new_offsets.push(new_indices.len() as i32);
                }
                values.get_mut("artist_vocab_indices").unwrap().set_i32(
                    &vocab
                        .into_iter()
                        .filter(|&v| v != artist_a)
                        .collect::<Vec<_>>(),
                );
                values
                    .get_mut("post_artist_offsets")
                    .unwrap()
                    .set_i32(&new_offsets);
                values
                    .get_mut("post_artist_indices")
                    .unwrap()
                    .set_i32(&new_indices);
            }),
            "artist_membership" => tensors(&stage.join("postings.safetensors"), |values| {
                let mut offsets = values["post_artist_offsets"].i32();
                let mut indices = values["post_artist_indices"].i32();
                assert_eq!(offsets[1], 1);
                indices.remove(0);
                for offset in &mut offsets[1..] {
                    *offset -= 1;
                }
                values
                    .get_mut("post_artist_offsets")
                    .unwrap()
                    .set_i32(&offsets);
                values
                    .get_mut("post_artist_indices")
                    .unwrap()
                    .set_i32(&indices);
            }),
            "occurrence" => {
                let path = stage.join("vocab.parquet");
                let mut entries = rows(&path);
                entries[0] = replace_field(&entries[0], "occ", Field::Long(51));
                write_rows(&path, &entries, entries.len(), false);
            }
            "metadata_index" => tensors(&stage.join("postings.safetensors"), |values| {
                let indices = values["ordinal_metadata_vocab_indices"].i32();
                values
                    .get_mut("ordinal_metadata_vocab_indices")
                    .unwrap()
                    .set_i32(
                        &indices
                            .into_iter()
                            .filter(|&v| v != metadata)
                            .collect::<Vec<_>>(),
                    );
            }),
            "metadata_support" => {
                // Repair both directions, declared size and occurrence: only
                // the independent minimum-support policy should reject.
                tensors(&stage.join("features.safetensors"), |values| {
                    let mut offsets = values["post_tag_offsets"].i32();
                    let mut indices = values["post_tag_indices"].i32();
                    let start = offsets[49] as usize;
                    let end = offsets[50] as usize;
                    let at = start
                        + indices[start..end]
                            .iter()
                            .position(|&v| v == metadata)
                            .unwrap();
                    indices.remove(at);
                    for offset in &mut offsets[50..] {
                        *offset -= 1;
                    }
                    values
                        .get_mut("post_tag_offsets")
                        .unwrap()
                        .set_i32(&offsets);
                    values
                        .get_mut("post_tag_indices")
                        .unwrap()
                        .set_i32(&indices);
                });
                tensors(&stage.join("postings.safetensors"), |values| {
                    let mut offsets = values["tag_post_offsets"].i32();
                    let mut indices = values["tag_post_indices"].i32();
                    let at = offsets[metadata as usize + 1] as usize - 1;
                    assert_eq!(indices.remove(at), 49);
                    for offset in &mut offsets[metadata as usize + 1..] {
                        *offset -= 1;
                    }
                    values
                        .get_mut("tag_post_offsets")
                        .unwrap()
                        .set_i32(&offsets);
                    values
                        .get_mut("tag_post_indices")
                        .unwrap()
                        .set_i32(&indices);
                });
                let path = stage.join("vocab.parquet");
                let entries = rows(&path)
                    .into_iter()
                    .map(|row| {
                        let index = row
                            .get_column_iter()
                            .find(|(name, _)| name.as_str() == "idx")
                            .unwrap()
                            .1;
                        if index == &Field::Int(metadata) {
                            replace_field(&row, "occ", Field::Long(49))
                        } else {
                            row
                        }
                    })
                    .collect::<Vec<_>>();
                write_rows(&path, &entries, entries.len(), false);
                let path = stage.join("manifest.json");
                let mut manifest: serde_json::Value =
                    serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                manifest["nonzero_count"] =
                    serde_json::json!(manifest["nonzero_count"].as_u64().unwrap() - 1);
                fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            }
            _ => unreachable!(),
        }
        let candidate = rebound(&stage);
        verify_corpus(&candidate).unwrap();
        // Identical bytes make all equality gates tautologies. Only an
        // independent semantic proof can reject this structurally valid corpus.
        assert!(
            equivalence::compare(&candidate, &candidate).is_err(),
            "{case}"
        );
    }
}

#[test]
fn comparison_distinguishes_valid_typed_rows_and_provenance() {
    use mutations::*;
    use parquet::record::{Field, RowAccessor};
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let release = build(&fixture(root.path())).unwrap().release_path;
    let catalog = TagCatalog::load(&release).unwrap();
    for case in [
        "alias_target",
        "alias_order",
        "vocab_order",
        "tag_id",
        "tag_id_type",
        "sources",
        "normalization",
    ] {
        let stage = root.path().join(case);
        copy_release(&release, &stage);
        match case {
            "alias_target" | "alias_order" => {
                let path = stage.join("aliases.parquet");
                let mut entries = rows(&path);
                if case == "alias_order" {
                    entries.swap(0, 1);
                } else {
                    assert_eq!(entries[0].get_string(0).unwrap(), "alias_chain");
                    entries[0] = replace_field(
                        &entries[0],
                        "canonical_idx",
                        Field::Int(catalog.resolve("canine").unwrap() as i32),
                    );
                }
                write_rows(&path, &entries, entries.len(), false);
            }
            "vocab_order" | "tag_id" | "tag_id_type" => {
                let path = stage.join("vocab.parquet");
                let mut entries = rows(&path);
                if case == "vocab_order" {
                    entries.swap(0, 1); // durable idx remains unchanged.
                } else if case == "tag_id" {
                    entries[0] = replace_field(&entries[0], "tag_id", Field::Long(999999));
                } else {
                    entries = entries
                        .into_iter()
                        .map(|row| {
                            let id = row.get_long(1).unwrap();
                            replace_field(&row, "tag_id", Field::Int(i32::try_from(id).unwrap()))
                        })
                        .collect();
                }
                write_rows(&path, &entries, entries.len(), case == "tag_id_type");
            }
            "sources" | "normalization" => {
                let path = stage.join("manifest.json");
                let mut manifest: serde_json::Value =
                    serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                if case == "sources" {
                    manifest["sources"]["tags.csv.gz"] = serde_json::json!("a".repeat(64));
                } else {
                    manifest["normalization"]["active_aliases"] = serde_json::json!(5);
                }
                fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            }
            _ => unreachable!(),
        }
        let candidate = rebound(&stage);
        verify_corpus(&candidate).unwrap();
        equivalence::compare(&candidate, &candidate).unwrap();
        assert!(
            equivalence::compare(&release, &candidate).is_err(),
            "{case}"
        );
    }
}

#[test]
fn comparison_requires_exact_bytes_for_every_declared_extra_payload() {
    use mutations::*;
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let release = build(&fixture(root.path())).unwrap().release_path;
    let stage = root.path().join("extra-reference");
    copy_release(&release, &stage);
    fs::write(stage.join("opaque.bin"), [0, 255, 1, 0]).unwrap();
    fs::write(stage.join("supplement.json"), b"{\"value\":1}\n").unwrap();
    let path = stage.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    for name in ["opaque.bin", "supplement.json"] {
        manifest["files"][name] = serde_json::json!({});
    }
    fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let reference = rebound(&stage);
    verify_corpus(&reference).unwrap();
    for case in [
        "unchanged",
        "same_size_change",
        "different_size_change",
        "json_formatting",
        "missing_declaration",
    ] {
        let parent = root.path().join(case);
        fs::create_dir(&parent).unwrap();
        let stage = parent.join("stage");
        copy_release(&reference, &stage);
        match case {
            "unchanged" => {}
            "same_size_change" => {
                fs::write(stage.join("opaque.bin"), [0, 255, 2, 0]).unwrap();
            }
            "different_size_change" => {
                fs::write(stage.join("opaque.bin"), b"longer opaque payload").unwrap();
            }
            "json_formatting" => {
                // Equal parsed values are insufficient for an unhandled payload.
                fs::write(stage.join("supplement.json"), b"{ \"value\": 1 }\n").unwrap();
            }
            "missing_declaration" => {
                let path = stage.join("manifest.json");
                let mut manifest: serde_json::Value =
                    serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                manifest["files"]
                    .as_object_mut()
                    .unwrap()
                    .remove("opaque.bin");
                fs::remove_file(stage.join("opaque.bin")).unwrap();
                fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            }
            _ => unreachable!(),
        }
        let candidate = rebound(&stage);
        verify_corpus(&candidate).unwrap();
        let comparison = equivalence::compare(&reference, &candidate);
        if case == "unchanged" {
            let report = comparison.unwrap();
            assert_eq!(report["status"], "equivalent");
            assert_eq!(report["extra_payloads"].as_object().unwrap().len(), 2);
            for name in ["opaque.bin", "supplement.json"] {
                assert_eq!(
                    report["extra_payloads"][name]["size_and_sha256_equal"],
                    true
                );
            }
        } else {
            let error = comparison.err().unwrap();
            let message = format!("{error:#}");
            let expected = if case == "missing_declaration" {
                "Declared release file sets differ"
            } else {
                "Declared extra payload bytes differ"
            };
            assert!(message.contains(expected), "{case}: {message}");
        }
    }
}

#[test]
fn comparison_accepts_alternate_parquet_row_groups() {
    use mutations::*;
    use parquet::file::reader::{FileReader, SerializedFileReader};
    let root = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let release = build(&fixture(root.path())).unwrap().release_path;
    let stage = root.path().join("row-groups");
    copy_release(&release, &stage);
    for name in ["vocab.parquet", "aliases.parquet"] {
        let path = stage.join(name);
        let entries = rows(&path);
        assert!(entries.len() > 2);
        write_rows(&path, &entries, 2, false);
        let original =
            SerializedFileReader::new(fs::File::open(release.join(name)).unwrap()).unwrap();
        let alternate = SerializedFileReader::new(fs::File::open(&path).unwrap()).unwrap();
        assert_ne!(original.num_row_groups(), alternate.num_row_groups());
        assert_ne!(
            fs::read(release.join(name)).unwrap(),
            fs::read(&path).unwrap()
        );
        assert_eq!(rows(&path), entries);
    }
    let candidate = rebound(&stage);
    verify_corpus(&candidate).unwrap();
    assert_eq!(
        equivalence::compare(&release, &candidate).unwrap()["status"],
        "equivalent"
    );
}

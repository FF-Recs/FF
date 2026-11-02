use crate::{
    contract::{self, AliasEntry, VocabularyEntry},
    input,
    output::{self, StagedTensors, TensorLayout},
    taxonomy,
};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const BUFFER: usize = 1024 * 1024;
const SPOOL_MAGIC: &[u8; 4] = b"FSP5";

// A thread-local observer at the real pre-publication phase lets unit tests
// deterministically rotate a synthetic source without races or production hooks.
#[cfg(test)]
thread_local! {
    static BEFORE_INPUT_RECHECK: std::cell::Cell<Option<fn(&BuildOptions, &Path)>> =
        const { std::cell::Cell::new(None) };
}

#[derive(Clone, Debug)]
pub struct BuildOptions {
    pub posts: PathBuf,
    pub tags: PathBuf,
    pub aliases: PathBuf,
    pub implications: PathBuf,
    pub work: PathBuf,
    pub releases: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct BuildReport {
    pub corpus_id: String,
    pub release_path: PathBuf,
    pub post_count: usize,
    pub tag_count: usize,
    pub nonzero_count: usize,
    pub active_aliases: usize,
    pub active_implications: usize,
    pub timings_seconds: BTreeMap<String, f64>,
}

pub fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

fn writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::with_capacity(
        BUFFER,
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}
fn finish(mut writer: BufWriter<File>) -> Result<()> {
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn source_paths(options: &BuildOptions) -> [(&'static str, &Path); 4] {
    [
        ("posts.csv.gz", options.posts.as_path()),
        ("tag_aliases.csv.gz", options.aliases.as_path()),
        ("tag_implications.csv.gz", options.implications.as_path()),
        ("tags.csv.gz", options.tags.as_path()),
    ]
}

fn spool_posts(
    path: &Path,
    output: &Path,
    taxonomy: &taxonomy::Taxonomy,
) -> Result<(usize, Vec<i64>)> {
    ensure!(
        taxonomy.tags.len() <= i32::MAX as usize,
        "Taxonomy exceeds Int32 format"
    );
    let mut csv = input::open(
        path,
        &[
            "id",
            "md5",
            "rating",
            "file_ext",
            "is_deleted",
            "tag_string",
            "score",
        ],
    )?;
    let id = csv.column("id")?;
    let md5 = csv.column("md5")?;
    let rating = csv.column("rating")?;
    let extension = csv.column("file_ext")?;
    let deleted = csv.column("is_deleted")?;
    let tags = csv.column("tag_string")?;
    let score = csv.column("score")?;
    let mut record = csv::ByteRecord::new();
    let mut normalizer = taxonomy.normalizer();
    let mut frequencies = vec![0i64; taxonomy.tags.len()];
    let mut encoded = Vec::<i32>::with_capacity(128);
    let mut file = writer(output)?;
    file.write_all(SPOOL_MAGIC)?;
    file.write_all(&0u32.to_le_bytes())?;
    let mut previous = 0i64;
    let mut count = 0usize;
    while csv.read(&mut record)? {
        if input::text(&record, deleted)? == "t" {
            continue;
        }
        let extension = input::text(&record, extension)?;
        if extension == "swf" {
            continue;
        }
        let extension_code = contract::extension_code(extension)?;
        let post_id = input::integer(&record, id)?;
        ensure!(
            post_id > previous,
            "Admitted posts must have positive, strictly ordered IDs: {post_id}"
        );
        previous = post_id;
        let mut header = [0u8; 38];
        header[..8].copy_from_slice(&post_id.to_le_bytes());
        hex::decode_to_slice(input::text(&record, md5)?, &mut header[8..24])
            .with_context(|| format!("Post {post_id} has an invalid MD5"))?;
        header[24] = extension_code;
        header[25] = contract::rating_code(input::text(&record, rating)?)?;
        let popularity = signed_log_score(input::integer(&record, score)?);
        header[26..34].copy_from_slice(&popularity.to_le_bytes());
        let normalized = normalizer
            .normalize(input::text(&record, tags)?)
            .with_context(|| format!("Normalizing post {post_id}"))?;
        header[34..].copy_from_slice(&u32::try_from(normalized.len())?.to_le_bytes());
        encoded.clear();
        for &tag in normalized {
            encoded.push(i32::try_from(tag)?);
            frequencies[tag] = frequencies[tag]
                .checked_add(1)
                .context("Frequency overflow")?;
        }
        file.write_all(&header)?;
        file.write_all(bytemuck::cast_slice(&encoded))?;
        count += 1;
        ensure!(
            count <= i32::MAX as usize,
            "Post count exceeds Int32 format"
        );
        if count.is_multiple_of(250_000) {
            eprintln!("[corpus] normalized {count} active posts");
        }
    }
    file.seek(SeekFrom::Start(4))?;
    file.write_all(&u32::try_from(count)?.to_le_bytes())?;
    finish(file)?;
    Ok((count, frequencies))
}

fn signed_log_score(score: i64) -> f64 {
    // Match FF-26 exactly; take the magnitude only after conversion, including MIN.
    (score.signum() as f64) * (score as f64).abs().ln_1p()
}

fn vocabulary(
    taxonomy: &taxonomy::Taxonomy,
    frequencies: &[i64],
) -> Result<(Vec<VocabularyEntry>, Vec<i32>, Vec<AliasEntry>)> {
    let mut selected = (0..taxonomy.tags.len())
        .filter(|&tag| {
            if taxonomy.canonical[tag] != tag || frequencies[tag] <= 0 {
                return false;
            }
            let definition = &taxonomy.tags[tag];
            match definition.category {
                0 | 3 | 4 | 5 => frequencies[tag] >= contract::MINIMUM_CONTENT_POSTS,
                1 => frequencies[tag] >= contract::MINIMUM_ARTIST_POSTS,
                7 => frequencies[tag] >= contract::MINIMUM_METADATA_POSTS,
                _ => false,
            }
        })
        .collect::<Vec<_>>();
    selected.sort_unstable_by(|&left, &right| {
        let l = &taxonomy.tags[left];
        let r = &taxonomy.tags[right];
        (l.category == 1)
            .cmp(&(r.category == 1))
            .then_with(|| frequencies[right].cmp(&frequencies[left]))
            .then_with(|| l.source_id.cmp(&r.source_id))
    });
    let mut mapping = vec![-1i32; taxonomy.tags.len()];
    let mut entries = Vec::with_capacity(selected.len());
    for (index, &tag) in selected.iter().enumerate() {
        let definition = &taxonomy.tags[tag];
        mapping[tag] = i32::try_from(index)?;
        entries.push(VocabularyEntry {
            index: mapping[tag],
            source_id: definition.source_id,
            name: definition.name.to_string(),
            category: definition.category,
            occurrences: frequencies[tag],
        });
    }
    let aliases = taxonomy
        .aliases
        .iter()
        .filter_map(|(alias, tag)| {
            (mapping[*tag] >= 0 && alias.as_str() != taxonomy.tags[*tag].name.as_ref()).then(|| {
                AliasEntry {
                    alias: alias.clone(),
                    canonical_index: mapping[*tag],
                }
            })
        })
        .collect();
    Ok((entries, mapping, aliases))
}

struct IndexedTensors {
    declarations: BTreeMap<&'static str, Vec<&'static str>>,
    nonzeros: usize,
}

fn occurrence_sum<'a>(mut entries: impl Iterator<Item = &'a VocabularyEntry>) -> Result<usize> {
    let total = entries.try_fold(0i32, |sum, entry| {
        let count =
            i32::try_from(entry.occurrences).context("Vocabulary occurrence exceeds Int32")?;
        ensure!(count >= 0, "Negative vocabulary occurrence count");
        sum.checked_add(count)
            .context("Planned links exceed Int32 format")
    })?;
    Ok(usize::try_from(total)?)
}

fn indexed_tensors(
    stage: &Path,
    spool_path: &Path,
    post_count: usize,
    vocabulary: &[VocabularyEntry],
    mapping: &[i32],
) -> Result<IndexedTensors> {
    let nonzeros_planned = occurrence_sum(vocabulary.iter())?;
    let proper_artists = || {
        vocabulary
            .iter()
            .filter(|e| e.category == 1 && !contract::PSEUDO_ARTISTS.contains(&e.name.as_str()))
    };
    let artist_links_planned = occurrence_sum(proper_artists())?;
    let artist_vocab = proper_artists().map(|e| e.index).collect::<Vec<_>>();
    let mut artist_by_vocab = vec![-1i32; vocabulary.len()];
    for (local, &global) in artist_vocab.iter().enumerate() {
        artist_by_vocab[usize::try_from(global)?] = i32::try_from(local)?;
    }
    let metadata_vocab = vocabulary
        .iter()
        .filter(|e| e.category == 7)
        .map(|e| e.index)
        .collect::<Vec<_>>();
    let post_offset_count = post_count
        .checked_add(1)
        .context("Post offset count overflow")?;
    let tag_offset_count = vocabulary
        .len()
        .checked_add(1)
        .context("Tag offset count overflow")?;
    let layout = |name, dtype, shape| TensorLayout { name, dtype, shape };
    let mut features = StagedTensors::create(
        &stage.join("features.safetensors"),
        "features",
        vec![
            layout("post_tag_offsets", Dtype::I32, vec![post_offset_count]),
            layout("post_tag_indices", Dtype::I32, vec![nonzeros_planned]),
        ],
    )?;
    let mut posts = StagedTensors::create(
        &stage.join("posts.safetensors"),
        "posts",
        vec![
            layout("post_ids", Dtype::I64, vec![post_count]),
            layout("popularity", Dtype::F64, vec![post_count]),
            layout("md5_bytes", Dtype::U8, vec![post_count, 16]),
            layout("file_extension_codes", Dtype::U8, vec![post_count]),
            layout("rating_codes", Dtype::U8, vec![post_count]),
        ],
    )?;
    let mut postings = StagedTensors::create(
        &stage.join("postings.safetensors"),
        "postings",
        vec![
            layout("tag_post_offsets", Dtype::I32, vec![tag_offset_count]),
            layout("tag_post_indices", Dtype::I32, vec![nonzeros_planned]),
            layout("artist_vocab_indices", Dtype::I32, vec![artist_vocab.len()]),
            layout(
                "ordinal_metadata_vocab_indices",
                Dtype::I32,
                vec![metadata_vocab.len()],
            ),
            layout("post_artist_offsets", Dtype::I32, vec![post_offset_count]),
            layout(
                "post_artist_indices",
                Dtype::I32,
                vec![artist_links_planned],
            ),
        ],
    )?;
    {
        let mut feature_segments = features.segments()?;
        let mut post_segments = posts.segments()?;
        let mut posting_segments = postings.segments()?;
        let mut forward = feature_segments
            .remove("post_tag_indices")
            .context("Missing forward segment")?;
        let mut post_offsets = feature_segments
            .remove("post_tag_offsets")
            .context("Missing post offsets")?;
        let mut ids = post_segments
            .remove("post_ids")
            .context("Missing post IDs")?;
        let mut md5 = post_segments
            .remove("md5_bytes")
            .context("Missing MD5 bytes")?;
        let mut extensions = post_segments
            .remove("file_extension_codes")
            .context("Missing extensions")?;
        let mut ratings = post_segments
            .remove("rating_codes")
            .context("Missing ratings")?;
        let mut popularity = post_segments
            .remove("popularity")
            .context("Missing popularity")?;
        let mut artist_indices = posting_segments
            .remove("post_artist_indices")
            .context("Missing artist indices")?;
        let mut artist_offsets = posting_segments
            .remove("post_artist_offsets")
            .context("Missing artist offsets")?;
        let mut tag_offsets = posting_segments
            .remove("tag_post_offsets")
            .context("Missing tag offsets")?;
        let mut inverted = posting_segments
            .remove("tag_post_indices")
            .context("Missing inverted indices")?;
        posting_segments
            .get_mut("artist_vocab_indices")
            .context("Missing artist vocabulary")?
            .append(bytemuck::cast_slice(&artist_vocab))?;
        posting_segments
            .get_mut("ordinal_metadata_vocab_indices")
            .context("Missing metadata vocabulary")?
            .append(bytemuck::cast_slice(&metadata_vocab))?;
        post_offsets.append(&0i32.to_le_bytes())?;
        artist_offsets.append(&0i32.to_le_bytes())?;
        let mut tag_counts = vec![0i32; vocabulary.len()];
        let mut spool = BufReader::with_capacity(BUFFER, File::open(spool_path)?);
        let mut spool_header = [0u8; 8];
        spool.read_exact(&mut spool_header)?;
        ensure!(
            &spool_header[..4] == SPOOL_MAGIC
                && u32::from_le_bytes(spool_header[4..].try_into()?) as usize == post_count,
            "Invalid normalized post spool header"
        );
        let mut input_tags = Vec::<i32>::with_capacity(128);
        let mut selected = Vec::<i32>::with_capacity(128);
        let mut artists = Vec::<i32>::with_capacity(8);
        let mut nonzeros = 0i32;
        let mut artist_links = 0i32;
        for _ in 0..post_count {
            let mut header = [0u8; 38];
            spool.read_exact(&mut header)?;
            let count = u32::from_le_bytes(header[34..].try_into()?) as usize;
            ensure!(
                count <= mapping.len(),
                "Normalized spool tag count exceeds taxonomy"
            );
            input_tags.resize(count, 0);
            spool.read_exact(bytemuck::cast_slice_mut(&mut input_tags))?;
            selected.clear();
            for &tag in &input_tags {
                let global = *mapping
                    .get(usize::try_from(tag)?)
                    .context("Normalized spool tag outside taxonomy")?;
                if global >= 0 {
                    selected.push(global);
                }
            }
            selected.sort_unstable();
            forward.append(bytemuck::cast_slice(&selected))?;
            nonzeros = nonzeros
                .checked_add(i32::try_from(selected.len())?)
                .context("Forward links exceed Int32 format")?;
            post_offsets.append(&nonzeros.to_le_bytes())?;
            artists.clear();
            for &tag in &selected {
                let tag = usize::try_from(tag)?;
                tag_counts[tag] = tag_counts[tag]
                    .checked_add(1)
                    .context("Tag posting count overflow")?;
                let artist = artist_by_vocab[tag];
                if artist >= 0 {
                    artists.push(artist);
                }
            }
            artist_indices.append(bytemuck::cast_slice(&artists))?;
            artist_links = artist_links
                .checked_add(i32::try_from(artists.len())?)
                .context("Artist links exceed Int32 format")?;
            artist_offsets.append(&artist_links.to_le_bytes())?;
            ids.append(&header[..8])?;
            md5.append(&header[8..24])?;
            extensions.append(&header[24..25])?;
            ratings.append(&header[25..26])?;
            popularity.append(&header[26..34])?;
        }
        ensure!(
            spool.read(&mut [0u8; 1])? == 0,
            "Normalized post spool has trailing bytes"
        );
        ensure!(
            usize::try_from(nonzeros)? == nonzeros_planned,
            "Actual forward links differ from plan"
        );
        ensure!(
            usize::try_from(artist_links)? == artist_links_planned,
            "Actual artist links differ from plan"
        );
        // Keep the replay's independent per-tag counts: do not derive these
        // offsets from the planned vocabulary frequencies.
        tag_offsets.append(&0i32.to_le_bytes())?;
        let mut inverted_total = 0i32;
        for (&actual, entry) in tag_counts.iter().zip(vocabulary) {
            ensure!(
                i64::from(actual) == entry.occurrences,
                "Actual occurrence count differs for tag '{}'",
                entry.name
            );
            inverted_total = inverted_total
                .checked_add(actual)
                .context("Postings exceed Int32 format")?;
            tag_offsets.append(&inverted_total.to_le_bytes())?;
        }
        ensure!(
            inverted_total == nonzeros,
            "Forward and inverted link counts differ"
        );
        let offsets = tag_offsets.i32_values()?;
        let post_offsets = post_offsets.i32_values()?;
        let forward = forward.i32_values()?;
        inverted.fill_i32(|inverted| {
            let mut cursors = offsets[..vocabulary.len()].to_vec();
            let mut actual_writes = 0usize;
            for row in 0..post_count {
                let post = i32::try_from(row)?;
                let start = usize::try_from(post_offsets[row])?;
                let end = usize::try_from(post_offsets[row + 1])?;
                for &tag in forward
                    .get(start..end)
                    .context("Forward row outside named segment")?
                {
                    let tag = usize::try_from(tag)?;
                    let cursor = cursors
                        .get_mut(tag)
                        .context("Forward tag outside vocabulary")?;
                    ensure!(
                        *cursor < offsets[tag + 1],
                        "Inverted posting cursor outside tag segment"
                    );
                    *inverted
                        .get_mut(usize::try_from(*cursor)?)
                        .context("Inverted cursor outside named segment")? = post;
                    // Each write consumes one bounded slot; the planned total
                    // is Int32-bounded, so these increments cannot overflow.
                    *cursor += 1;
                    actual_writes += 1;
                }
            }
            ensure!(
                cursors.as_slice() == &offsets[1..],
                "Inverted posting cursors missed boundaries"
            );
            Ok(actual_writes)
        })?;
    }
    let declarations = BTreeMap::from([
        ("features.safetensors", features.finish()?),
        ("posts.safetensors", posts.finish()?),
        ("postings.safetensors", postings.finish()?),
    ]);
    Ok(IndexedTensors {
        declarations,
        nonzeros: nonzeros_planned,
    })
}

#[cfg(target_os = "linux")]
pub(crate) fn publish(stage: &Path, destination: &Path) -> Result<()> {
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        stage,
        rustix::fs::CWD,
        destination,
        rustix::fs::RenameFlags::NOREPLACE,
    )
    .with_context(|| format!("Publishing new-only directory {}", destination.display()))?;
    Ok(())
}
#[cfg(not(target_os = "linux"))]
pub(crate) fn publish(stage: &Path, destination: &Path) -> Result<()> {
    ensure!(
        !destination.try_exists()?,
        "Refusing to replace {}",
        destination.display()
    );
    fs::rename(stage, destination)?;
    Ok(())
}

pub fn build(options: &BuildOptions) -> Result<BuildReport> {
    ensure!(
        cfg!(target_endian = "little"),
        "Corpus construction requires little-endian host"
    );
    let started = Instant::now();
    let mut timings = BTreeMap::new();
    let input_paths = source_paths(options);
    let before = input_paths
        .iter()
        .map(|&(name, path)| Ok((name.to_string(), sha256(path)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    timings.insert("input_hashes".into(), started.elapsed().as_secs_f64());
    fs::create_dir_all(&options.work)?;
    fs::create_dir_all(&options.releases)?;
    let work = tempfile::Builder::new()
        .prefix("run-")
        .tempdir_in(&options.work)?;
    let stage = tempfile::Builder::new()
        .prefix(".build-")
        .tempdir_in(&options.releases)?;
    let phase = Instant::now();
    eprintln!("[corpus] loading taxonomy and canonical closure");
    let taxonomy = taxonomy::load(&options.tags, &options.aliases, &options.implications)?;
    timings.insert("taxonomy".into(), phase.elapsed().as_secs_f64());
    let active_aliases = taxonomy.active_aliases;
    let active_implications = taxonomy.active_implications;
    let phase = Instant::now();
    eprintln!("[corpus] normalizing supported active posts");
    let spool = work.path().join("normalized-posts.bin");
    let (post_count, frequencies) = spool_posts(&options.posts, &spool, &taxonomy)?;
    timings.insert("normalize_posts".into(), phase.elapsed().as_secs_f64());
    let (vocabulary, mapping, aliases) = vocabulary(&taxonomy, &frequencies)?;
    drop(frequencies);
    drop(taxonomy);
    let phase = Instant::now();
    eprintln!(
        "[corpus] generating indexed staged tensors: {post_count} posts, {} selected tags",
        vocabulary.len()
    );
    let tensors = indexed_tensors(stage.path(), &spool, post_count, &vocabulary, &mapping)?;
    drop(mapping);
    timings.insert("index_payloads".into(), phase.elapsed().as_secs_f64());
    let phase = Instant::now();
    eprintln!("[corpus] writing Parquet and hashing staged release");
    output::write_vocabulary(&stage.path().join("vocab.parquet"), &vocabulary)?;
    output::write_aliases(&stage.path().join("aliases.parquet"), &aliases)?;
    let mut files = BTreeMap::<&str, Value>::new();
    let mut identity = Sha256::new();
    for name in [
        "aliases.parquet",
        "features.safetensors",
        "postings.safetensors",
        "posts.safetensors",
        "vocab.parquet",
    ] {
        let path = stage.path().join(name);
        let digest = sha256(&path)?;
        identity.update(name.as_bytes());
        identity.update(digest.as_bytes());
        let mut entry = json!({"bytes":path.metadata()?.len(), "sha256":digest});
        if let Some(names) = tensors.declarations.get(name) {
            entry["tensors"] = json!(names);
        }
        files.insert(name, entry);
    }
    let corpus_id = hex::encode(identity.finalize())[..20].to_string();
    let manifest = json!({"format_version":5, "corpus_id":corpus_id, "post_count":post_count,
        "tag_count":vocabulary.len(), "nonzero_count":tensors.nonzeros, "feature_model":"binary_presence_and_signed_log_popularity",
        "normalization":{"active_aliases":active_aliases,"active_implications":active_implications,
            "alias_policy":"terminal_active_alias","implication_policy":"transitive_active_closure_after_aliases"},
        "popularity":{"source_column":"score","transform":"sign(score)*ln(1+abs(score))","arithmetic":"f64"},
        "sources":before,"files":files});
    let mut manifest_file = writer(&stage.path().join("manifest.json"))?;
    serde_json::to_writer_pretty(&mut manifest_file, &manifest)?;
    manifest_file.write_all(b"\n")?;
    finish(manifest_file)?;
    timings.insert(
        "write_and_hash_release".into(),
        phase.elapsed().as_secs_f64(),
    );
    let phase = Instant::now();
    eprintln!("[corpus] verifying release through Rust corpus and catalog consumers");
    ff_core::corpus::verify_corpus(stage.path())?;
    #[cfg(test)]
    BEFORE_INPUT_RECHECK.with(|observer| {
        if let Some(observe) = observer.get() {
            observe(options, stage.path());
        }
    });
    for (name, path) in input_paths {
        ensure!(
            sha256(path)? == before[name],
            "Input changed during build: {}",
            path.display()
        );
    }
    timings.insert(
        "verify_and_rehash_inputs".into(),
        phase.elapsed().as_secs_f64(),
    );
    let release_hash = sha256(&stage.path().join("manifest.json"))?;
    let destination = options.releases.join(release_hash);
    // Directory rename is atomic and no-clobber on supported Linux/Windows targets.
    publish(stage.path(), &destination)?;
    #[cfg(unix)]
    File::open(&options.releases)?.sync_all()?;
    timings.insert("total".into(), started.elapsed().as_secs_f64());
    eprintln!("[corpus] published {}", destination.display());
    Ok(BuildReport {
        corpus_id,
        release_path: destination,
        post_count,
        tag_count: vocabulary.len(),
        nonzero_count: tensors.nonzeros,
        active_aliases,
        active_implications,
        timings_seconds: timings,
    })
}

#[cfg(test)]
#[path = "../tests/unit/build.rs"]
mod tests;

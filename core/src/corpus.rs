use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufReader, Read},
    marker::PhantomData,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, ensure};
use bytemuck::Pod;
use memmap2::Mmap;
use safetensors::{
    SafeTensors,
    tensor::{Dtype, Metadata},
};
use serde::{
    Deserialize, Deserializer,
    de::{DeserializeSeed, IgnoredAny, MapAccess, Visitor},
};
use sha2::{Digest, Sha256};

const MAX_HEADER: usize = 16 * 1024 * 1024;
const REQUIRED_FILES: &[&str] = &[
    "features.safetensors",
    "posts.safetensors",
    "postings.safetensors",
    "vocab.parquet",
    "aliases.parquet",
];

// Unlike collecting a JSON Value, this rejects duplicate keys before they can overwrite data.
#[derive(Debug)]
struct UniqueMap<T>(BTreeMap<String, T>);
impl<'de, T: Deserialize<'de>> Deserialize<'de> for UniqueMap<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct MapVisitor<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> Visitor<'de> for MapVisitor<T> {
            type Value = UniqueMap<T>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an object with unique keys")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut access: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut map = BTreeMap::new();
                while let Some((key, value)) = access.next_entry::<String, T>()? {
                    if map.insert(key.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!("Duplicate key '{key}'")));
                    }
                }
                Ok(UniqueMap(map))
            }
        }
        deserializer.deserialize_map(MapVisitor(PhantomData))
    }
}

#[derive(Deserialize)]
pub(crate) struct Payload {
    pub(crate) bytes: u64,
    sha256: String,
    tensors: Option<Vec<String>>,
}
#[derive(Deserialize)]
pub(crate) struct Manifest {
    format_version: u32,
    pub(crate) corpus_id: String,
    pub(crate) post_count: usize,
    pub(crate) tag_count: usize,
    nonzero_count: usize,
    feature_model: String,
    files: UniqueMap<Payload>,
}

fn valid_digest(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn bounded_read(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("Opening {}", path.display()))?;
    let mut data = Vec::new();
    file.take(MAX_HEADER as u64 + 1).read_to_end(&mut data)?;
    ensure!(
        data.len() <= MAX_HEADER,
        "Metadata exceeds {MAX_HEADER} bytes: {}",
        path.display()
    );
    Ok(data)
}
pub(crate) fn read_manifest(root: &Path) -> Result<Manifest> {
    let bytes = bounded_read(&root.join("manifest.json"))?;
    let manifest: Manifest = serde_json::from_slice(&bytes).context("Invalid corpus manifest")?;
    ensure!(
        manifest.format_version == 5,
        "Unsupported corpus format {}",
        manifest.format_version
    );
    ensure!(
        manifest.feature_model == "binary_presence_and_signed_log_popularity",
        "Unsupported feature model '{}'",
        manifest.feature_model
    );
    ensure!(!manifest.corpus_id.is_empty(), "Empty corpus ID");
    ensure!(
        manifest.post_count <= i32::MAX as usize
            && manifest.tag_count <= i32::MAX as usize
            && manifest.nonzero_count <= i32::MAX as usize,
        "Corpus counts exceed Int32 index format"
    );
    for name in REQUIRED_FILES {
        ensure!(
            manifest.files.0.contains_key(*name),
            "Manifest missing {name}"
        );
    }
    for (name, payload) in &manifest.files.0 {
        validate_payload_name(name)?;
        ensure!(valid_digest(&payload.sha256), "Invalid SHA256 for {name}");
    }
    Ok(manifest)
}
fn validate_payload_name(name: &str) -> Result<()> {
    let mut components = Path::new(name).components();
    ensure!(
        !name.contains('\\')
            && matches!(components.next(), Some(Component::Normal(_)))
            && components.next().is_none(),
        "Unsafe corpus payload name '{name}'"
    );
    Ok(())
}
pub(crate) fn payload_path(root: &Path, manifest: &Manifest, name: &str) -> Result<PathBuf> {
    let payload = manifest
        .files
        .0
        .get(name)
        .with_context(|| format!("Undeclared corpus payload '{name}'"))?;
    validate_payload_name(name)?;
    let canonical_root = root.canonicalize()?;
    let path = root
        .join(name)
        .canonicalize()
        .with_context(|| format!("Resolving corpus payload {name}"))?;
    ensure!(
        path.starts_with(&canonical_root),
        "Corpus payload escapes release directory: {name}"
    );
    let metadata = path.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() == payload.bytes,
        "Corpus payload size mismatch for {name}: expected {}, got {}",
        payload.bytes,
        metadata.len()
    );
    Ok(path)
}
fn streaming_sha256(path: &Path) -> Result<String> {
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

/// Resolve only the pinned manifest-addressed release; never select by directory order or mtime.
pub fn resolve_corpus(repository: &Path) -> Result<PathBuf> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Pin {
        manifest_sha256: String,
    }
    let pin: Pin = serde_json::from_slice(&bounded_read(&repository.join("corpus-release.json"))?)?;
    ensure!(
        valid_digest(&pin.manifest_sha256),
        "Release pin must be 64 lowercase hexadecimal characters"
    );
    let root = repository.join("data/corpora").join(&pin.manifest_sha256);
    ensure!(
        streaming_sha256(&root.join("manifest.json"))? == pin.manifest_sha256,
        "Corpus manifest checksum does not match release pin"
    );
    Ok(root)
}

/// Explicit integrity operation. All declared payload files are streamed.
pub fn verify_corpus(root: &Path) -> Result<()> {
    let manifest = read_manifest(root)?;
    for (name, entry) in &manifest.files.0 {
        let path = payload_path(root, &manifest, name)?;
        ensure!(
            streaming_sha256(&path)? == entry.sha256,
            "Corpus payload checksum mismatch: {name}"
        );
    }
    let corpus = Corpus::open(root)?;
    let catalog = crate::catalog::TagCatalog::load(root)?;
    ensure!(
        catalog.len() == corpus.tag_count,
        "Vocabulary count differs from corpus"
    );
    Ok(())
}

// The official reader accepts unknown descriptor fields and collects names/metadata
// into HashMaps, losing duplicate keys. FF admits a stricter envelope first, without
// owning shapes or interpreting their byte lengths. The official reader remains the
// authority for descriptor types, shape arithmetic, contiguous ranges and coverage.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DescriptorAdmission {
    dtype: Dtype,
    #[serde(rename = "shape")]
    _shape: IgnoredAny,
    data_offsets: (usize, usize),
}

struct HeaderAdmission {
    payload_len: usize,
}
impl<'de> DeserializeSeed<'de> for HeaderAdmission {
    type Value = BTreeSet<String>;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for HeaderAdmission {
    type Value = BTreeSet<String>;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("an FF SafeTensors header with unique keys")
    }

    fn visit_map<M: MapAccess<'de>>(
        self,
        mut access: M,
    ) -> std::result::Result<Self::Value, M::Error> {
        let mut names = BTreeSet::new();
        while let Some(name) = access.next_key::<String>()? {
            if names.contains(&name) {
                return Err(serde::de::Error::custom(format!("Duplicate key '{name}'")));
            }
            if name == "__metadata__" {
                access.next_value::<UniqueMap<IgnoredAny>>()?;
            } else {
                let descriptor = access.next_value::<DescriptorAdmission>()?;
                if !matches!(
                    descriptor.dtype,
                    Dtype::U8 | Dtype::I32 | Dtype::I64 | Dtype::F32 | Dtype::F64
                ) {
                    return Err(serde::de::Error::custom("Unsupported FF tensor dtype"));
                }
                // Bound resource claims before read_metadata's unchecked final
                // endpoint + header addition (safetensors 0.8.0 tensor.rs:420).
                // This is not a second shape/range/coverage validator.
                if descriptor.data_offsets.1 > self.payload_len {
                    return Err(serde::de::Error::custom("Tensor exceeds mapped payload"));
                }
            }
            names.insert(name);
        }
        names.remove("__metadata__");
        Ok(names)
    }
}
struct TensorFile {
    map: Arc<Mmap>,
    data_start: usize,
    metadata: Metadata,
}
impl TensorFile {
    fn open(path: &Path, group: &str, declared: Option<&[String]>) -> Result<Self> {
        ensure!(
            cfg!(target_endian = "little"),
            "Corpus typed mappings require little-endian host"
        );
        let file = File::open(path)?;
        ensure!(file.metadata()?.len() >= 10, "SafeTensors file too short");
        // Releases are immutable, published files. The mapping is read-only and retained by every slice owner.
        let map = Arc::new(
            unsafe { Mmap::map(&file) }.with_context(|| format!("Mapping {}", path.display()))?,
        );
        let header_length = usize::try_from(u64::from_le_bytes(map[..8].try_into()?))?;
        ensure!(
            (1..=MAX_HEADER).contains(&header_length),
            "Invalid SafeTensors header length"
        );
        let data_start = 8usize
            .checked_add(header_length)
            .context("Header overflow")?;
        ensure!(
            data_start <= map.len(),
            "SafeTensors header exceeds file bounds"
        );
        let mut deserializer = serde_json::Deserializer::from_slice(&map[8..data_start]);
        let names = HeaderAdmission {
            payload_len: map.len() - data_start,
        }
        .deserialize(&mut deserializer)
        .context("Invalid FF SafeTensors header")?;
        deserializer
            .end()
            .context("Trailing SafeTensors header data")?;
        let (validated_header_length, metadata) =
            SafeTensors::read_metadata(&map).context("Invalid SafeTensors container")?;
        debug_assert_eq!(header_length, validated_header_length);
        let values = metadata
            .metadata()
            .as_ref()
            .context("Missing SafeTensors metadata")?;
        ensure!(
            values.get("ff_format_version").map(String::as_str) == Some("5")
                && values.get("ff_group").map(String::as_str) == Some(group),
            "Incompatible SafeTensors metadata for {group}"
        );
        if let Some(declared) = declared {
            let mut declared_names: Vec<_> = declared.iter().map(String::as_str).collect();
            declared_names.sort_unstable();
            ensure!(
                declared_names
                    .iter()
                    .copied()
                    .eq(names.iter().map(String::as_str)),
                "Manifest tensor list differs from {group} file"
            );
        }
        Ok(Self {
            map,
            data_start,
            metadata,
        })
    }
    fn tensor<T: Pod>(
        &self,
        name: &str,
        dtype: Dtype,
        shape: Option<&[usize]>,
    ) -> Result<Tensor<T>> {
        let descriptor = self
            .metadata
            .info(name)
            .with_context(|| format!("Missing tensor '{name}'"))?;
        ensure!(descriptor.dtype == dtype, "Wrong dtype for {name}");
        if let Some(shape) = shape {
            ensure!(
                descriptor.shape == shape,
                "Wrong shape for {name}: {:?}, expected {:?}",
                descriptor.shape,
                shape
            );
        } else {
            ensure!(descriptor.shape.len() == 1, "Expected vector tensor {name}");
        }
        let start = self.data_start + descriptor.data_offsets.0;
        let end = self.data_start + descriptor.data_offsets.1;
        let bytes = &self.map[start..end];
        ensure!(
            bytemuck::try_cast_slice::<u8, T>(bytes).is_ok(),
            "Unaligned tensor {name}; zero-copy mapping is not representable"
        );
        Ok(Tensor::Mapped {
            map: Arc::clone(&self.map),
            start,
            end,
            marker: PhantomData,
        })
    }
}
enum Tensor<T> {
    Mapped {
        map: Arc<Mmap>,
        start: usize,
        end: usize,
        marker: PhantomData<T>,
    },
    #[cfg(test)]
    Owned(Vec<T>),
}
impl<T: Pod> Tensor<T> {
    fn slice(&self) -> &[T] {
        match self {
            Self::Mapped {
                map, start, end, ..
            } => bytemuck::cast_slice(&map[*start..*end]),
            #[cfg(test)]
            Self::Owned(values) => values,
        }
    }
}

/// Immutable release view. Huge incidence tensors are always borrowed from file mappings.
pub struct Corpus {
    pub root: PathBuf,
    pub corpus_id: String,
    pub manifest_sha256: String,
    pub post_count: usize,
    pub tag_count: usize,
    pub artist_count: usize,
    ids: Tensor<i64>,
    md5s: Tensor<u8>,
    extensions: Tensor<u8>,
    ratings: Tensor<u8>,
    popularities: Tensor<f64>,
    post_offsets: Tensor<i32>,
    post_indices: Tensor<i32>,
    tag_offsets: Tensor<i32>,
    tag_indices: Tensor<i32>,
    artist_vocab: Tensor<i32>,
    metadata_vocab: Tensor<i32>,
    artist_offsets: Tensor<i32>,
    artist_indices: Tensor<i32>,
}
impl Corpus {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref().canonicalize()?;
        let manifest = read_manifest(&root)?;
        let load = |name: &str, group: &str| -> Result<TensorFile> {
            let path = payload_path(&root, &manifest, name)?;
            TensorFile::open(&path, group, manifest.files.0[name].tensors.as_deref())
        };
        let features = load("features.safetensors", "features")?;
        let posts = load("posts.safetensors", "posts")?;
        let postings = load("postings.safetensors", "postings")?;
        let n = manifest.post_count;
        let p = manifest.tag_count;
        let nnz = manifest.nonzero_count;
        let artist_vocab = postings.tensor("artist_vocab_indices", Dtype::I32, None)?;
        let corpus = Self {
            root: root.clone(),
            corpus_id: manifest.corpus_id,
            manifest_sha256: streaming_sha256(&root.join("manifest.json"))?,
            post_count: n,
            tag_count: p,
            artist_count: artist_vocab.slice().len(),
            ids: posts.tensor("post_ids", Dtype::I64, Some(&[n]))?,
            md5s: posts.tensor("md5_bytes", Dtype::U8, Some(&[n, 16]))?,
            extensions: posts.tensor("file_extension_codes", Dtype::U8, Some(&[n]))?,
            ratings: posts.tensor("rating_codes", Dtype::U8, Some(&[n]))?,
            popularities: posts.tensor("popularity", Dtype::F64, Some(&[n]))?,
            post_offsets: features.tensor("post_tag_offsets", Dtype::I32, Some(&[n + 1]))?,
            post_indices: features.tensor("post_tag_indices", Dtype::I32, Some(&[nnz]))?,
            tag_offsets: postings.tensor("tag_post_offsets", Dtype::I32, Some(&[p + 1]))?,
            tag_indices: postings.tensor("tag_post_indices", Dtype::I32, Some(&[nnz]))?,
            artist_vocab,
            metadata_vocab: postings.tensor("ordinal_metadata_vocab_indices", Dtype::I32, None)?,
            artist_offsets: postings.tensor("post_artist_offsets", Dtype::I32, Some(&[n + 1]))?,
            artist_indices: postings.tensor("post_artist_indices", Dtype::I32, None)?,
        };
        corpus.validate()?;
        Ok(corpus)
    }
    pub fn post_ids(&self) -> &[i64] {
        self.ids.slice()
    }
    pub fn popularities(&self) -> &[f64] {
        self.popularities.slice()
    }
    pub fn popularity(&self, row: usize) -> f64 {
        self.popularities()[row]
    }
    pub fn post_tag_offsets(&self) -> &[i32] {
        self.post_offsets.slice()
    }
    pub fn post_tag_indices(&self) -> &[i32] {
        self.post_indices.slice()
    }
    pub fn tag_post_offsets(&self) -> &[i32] {
        self.tag_offsets.slice()
    }
    pub fn tag_post_indices(&self) -> &[i32] {
        self.tag_indices.slice()
    }
    pub fn artist_vocab_indices(&self) -> &[i32] {
        self.artist_vocab.slice()
    }
    pub fn ordinal_metadata_vocab_indices(&self) -> &[i32] {
        self.metadata_vocab.slice()
    }
    pub fn post_tags(&self, row: usize) -> &[i32] {
        relation(self.post_tag_offsets(), self.post_tag_indices(), row)
    }
    pub fn post_artists(&self, row: usize) -> &[i32] {
        relation(
            self.artist_offsets.slice(),
            self.artist_indices.slice(),
            row,
        )
    }
    pub fn tag_posts(&self, tag: usize) -> &[i32] {
        relation(self.tag_post_offsets(), self.tag_post_indices(), tag)
    }
    pub fn find_post(&self, id: i64) -> Option<usize> {
        self.post_ids().binary_search(&id).ok()
    }
    pub fn md5(&self, row: usize) -> String {
        assert!(row < self.post_count);
        hex::encode(&self.md5s.slice()[row * 16..(row + 1) * 16])
    }
    pub fn extension(&self, row: usize) -> &'static str {
        match self.extensions.slice()[row] {
            1 => "jpg",
            2 => "png",
            3 => "gif",
            4 => "webm",
            6 => "mp4",
            7 => "webp",
            _ => unreachable!("validated extension"),
        }
    }
    pub fn rating_code(&self, row: usize) -> u8 {
        self.ratings.slice()[row]
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.popularities().len() == self.post_count
                && self.popularities().iter().all(|value| value.is_finite()),
            "Popularity must contain one finite value per post"
        );
        ensure!(
            self.post_ids().iter().all(|&id| id > 0)
                && self.post_ids().windows(2).all(|w| w[0] < w[1]),
            "Post IDs must be positive and strictly increasing"
        );
        validate_csr(
            self.post_tag_offsets(),
            self.post_tag_indices(),
            self.tag_count,
            "post/tag",
        )?;
        validate_csr(
            self.tag_post_offsets(),
            self.tag_post_indices(),
            self.post_count,
            "tag/post",
        )?;
        validate_csr(
            self.artist_offsets.slice(),
            self.artist_indices.slice(),
            self.artist_count,
            "post/artist",
        )?;
        validate_sorted(
            self.artist_vocab_indices(),
            self.tag_count,
            "artist vocabulary",
        )?;
        validate_sorted(
            self.ordinal_metadata_vocab_indices(),
            self.tag_count,
            "ordinal metadata vocabulary",
        )?;
        let mut artists = self.artist_vocab_indices().iter().peekable();
        for tag in self.ordinal_metadata_vocab_indices() {
            while artists.peek().is_some_and(|artist| *artist < tag) {
                artists.next();
            }
            ensure!(
                artists.peek().is_none_or(|artist| *artist != tag),
                "Artist and ordinal metadata vocabularies overlap"
            );
        }
        for row in 0..self.post_count {
            ensure!(
                matches!(self.extensions.slice()[row], 1 | 2 | 3 | 4 | 6 | 7),
                "Invalid extension code at row {row}"
            );
            ensure!(
                (1..=3).contains(&self.ratings.slice()[row]),
                "Invalid rating code at row {row}"
            );
            let post_artists = self.post_artists(row);
            // The artist CSR uses artist-local indices, never global vocabulary indices.
            for &artist in post_artists {
                ensure!(
                    self.post_tags(row)
                        .binary_search(&self.artist_vocab_indices()[artist as usize])
                        .is_ok(),
                    "Post artist is absent from tags at row {row}"
                );
            }
        }
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn fixture(
        post_ids: &[i64],
        post_tags: &[Vec<i32>],
        tag_count: usize,
        artist_vocab_indices: &[i32],
        ordinal_metadata_vocab_indices: &[i32],
    ) -> Self {
        assert_eq!(post_ids.len(), post_tags.len());
        fn flatten(rows: &[Vec<i32>]) -> (Tensor<i32>, Tensor<i32>) {
            let mut offsets = vec![0];
            let mut indices = Vec::new();
            for row in rows {
                indices.extend_from_slice(row);
                offsets.push(i32::try_from(indices.len()).unwrap());
            }
            (Tensor::Owned(offsets), Tensor::Owned(indices))
        }
        let mut tags = vec![Vec::new(); tag_count];
        let mut artists = Vec::new();
        for (row, values) in post_tags.iter().enumerate() {
            for &tag in values {
                tags[tag as usize].push(row as i32);
            }
            artists.push(
                values
                    .iter()
                    .filter_map(|tag| {
                        artist_vocab_indices
                            .binary_search(tag)
                            .ok()
                            .map(|artist| artist as i32)
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let (post_offsets, post_indices) = flatten(post_tags);
        let (tag_offsets, tag_indices) = flatten(&tags);
        let (artist_offsets, artist_indices) = flatten(&artists);
        let corpus = Self {
            root: PathBuf::new(),
            corpus_id: "synthetic-test".into(),
            manifest_sha256: "0".repeat(64),
            post_count: post_ids.len(),
            tag_count,
            artist_count: artist_vocab_indices.len(),
            ids: Tensor::Owned(post_ids.to_vec()),
            md5s: Tensor::Owned(vec![0; post_ids.len() * 16]),
            extensions: Tensor::Owned(vec![1; post_ids.len()]),
            ratings: Tensor::Owned(vec![1; post_ids.len()]),
            popularities: Tensor::Owned(vec![0.0; post_ids.len()]),
            post_offsets,
            post_indices,
            tag_offsets,
            tag_indices,
            artist_vocab: Tensor::Owned(artist_vocab_indices.to_vec()),
            metadata_vocab: Tensor::Owned(ordinal_metadata_vocab_indices.to_vec()),
            artist_offsets,
            artist_indices,
        };
        corpus.validate().unwrap();
        corpus
    }
    #[cfg(test)]
    pub(crate) fn with_popularities(mut self, popularities: &[f64]) -> Self {
        self.popularities = Tensor::Owned(popularities.to_vec());
        self.validate().unwrap();
        self
    }
}
fn relation<'a>(offsets: &[i32], values: &'a [i32], row: usize) -> &'a [i32] {
    assert!(row < offsets.len() - 1, "Corpus row out of bounds");
    &values[offsets[row] as usize..offsets[row + 1] as usize]
}
fn validate_sorted(values: &[i32], upper: usize, name: &str) -> Result<()> {
    let mut previous = -1;
    for &value in values {
        ensure!(
            value >= 0 && (value as usize) < upper && value > previous,
            "{name} contains out-of-range, duplicate or unsorted index {value}"
        );
        previous = value;
    }
    Ok(())
}
fn validate_csr(offsets: &[i32], values: &[i32], upper: usize, name: &str) -> Result<()> {
    ensure!(
        offsets.first() == Some(&0) && offsets.last().copied() == i32::try_from(values.len()).ok(),
        "{name} offsets do not cover relation"
    );
    for pair in offsets.windows(2) {
        ensure!(
            pair[0] >= 0 && pair[0] <= pair[1] && pair[1] as usize <= values.len(),
            "{name} offset outside relation"
        );
        validate_sorted(&values[pair[0] as usize..pair[1] as usize], upper, name)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/corpus.rs"]
mod tests;

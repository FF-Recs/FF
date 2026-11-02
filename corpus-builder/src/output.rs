use std::{
    collections::{BTreeMap, HashSet},
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, ensure};
use memmap2::MmapMut;
use parquet::{
    basic::Compression,
    data_type::{ByteArray, ByteArrayType, DataType, Int32Type, Int64Type},
    file::{
        properties::WriterProperties,
        writer::{SerializedFileWriter, SerializedRowGroupWriter},
    },
    schema::parser::parse_message_type,
};
use safetensors::{
    Dtype,
    tensor::{Metadata, TensorInfo},
};
use serde::{Serialize, ser::SerializeMap};

use crate::contract::{AliasEntry, VocabularyEntry};

const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;
const PARQUET_ROWS: usize = 64 * 1024;

pub(crate) struct TensorLayout {
    pub name: &'static str,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

fn new_output(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("Creating new output {}", path.display()))
}

fn tensor_bytes(spec: &TensorLayout) -> Result<usize> {
    ensure!(
        matches!(
            spec.dtype,
            Dtype::U8 | Dtype::I32 | Dtype::F32 | Dtype::I64 | Dtype::F64
        ),
        "Unsupported FF SafeTensors dtype '{}'",
        spec.dtype
    );
    let elements = spec.shape.iter().try_fold(1usize, |total, &dimension| {
        ensure!(
            dimension <= i64::MAX as usize,
            "Tensor dimension exceeds Int64"
        );
        let count = total
            .checked_mul(dimension)
            .context("Tensor shape overflow")?;
        ensure!(
            count <= i64::MAX as usize,
            "Tensor shape exceeds Int64 capacity"
        );
        Ok::<_, anyhow::Error>(count)
    })?;
    let bytes = elements
        .checked_mul(spec.dtype.bitsize() / 8)
        .context("Tensor byte length overflow")?;
    ensure!(
        bytes <= i64::MAX as usize,
        "Tensor byte length exceeds Int64 capacity"
    );
    Ok(bytes)
}

#[derive(Serialize)]
struct FormatMetadata<'a> {
    ff_format_version: &'static str,
    ff_group: &'a str,
}

/// Serialize official descriptors without the upstream metadata HashMap's
/// nondeterministic ordering or a second JSON value tree.
struct CanonicalHeader<'a> {
    group: &'a str,
    names: &'a [&'static str],
    metadata: &'a Metadata,
}

impl Serialize for CanonicalHeader<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.names.len() + 1))?;
        map.serialize_entry(
            "__metadata__",
            &FormatMetadata {
                ff_format_version: "5",
                ff_group: self.group,
            },
        )?;
        for &name in self.names {
            map.serialize_entry(name, self.metadata.info(name).expect("planned tensor"))?;
        }
        map.end()
    }
}

/// A new-only final container, private until the enclosing release is published.
/// Physical order is descending dtype width, with no padding between tensors.
pub(crate) struct StagedTensors {
    path: PathBuf,
    file: Option<File>,
    map: Option<MmapMut>,
    metadata: Metadata,
    names: Vec<&'static str>,
    written: Vec<usize>,
    data_start: usize,
    complete: bool,
}

impl StagedTensors {
    pub fn create(path: &Path, group: &str, mut specs: Vec<TensorLayout>) -> Result<Self> {
        let mut unique = HashSet::with_capacity(specs.len());
        for spec in &specs {
            ensure!(
                !spec.name.trim().is_empty() && spec.name != "__metadata__",
                "Invalid tensor name '{}'",
                spec.name
            );
            ensure!(
                unique.insert(spec.name),
                "Duplicate tensor name '{}'",
                spec.name
            );
        }
        specs.sort_unstable_by(|left, right| {
            right
                .dtype
                .cmp(&left.dtype)
                .then_with(|| left.name.cmp(right.name))
        });
        let mut offset = 0usize;
        let mut names = Vec::with_capacity(specs.len());
        let mut descriptors = Vec::with_capacity(specs.len());
        for spec in specs {
            let length = tensor_bytes(&spec).with_context(|| format!("Tensor '{}'", spec.name))?;
            let end = offset
                .checked_add(length)
                .context("SafeTensors payload overflow")?;
            ensure!(
                end <= i64::MAX as usize,
                "SafeTensors payload exceeds Int64 capacity"
            );
            names.push(spec.name);
            descriptors.push((
                spec.name.to_string(),
                TensorInfo {
                    dtype: spec.dtype,
                    shape: spec.shape,
                    data_offsets: (offset, end),
                },
            ));
            offset = end;
        }
        // The official library owns standard descriptor/layout validation.
        let metadata = Metadata::new(None, descriptors)?;
        let header = serde_json::to_vec(&CanonicalHeader {
            group,
            names: &names,
            metadata: &metadata,
        })?;
        let padded = header
            .len()
            .checked_add(7)
            .context("SafeTensors header overflow")?
            & !7;
        ensure!(
            padded <= MAX_HEADER_BYTES,
            "SafeTensors header exceeds runtime limit"
        );
        let data_start = 8usize
            .checked_add(padded)
            .context("SafeTensors header overflow")?;
        let file_length = data_start
            .checked_add(offset)
            .context("SafeTensors file length overflow")?;
        ensure!(
            file_length <= i64::MAX as usize,
            "SafeTensors file exceeds Int64 capacity"
        );
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("Creating new output {}", path.display()))?;
        // Install cleanup ownership immediately after creation, including allocation/map failures.
        let mut output = Self {
            path: path.to_path_buf(),
            file: Some(file),
            map: None,
            metadata,
            written: vec![0; names.len()],
            names,
            data_start,
            complete: false,
        };
        #[cfg(target_os = "linux")]
        rustix::fs::fallocate(
            output.file.as_ref().unwrap(),
            rustix::fs::FallocateFlags::empty(),
            0,
            u64::try_from(file_length)?,
        )
        .context("Reserving staged tensor storage before writable mapping")?;
        #[cfg(not(target_os = "linux"))]
        output
            .file
            .as_ref()
            .unwrap()
            .set_len(u64::try_from(file_length)?)?;
        // Only this unpublished builder owns the file, and its length never changes.
        let mut map = unsafe { MmapMut::map_mut(output.file.as_ref().unwrap()) }?;
        map[..8].copy_from_slice(&(padded as u64).to_le_bytes());
        map[8..8 + header.len()].copy_from_slice(&header);
        map[8 + header.len()..data_start].fill(b' ');
        output.map = Some(map);
        Ok(output)
    }

    /// Extract disjoint named views once; no lookup or allocation occurs per row.
    pub fn segments(&mut self) -> Result<BTreeMap<&'static str, TensorSegment<'_>>> {
        ensure!(
            self.written.iter().all(|&n| n == 0),
            "Tensor segments already written"
        );
        let mut remaining =
            &mut self.map.as_mut().context("Closed tensor mapping")?[self.data_start..];
        let mut segments = BTreeMap::new();
        for (&name, written) in self.names.iter().zip(&mut self.written) {
            let info = self.metadata.info(name).expect("planned tensor");
            let length = info.data_offsets.1 - info.data_offsets.0;
            let (data, tail) = remaining.split_at_mut(length);
            remaining = tail;
            segments.insert(
                name,
                TensorSegment {
                    name,
                    dtype: info.dtype,
                    data,
                    written,
                },
            );
        }
        ensure!(remaining.is_empty(), "Unassigned tensor payload bytes");
        Ok(segments)
    }

    pub fn finish(mut self) -> Result<Vec<&'static str>> {
        for (&name, &written) in self.names.iter().zip(&self.written) {
            let info = self.metadata.info(name).expect("planned tensor");
            ensure!(
                written == info.data_offsets.1 - info.data_offsets.0,
                "Tensor '{name}' underfilled: {written} bytes"
            );
        }
        let map = self.map.take().context("Closed tensor mapping")?;
        map.flush()?;
        drop(map);
        self.file.take().unwrap().sync_all()?;
        self.complete = true;
        let mut names = std::mem::take(&mut self.names);
        names.sort_unstable();
        Ok(names)
    }
}

impl Drop for StagedTensors {
    fn drop(&mut self) {
        drop(self.map.take());
        drop(self.file.take());
        if !self.complete {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

pub(crate) struct TensorSegment<'a> {
    name: &'static str,
    dtype: Dtype,
    data: &'a mut [u8],
    written: &'a mut usize,
}

impl TensorSegment<'_> {
    pub fn append(&mut self, bytes: &[u8]) -> Result<()> {
        let end = self
            .written
            .checked_add(bytes.len())
            .context("Tensor cursor overflow")?;
        let target = self
            .data
            .get_mut(*self.written..end)
            .with_context(|| format!("Tensor '{}' overfilled", self.name))?;
        target.copy_from_slice(bytes);
        *self.written = end;
        Ok(())
    }

    pub fn i32_values(&self) -> Result<&[i32]> {
        ensure!(
            self.dtype == Dtype::I32 && *self.written == self.data.len(),
            "Tensor '{}' is not a filled I32 segment",
            self.name
        );
        bytemuck::try_cast_slice(self.data)
            .map_err(|e| anyhow::anyhow!("Tensor '{}': {e}", self.name))
    }

    /// The transpose caller proves every tag's bounds/final cursor independently;
    /// also require its actual write total to equal this exact named segment.
    pub fn fill_i32(&mut self, fill: impl FnOnce(&mut [i32]) -> Result<usize>) -> Result<()> {
        ensure!(
            self.dtype == Dtype::I32 && *self.written == 0,
            "Invalid indexed tensor fill"
        );
        let values = bytemuck::try_cast_slice_mut(self.data)
            .map_err(|e| anyhow::anyhow!("Tensor '{}': {e}", self.name))?;
        let actual = fill(values)?;
        ensure!(
            actual == values.len(),
            "Tensor '{}' indexed fill count differs",
            self.name
        );
        *self.written = self.data.len();
        Ok(())
    }
}

fn write_column<T: DataType>(
    group: &mut SerializedRowGroupWriter<'_, &mut File>,
    values: &[T::T],
) -> Result<()> {
    let mut column = group
        .next_column()?
        .context("Missing Parquet output column")?;
    let written = column.typed::<T>().write_batch(values, None, None)?;
    ensure!(written == values.len(), "Incomplete Parquet column write");
    column.close()?;
    Ok(())
}

pub fn write_vocabulary(path: &Path, entries: &[VocabularyEntry]) -> Result<()> {
    let schema = Arc::new(parse_message_type(
        "message vocabulary { REQUIRED INT32 idx (INTEGER(32,true)); REQUIRED INT64 tag_id (INTEGER(64,true)); REQUIRED BINARY name (STRING); REQUIRED INT32 category (INTEGER(32,true)); REQUIRED INT64 occ (INTEGER(64,true)); }",
    )?);
    let properties = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build(),
    );
    let mut file = new_output(path)?;
    let mut writer = SerializedFileWriter::new(&mut file, schema, properties)?;
    let capacity = entries.len().min(PARQUET_ROWS);
    let mut integers = Vec::with_capacity(capacity);
    let mut longs = Vec::with_capacity(capacity);
    let mut strings = Vec::with_capacity(capacity);
    for entries in entries.chunks(PARQUET_ROWS) {
        let mut group = writer.next_row_group()?;
        integers.clear();
        integers.extend(entries.iter().map(|entry| entry.index));
        write_column::<Int32Type>(&mut group, &integers)?;
        longs.clear();
        longs.extend(entries.iter().map(|entry| entry.source_id));
        write_column::<Int64Type>(&mut group, &longs)?;
        strings.clear();
        strings.extend(
            entries
                .iter()
                .map(|entry| ByteArray::from(entry.name.as_str())),
        );
        write_column::<ByteArrayType>(&mut group, &strings)?;
        integers.clear();
        integers.extend(entries.iter().map(|entry| entry.category));
        write_column::<Int32Type>(&mut group, &integers)?;
        longs.clear();
        longs.extend(entries.iter().map(|entry| entry.occurrences));
        write_column::<Int64Type>(&mut group, &longs)?;
        group.close()?;
    }
    writer.close()?;
    file.sync_all()?;
    Ok(())
}

pub fn write_aliases(path: &Path, entries: &[AliasEntry]) -> Result<()> {
    let schema = Arc::new(parse_message_type(
        "message aliases { REQUIRED BINARY alias (STRING); REQUIRED INT32 canonical_idx (INTEGER(32,true)); }",
    )?);
    let properties = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build(),
    );
    let mut file = new_output(path)?;
    let mut writer = SerializedFileWriter::new(&mut file, schema, properties)?;
    let capacity = entries.len().min(PARQUET_ROWS);
    let mut strings = Vec::with_capacity(capacity);
    let mut indices = Vec::with_capacity(capacity);
    for entries in entries.chunks(PARQUET_ROWS) {
        let mut group = writer.next_row_group()?;
        strings.clear();
        strings.extend(
            entries
                .iter()
                .map(|entry| ByteArray::from(entry.alias.as_str())),
        );
        write_column::<ByteArrayType>(&mut group, &strings)?;
        indices.clear();
        indices.extend(entries.iter().map(|entry| entry.canonical_index));
        write_column::<Int32Type>(&mut group, &indices)?;
        group.close()?;
    }
    writer.close()?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/output.rs"]
mod tests;

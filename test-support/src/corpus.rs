//! Small, real format-5 corpus shared by core and desktop owner tests.
use anyhow::{Result, ensure};
use parquet::{
    data_type::{ByteArray, ByteArrayType, Int32Type, Int64Type},
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    schema::parser::parse_message_type,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Write, path::Path, sync::Arc};

type Tensor = (&'static str, &'static str, Vec<usize>, Vec<u8>);

fn integers(name: &'static str, values: &[i32]) -> Tensor {
    (
        name,
        "I32",
        vec![values.len()],
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

fn descriptor(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path)?;
    Ok(json!({"bytes":bytes.len(),"sha256":format!("{:x}", Sha256::digest(&bytes))}))
}

fn tensors(root: &Path, group: &str, mut values: Vec<Tensor>) -> Result<Value> {
    // Safetensors requires naturally aligned data, including the f64 arrays.
    values.sort_by_key(|(_, dtype, _, _)| {
        std::cmp::Reverse(match *dtype {
            "I64" | "F64" => 8,
            "I32" => 4,
            "U8" => 1,
            _ => unreachable!(),
        })
    });
    let mut header = Map::new();
    header.insert(
        "__metadata__".into(),
        json!({"ff_format_version":"5","ff_group":group}),
    );
    let mut payload = Vec::new();
    let mut names = Vec::new();
    for (name, dtype, shape, bytes) in values {
        let start = payload.len();
        payload.extend(bytes);
        header.insert(
            name.into(),
            json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}),
        );
        names.push(name);
    }
    let mut header = serde_json::to_vec(&header)?;
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let path = root.join(format!("{group}.safetensors"));
    let mut file = File::create(&path)?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&payload)?;
    drop(file);
    let mut result = descriptor(&path)?;
    result["tensors"] = json!(names);
    Ok(result)
}

/// IDs are 1..=post_count. Each four-row block is:
/// artist_a+wolf, artist_a+fox, artist_b+wolf, artist_a+artist_b+wolf.
/// `vulpine` resolves to `fox`; artist and content membership is deliberately uneven.
pub fn write(root: &Path, post_count: usize) -> Result<()> {
    ensure!(
        post_count <= i32::MAX as usize / 3,
        "Fixture exceeds CSR index capacity"
    );
    std::fs::create_dir_all(root)?;
    let mut offsets = vec![0];
    let mut indices = Vec::new();
    let mut artist_offsets = vec![0];
    let mut artists = Vec::new();
    let mut inverted = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for row in 0..post_count {
        let tags: &[i32] = match row % 4 {
            0 => &[0, 2],
            1 => &[0, 3],
            2 => &[1, 2],
            _ => &[0, 1, 2],
        };
        for &tag in tags {
            indices.push(tag);
            inverted[tag as usize].push(row as i32);
            if tag < 2 {
                artists.push(tag);
            }
        }
        offsets.push(indices.len() as i32);
        artist_offsets.push(artists.len() as i32);
    }
    let mut tag_offsets = vec![0];
    let mut tag_indices = Vec::new();
    for rows in &inverted {
        tag_indices.extend_from_slice(rows);
        tag_offsets.push(tag_indices.len() as i32);
    }
    let mut files = Map::new();
    files.insert(
        "features.safetensors".into(),
        tensors(
            root,
            "features",
            vec![
                integers("post_tag_offsets", &offsets),
                integers("post_tag_indices", &indices),
            ],
        )?,
    );
    let ids: Vec<u8> = (1..=post_count as i64).flat_map(i64::to_le_bytes).collect();
    let md5s: Vec<u8> = (1..=post_count as u64)
        .flat_map(|id| {
            let mut bytes = [0; 16];
            bytes[..8].copy_from_slice(&id.to_le_bytes());
            bytes
        })
        .collect();
    files.insert(
        "posts.safetensors".into(),
        tensors(
            root,
            "posts",
            vec![
                ("post_ids", "I64", vec![post_count], ids),
                (
                    "popularity",
                    "F64",
                    vec![post_count],
                    (0..post_count)
                        .flat_map(|row| ((row % 7) as f64 - 3.0).to_le_bytes())
                        .collect(),
                ),
                ("md5_bytes", "U8", vec![post_count, 16], md5s),
                (
                    "file_extension_codes",
                    "U8",
                    vec![post_count],
                    vec![1; post_count],
                ),
                ("rating_codes", "U8", vec![post_count], vec![1; post_count]),
            ],
        )?,
    );
    files.insert(
        "postings.safetensors".into(),
        tensors(
            root,
            "postings",
            vec![
                integers("tag_post_offsets", &tag_offsets),
                integers("tag_post_indices", &tag_indices),
                integers("artist_vocab_indices", &[0, 1]),
                integers("ordinal_metadata_vocab_indices", &[]),
                integers("post_artist_offsets", &artist_offsets),
                integers("post_artist_indices", &artists),
            ],
        )?,
    );

    let path = root.join("vocab.parquet");
    let schema = Arc::new(parse_message_type(
        "message vocabulary { REQUIRED INT32 idx; REQUIRED BINARY name (UTF8); REQUIRED INT32 category; REQUIRED INT64 occ; }",
    )?);
    let mut writer = SerializedFileWriter::new(
        File::create(&path)?,
        schema,
        Arc::new(WriterProperties::default()),
    )?;
    let mut group = writer.next_row_group()?;
    let mut column = group.next_column()?.unwrap();
    column
        .typed::<Int32Type>()
        .write_batch(&[0, 1, 2, 3], None, None)?;
    column.close()?;
    let mut column = group.next_column()?.unwrap();
    let names: Vec<_> = ["artist_a", "artist_b", "wolf", "fox"]
        .into_iter()
        .map(ByteArray::from)
        .collect();
    column
        .typed::<ByteArrayType>()
        .write_batch(&names, None, None)?;
    column.close()?;
    let mut column = group.next_column()?.unwrap();
    column
        .typed::<Int32Type>()
        .write_batch(&[1, 1, 0, 0], None, None)?;
    column.close()?;
    let mut column = group.next_column()?.unwrap();
    let occurrences: Vec<_> = inverted.iter().map(|rows| rows.len() as i64).collect();
    column
        .typed::<Int64Type>()
        .write_batch(&occurrences, None, None)?;
    column.close()?;
    group.close()?;
    writer.close()?;
    files.insert("vocab.parquet".into(), descriptor(&path)?);

    let path = root.join("aliases.parquet");
    let schema = Arc::new(parse_message_type(
        "message aliases { REQUIRED BINARY alias (UTF8); REQUIRED INT32 canonical_idx; }",
    )?);
    let mut writer = SerializedFileWriter::new(
        File::create(&path)?,
        schema,
        Arc::new(WriterProperties::default()),
    )?;
    let mut group = writer.next_row_group()?;
    let mut column = group.next_column()?.unwrap();
    column
        .typed::<ByteArrayType>()
        .write_batch(&[ByteArray::from("vulpine")], None, None)?;
    column.close()?;
    let mut column = group.next_column()?.unwrap();
    column.typed::<Int32Type>().write_batch(&[3], None, None)?;
    column.close()?;
    group.close()?;
    writer.close()?;
    files.insert("aliases.parquet".into(), descriptor(&path)?);
    std::fs::write(
        root.join("manifest.json"),
        serde_json::to_vec(&json!({
            "format_version":5,"corpus_id":format!("synthetic-owner-{post_count}"),
            "post_count":post_count,"tag_count":4,"nonzero_count":indices.len(),
            "feature_model":"binary_presence_and_signed_log_popularity","files":files,
        }))?,
    )?;
    Ok(())
}

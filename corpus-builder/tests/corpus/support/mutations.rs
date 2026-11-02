use parquet::{
    data_type::{ByteArray, ByteArrayType, Int32Type, Int64Type},
    file::{
        properties::WriterProperties,
        reader::{FileReader, SerializedFileReader},
        writer::SerializedFileWriter,
    },
    record::{Field, Row},
    schema::parser::parse_message_type,
};
use safetensors::{Dtype, SafeTensors, tensor::TensorInfo};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

pub fn copy_release(source: &Path, destination: &Path) {
    fs::create_dir(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), destination.join(entry.file_name())).unwrap();
    }
}

// Rebind every payload and both release identities, including changed sizes.
pub fn rebound(stage: &Path) -> PathBuf {
    let manifest_path = stage.join("manifest.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let mut identity = Sha256::new();
    for (name, entry) in manifest["files"].as_object_mut().unwrap() {
        let path = stage.join(name);
        let digest = ff_corpus_builder::build::sha256(&path).unwrap();
        entry["sha256"] = json!(digest);
        entry["bytes"] = json!(path.metadata().unwrap().len());
        identity.update(name.as_bytes());
        identity.update(digest.as_bytes());
    }
    manifest["corpus_id"] = json!(hex::encode(identity.finalize())[..20].to_string());
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let destination =
        stage.with_file_name(ff_corpus_builder::build::sha256(&manifest_path).unwrap());
    fs::rename(stage, &destination).unwrap();
    destination
}

pub struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    pub bytes: Vec<u8>,
}
impl Tensor {
    pub fn i32(&self) -> Vec<i32> {
        assert_eq!(self.dtype, Dtype::I32);
        self.bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    pub fn set_i32(&mut self, values: &[i32]) {
        assert_eq!(self.dtype, Dtype::I32);
        self.shape = vec![values.len()];
        self.bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    }
}

pub fn tensors(path: &Path, change: impl FnOnce(&mut BTreeMap<String, Tensor>)) {
    let original = fs::read(path).unwrap();
    let (_, metadata) = SafeTensors::read_metadata(&original).unwrap();
    let decoded = SafeTensors::deserialize(&original).unwrap();
    let mut values = decoded
        .tensors()
        .into_iter()
        .map(|(name, tensor)| {
            (
                name,
                Tensor {
                    dtype: tensor.dtype(),
                    shape: tensor.shape().to_vec(),
                    bytes: tensor.data().to_vec(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    change(&mut values);
    let mut names = values.keys().collect::<Vec<_>>();
    names.sort_by_key(|name| (std::cmp::Reverse(values[*name].dtype.bitsize()), *name));
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!(metadata.metadata()));
    let mut payload = Vec::new();
    for name in names {
        let tensor = &values[name];
        let start = payload.len();
        payload.extend_from_slice(&tensor.bytes);
        header.insert(
            name.clone(),
            serde_json::to_value(TensorInfo {
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                data_offsets: (start, payload.len()),
            })
            .unwrap(),
        );
    }
    let mut encoded = serde_json::to_vec(&header).unwrap();
    encoded.resize((encoded.len() + 7) & !7, b' ');
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.extend(payload);
    fs::write(path, bytes).unwrap();
}

pub fn rows(path: &Path) -> Vec<Row> {
    SerializedFileReader::new(fs::File::open(path).unwrap())
        .unwrap()
        .get_row_iter(None)
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

pub fn replace_field(row: &Row, name: &str, field: Field) -> Row {
    Row::new(
        row.get_column_iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    if key == name {
                        field.clone()
                    } else {
                        value.clone()
                    },
                )
            })
            .collect(),
    )
}

// An independent small writer controls row groups without making encoding bytes
// part of logical comparison. Schemas mirror the published typed wire contract.
pub fn write_rows(path: &Path, rows: &[Row], group_size: usize, narrow_tag_id: bool) {
    let vocabulary = path.file_name().unwrap() == "vocab.parquet";
    let schema = if vocabulary {
        if narrow_tag_id {
            "message vocabulary { REQUIRED INT32 idx (INTEGER(32,true)); REQUIRED INT32 tag_id (INTEGER(32,true)); REQUIRED BINARY name (STRING); REQUIRED INT32 category (INTEGER(32,true)); REQUIRED INT64 occ (INTEGER(64,true)); }"
        } else {
            "message vocabulary { REQUIRED INT32 idx (INTEGER(32,true)); REQUIRED INT64 tag_id (INTEGER(64,true)); REQUIRED BINARY name (STRING); REQUIRED INT32 category (INTEGER(32,true)); REQUIRED INT64 occ (INTEGER(64,true)); }"
        }
    } else {
        "message aliases { REQUIRED BINARY alias (STRING); REQUIRED INT32 canonical_idx (INTEGER(32,true)); }"
    };
    let mut writer = SerializedFileWriter::new(
        fs::File::create(path).unwrap(),
        Arc::new(parse_message_type(schema).unwrap()),
        Arc::new(WriterProperties::builder().build()),
    )
    .unwrap();
    for chunk in rows.chunks(group_size) {
        let mut group = writer.next_row_group().unwrap();
        for column_index in 0..if vocabulary { 5 } else { 2 } {
            let fields = chunk
                .iter()
                .map(|row| row.get_column_iter().nth(column_index).unwrap().1)
                .collect::<Vec<_>>();
            let mut column = group.next_column().unwrap().unwrap();
            match fields[0] {
                Field::Int(_) => {
                    let values = fields
                        .iter()
                        .map(|f| match f {
                            Field::Int(v) => *v,
                            _ => panic!("mixed column"),
                        })
                        .collect::<Vec<_>>();
                    column
                        .typed::<Int32Type>()
                        .write_batch(&values, None, None)
                        .unwrap();
                }
                Field::Long(_) => {
                    let values = fields
                        .iter()
                        .map(|f| match f {
                            Field::Long(v) => *v,
                            _ => panic!("mixed column"),
                        })
                        .collect::<Vec<_>>();
                    column
                        .typed::<Int64Type>()
                        .write_batch(&values, None, None)
                        .unwrap();
                }
                Field::Str(_) => {
                    let values = fields
                        .iter()
                        .map(|f| match f {
                            Field::Str(v) => ByteArray::from(v.as_str()),
                            _ => panic!("mixed column"),
                        })
                        .collect::<Vec<_>>();
                    column
                        .typed::<ByteArrayType>()
                        .write_batch(&values, None, None)
                        .unwrap();
                }
                _ => panic!("unsupported fixture column"),
            }
            column.close().unwrap();
        }
        group.close().unwrap();
    }
    writer.close().unwrap();
}

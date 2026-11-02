use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    fs::File,
    path::Path,
};

use anyhow::{Context, Result, bail, ensure};
use parquet::{
    file::{reader::FileReader, serialized_reader::SerializedFileReader},
    record::{Field, Row},
};

use crate::{
    corpus::{payload_path, read_manifest},
    types::TagSuggestion,
};

#[derive(Clone, Debug)]
pub struct Tag {
    pub name: String,
    pub category: i32,
    pub post_count: i64,
}

/// Global vocabulary indices are durable corpus indices; aliases point directly to terminal tags.
pub struct TagCatalog {
    tags: Vec<Tag>,
    by_name: HashMap<String, usize>,
}
impl TagCatalog {
    pub fn load(root: &Path) -> Result<Self> {
        let manifest = read_manifest(root)?;
        let reader = parquet_reader(
            &payload_path(root, &manifest, "vocab.parquet")?,
            &["idx", "name", "category", "occ"],
        )?;
        let mut slots: Vec<Option<Tag>> = vec![None; manifest.tag_count];
        for row in reader.get_row_iter(None)? {
            let row = row?;
            let index = integer(&row, "idx")?;
            ensure!(
                index >= 0 && (index as usize) < slots.len(),
                "Vocabulary index {index} outside declared tag count"
            );
            let tag = Tag {
                name: normalize(string(&row, "name")?),
                category: integer(&row, "category")?,
                post_count: long(&row, "occ")?,
            };
            ensure!(
                slots[index as usize].replace(tag).is_none(),
                "Duplicate vocabulary index {index}"
            );
        }
        let tags = slots
            .into_iter()
            .enumerate()
            .map(|(index, tag)| tag.with_context(|| format!("Missing vocabulary index {index}")))
            .collect::<Result<Vec<_>>>()?;
        let mut catalog = Self::from_tags(tags)?;
        let reader = parquet_reader(
            &payload_path(root, &manifest, "aliases.parquet")?,
            &["alias", "canonical_idx"],
        )?;
        for row in reader.get_row_iter(None)? {
            let row = row?;
            let index = integer(&row, "canonical_idx")?;
            ensure!(index >= 0, "Negative canonical alias index");
            catalog.add_alias(string(&row, "alias")?, index as usize)?;
        }
        Ok(catalog)
    }
    pub fn len(&self) -> usize {
        self.tags.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }
    pub fn resolve(&self, name: &str) -> Option<usize> {
        let name = normalize(name);
        self.by_name
            .get(&name)
            .or_else(|| self.by_name.get(&name.replace('-', "_")))
            .copied()
    }
    pub fn tag(&self, index: usize) -> &Tag {
        &self.tags[index]
    }
    /// Exact, prefix, then substring; ties prefer popular tags and then ordinal name order.
    /// Keep only the requested number of candidates instead of sorting the full vocabulary.
    pub fn search(&self, query: &str, limit: usize) -> Vec<TagSuggestion> {
        let query = normalize(query);
        if query.is_empty() || limit == 0 {
            return Vec::new();
        }
        let alternate = query.replace('-', "_");
        let mut best = BinaryHeap::with_capacity(limit.min(self.tags.len()));
        for (index, tag) in self.tags.iter().enumerate() {
            let rank = match (
                match_rank(&tag.name, &query),
                match_rank(&tag.name, &alternate),
            ) {
                (Some(a), Some(b)) => a.min(b),
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => continue,
            };
            let key = (rank, Reverse(tag.post_count), tag.name.as_str(), index);
            if best.len() < limit {
                best.push(key);
            } else if best.peek().is_some_and(|worst| key < *worst) {
                best.pop();
                best.push(key);
            }
        }
        best.into_sorted_vec()
            .into_iter()
            .map(|(_, _, _, index)| {
                let tag = &self.tags[index];
                TagSuggestion {
                    name: tag.name.clone(),
                    category: category_name(tag.category).to_owned(),
                    post_count: tag.post_count,
                }
            })
            .collect()
    }
    fn from_tags(tags: Vec<Tag>) -> Result<Self> {
        let mut by_name = HashMap::with_capacity(tags.len());
        for (index, tag) in tags.iter().enumerate() {
            ensure!(
                !tag.name.is_empty() && tag.post_count >= 0,
                "Empty tag name or negative occurrence count at {index}"
            );
            ensure!(
                by_name.insert(tag.name.clone(), index).is_none(),
                "Duplicate normalized vocabulary tag '{}'",
                tag.name
            );
        }
        Ok(Self { tags, by_name })
    }
    fn add_alias(&mut self, alias: &str, index: usize) -> Result<()> {
        ensure!(
            index < self.tags.len(),
            "Alias '{alias}' points outside vocabulary at {index}"
        );
        let alias = normalize(alias);
        if let Some(existing) = self.by_name.get(&alias) {
            ensure!(
                *existing == index,
                "Alias '{alias}' conflicts with tag {existing}"
            );
        }
        self.by_name.insert(alias, index);
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn fixture(mut tags: Vec<Tag>, aliases: Vec<(String, usize)>) -> Result<Self> {
        for tag in &mut tags {
            tag.name = normalize(&tag.name);
        }
        let mut catalog = Self::from_tags(tags)?;
        for (alias, index) in aliases {
            catalog.add_alias(&alias, index)?;
        }
        Ok(catalog)
    }
}
fn parquet_reader(path: &Path, required: &[&str]) -> Result<SerializedFileReader<File>> {
    let reader = SerializedFileReader::new(File::open(path)?)
        .with_context(|| format!("Reading {}", path.display()))?;
    let schema = reader.metadata().file_metadata().schema_descr();
    for name in required {
        ensure!(
            schema
                .columns()
                .iter()
                .filter(|column| column.name() == *name)
                .count()
                == 1,
            "Parquet file {} requires unique column '{name}'",
            path.display()
        );
    }
    Ok(reader)
}
fn field<'a>(row: &'a Row, name: &str) -> Result<&'a Field> {
    row.get_column_iter()
        .find(|(column, _)| column.as_str() == name)
        .map(|(_, value)| value)
        .with_context(|| format!("Missing Parquet field '{name}'"))
}
fn integer(row: &Row, name: &str) -> Result<i32> {
    match field(row, name)? {
        Field::Int(value) => Ok(*value),
        _ => bail!("Parquet field '{name}' must be non-null Int32"),
    }
}
fn long(row: &Row, name: &str) -> Result<i64> {
    match field(row, name)? {
        Field::Long(value) => Ok(*value),
        _ => bail!("Parquet field '{name}' must be non-null Int64"),
    }
}
fn string<'a>(row: &'a Row, name: &str) -> Result<&'a str> {
    match field(row, name)? {
        Field::Str(value) => Ok(value),
        _ => bail!("Parquet field '{name}' must be non-null string"),
    }
}
fn normalize(value: &str) -> String {
    let value = value.trim().to_lowercase();
    value.strip_prefix(['+', '-']).unwrap_or(&value).to_owned()
}
fn match_rank(name: &str, query: &str) -> Option<u8> {
    if name == query {
        Some(0)
    } else if name.starts_with(query) {
        Some(1)
    } else if name.contains(query) {
        Some(2)
    } else {
        None
    }
}
fn category_name(category: i32) -> &'static str {
    match category {
        0 => "general",
        1 => "artist",
        3 => "copyright",
        4 => "character",
        5 => "species",
        6 => "invalid",
        7 => "meta",
        8 => "lore",
        _ => "unknown",
    }
}

#[cfg(test)]
#[path = "../tests/unit/catalog.rs"]
mod tests;

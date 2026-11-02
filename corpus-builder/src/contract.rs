use std::sync::Arc;

use anyhow::{Result, bail};

#[derive(Debug, Clone)]
pub struct TagDefinition {
    pub source_id: i64,
    pub name: Arc<str>,
    pub category: i32,
}

#[derive(Debug, Clone)]
pub struct VocabularyEntry {
    pub index: i32,
    pub source_id: i64,
    pub name: String,
    pub category: i32,
    pub occurrences: i64,
}

#[derive(Debug, Clone)]
pub struct AliasEntry {
    pub alias: String,
    pub canonical_index: i32,
}

pub const MINIMUM_CONTENT_POSTS: i64 = 50;
pub const MINIMUM_ARTIST_POSTS: i64 = 10;
pub const MINIMUM_METADATA_POSTS: i64 = 50;
pub const CONTENT_CATEGORIES: &[i32] = &[0, 3, 4, 5];

pub const PSEUDO_ARTISTS: &[&str] = &[
    "conditional_dnp",
    "third-party_edit",
    "sound_warning",
    "ai_assisted",
    "ai_generated_background",
    "unknown_artist",
    "anonymous_artist",
];

pub fn extension_code(value: &str) -> Result<u8> {
    match value {
        "jpg" => Ok(1),
        "png" => Ok(2),
        "gif" => Ok(3),
        "webm" => Ok(4),
        "mp4" => Ok(6),
        "webp" => Ok(7),
        _ => bail!("Unknown media extension '{value}'."),
    }
}

pub fn rating_code(value: &str) -> Result<u8> {
    match value {
        "s" => Ok(1),
        "q" => Ok(2),
        "e" => Ok(3),
        _ => bail!("Unknown content rating '{value}'."),
    }
}

#[cfg(test)]
#[path = "../tests/unit/contract.rs"]
mod tests;

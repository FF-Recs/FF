use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum Grade {
    Dislike = 0,
    Meh = 1,
    Like = 2,
    Love = 3,
}
impl Grade {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dislike => "dislike",
            Self::Meh => "meh",
            Self::Like => "like",
            Self::Love => "love",
        }
    }
    pub fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "dislike" => Ok(Self::Dislike),
            "meh" => Ok(Self::Meh),
            "like" => Ok(Self::Like),
            "love" => Ok(Self::Love),
            _ => anyhow::bail!("Unknown grade '{value}'"),
        }
    }
    pub fn from_u8(value: u8) -> anyhow::Result<Self> {
        match value {
            0 => Ok(Self::Dislike),
            1 => Ok(Self::Meh),
            2 => Ok(Self::Like),
            3 => Ok(Self::Love),
            _ => anyhow::bail!("Grade outside [0,3]: {value}"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Card {
    pub post_id: i64,
    pub md5: String,
    pub file_extension: String,
    pub content_rating: String,
    pub selection_score: f32,
    pub current_grade: Option<Grade>,
    pub facet: String,
    pub session_id: String,
    pub epoch: u32,
    pub position: usize,
    pub universe_size: usize,
    pub model_id: String,
}
impl Card {
    pub fn media_url(&self) -> String {
        format!(
            "https://static1.e621.net/data/{}/{}/{}.{}",
            &self.md5[..2],
            &self.md5[2..4],
            self.md5,
            self.file_extension
        )
    }
    pub fn alternate_mp4_url(&self) -> String {
        format!(
            "https://static1.e621.net/data/sample/{}/{}/{}_alt.mp4",
            &self.md5[..2],
            &self.md5[2..4],
            self.md5
        )
    }
}

#[derive(Clone, Debug)]
pub struct StoredRating {
    pub post_id: i64,
    pub grade: Grade,
    pub first_rated_at: f64,
    pub updated_at: f64,
    pub revision: i64,
    pub context_json: String,
}
#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub post_id: i64,
    pub predicted_grade: u8,
    pub recorded_grade: Grade,
    pub rating_revision: i64,
    pub model_id: String,
    pub review_sequence: i64,
    pub updated_at: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RatingPrediction {
    pub predicted_grade: u8,
    pub model_id: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct RatingResult {
    pub post_id: i64,
    pub grade: Grade,
    pub revision: i64,
    pub changed: bool,
}
#[derive(Clone, Debug)]
pub struct RatingContext {
    pub json: String,
    pub is_auto: bool,
    pub facet: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Health {
    pub corpus_id: String,
    pub model_id: String,
    pub model_status: crate::model::ModelStatus,
    pub rating_count: usize,
    pub refreshing: bool,
    pub refresh_error: Option<String>,
}
#[derive(Clone, Debug)]
pub struct Facet {
    pub slug: &'static str,
    pub label: &'static str,
    pub cohort_size: usize,
    pub requires_model: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct TagSuggestion {
    pub name: String,
    pub category: String,
    pub post_count: i64,
}
#[derive(Clone, Debug)]
pub struct GalleryStartRequest {
    pub facet: String,
    pub page_size: usize,
    pub filter: String,
    pub exclude_post_ids: Vec<i64>,
}
#[derive(Clone, Debug)]
pub struct GalleryNextRequest {
    pub session_id: String,
    pub page_size: usize,
    pub exclude_post_ids: Vec<i64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Gallery {
    pub session_id: String,
    pub facet: String,
    pub items: Vec<Card>,
    pub exhausted: bool,
}
#[derive(Clone, Debug)]
pub struct AutoStartRequest {
    pub preset: String,
    pub page_size: usize,
    pub seed: i32,
    pub filter: String,
    pub exclude_post_ids: Vec<i64>,
}
#[derive(Clone, Debug)]
pub struct AutoNextRequest {
    pub session_id: String,
    pub page_size: usize,
    pub exclude_post_ids: Vec<i64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct AutoPage {
    pub session_id: String,
    pub items: Vec<Card>,
    pub seed: i32,
    pub epoch: u32,
    pub universe_size: usize,
    pub emitted_count: usize,
    pub exhausted: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserSettings {
    pub left_handed: bool,
    pub video_muted: bool,
    pub auto_preset: String,
    pub auto_delay_seconds: u32,
    #[serde(default)]
    pub blacklist: Vec<String>,
}
impl Default for UserSettings {
    fn default() -> Self {
        Self {
            left_handed: false,
            video_muted: true,
            auto_preset: "balanced".into(),
            auto_delay_seconds: 5,
            blacklist: Vec::new(),
        }
    }
}

/// A durable, actually presented navigation entry. Its offer is never reranked.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrowseEntry {
    pub entry_id: String,
    pub card: Card,
}

/// A UI context spans recommendation cohorts; lookahead is not durable history.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrowseContext {
    pub context_id: String,
    pub mode: String,
    pub filter: String,
    pub entries: Vec<BrowseEntry>,
    pub cursor: Option<String>,
}

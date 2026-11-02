use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::types::{
    BrowseContext, BrowseEntry, Card, Checkpoint, Grade, RatingContext, RatingPrediction,
    RatingResult, StoredRating, UserSettings,
};

mod facts;
mod history;
mod seed;

#[cfg(test)]
#[path = "../tests/unit/store/fixtures.rs"]
pub(crate) mod fixtures;

const SCHEMA: &str = include_str!("schema.sql");
const SCHEMA_VERSION: i64 = 13;

pub(crate) struct GalleryOwner<'a> {
    pub session: &'a str,
    pub facet: &'a str,
    pub filter: &'a str,
    pub capacity: usize,
    pub issued: usize,
}

/// Owned by the application actor; no connection pool or background database writer.
pub struct Store {
    pub path: PathBuf,
    connection: Connection,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path =
            std::path::absolute(path.as_ref()).context("Resolving interaction database path")?;
        let connection = (|| -> Result<Connection> {
            fs::create_dir_all(
                path.parent()
                    .context("Interaction database has no parent")?,
            )?;
            let mut connection = Connection::open(&path)?;
            connection.busy_timeout(Duration::from_secs(5))?;
            // Validate the supported layout before changing persistent pragmas.
            history::open_schema(&mut connection)?;
            connection.pragma_update(None, "foreign_keys", true)?;
            let journal: String =
                connection.pragma_query_value(None, "journal_mode", |row| row.get(0))?;
            if !journal.eq_ignore_ascii_case("wal") {
                let mode: String =
                    connection.query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))?;
                ensure!(
                    mode.eq_ignore_ascii_case("wal"),
                    "SQLite refused WAL journal mode: {mode}"
                );
            }
            connection.pragma_update(None, "synchronous", "FULL")?;
            Ok(connection)
        })()
        .with_context(|| format!("Opening interaction database {}", path.display()))?;
        Ok(Self { path, connection })
    }

    pub fn ratings(&self) -> Result<Vec<StoredRating>> {
        facts::ratings(&self.connection)
            .with_context(|| format!("Reading ratings from {}", self.path.display()))
    }

    /// Materialize one consistent ratings read without holding a reader during fitting.
    pub(crate) fn snapshot_ratings_at(path: &Path) -> Result<Vec<StoredRating>> {
        let phase = Instant::now();
        let result = (|| -> Result<Vec<StoredRating>> {
            let mut source = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            source.busy_timeout(Duration::from_secs(5))?;
            let transaction = source
                .transaction_with_behavior(TransactionBehavior::Deferred)
                .context("Starting model snapshot read transaction")?;
            let ratings = facts::ratings(&transaction)?;
            transaction
                .commit()
                .context("Releasing model snapshot read transaction")?;
            Ok(ratings)
        })();
        tracing::info!(
            phase = "snapshot_total",
            elapsed_seconds = phase.elapsed().as_secs_f64(),
            "Model snapshot phase"
        );
        result.with_context(|| format!("Snapshotting ratings from {}", path.display()))
    }

    pub fn unavailable(&self) -> Result<HashSet<i64>> {
        facts::unavailable(&self.connection).context("Reading rated and explicitly viewed post IDs")
    }

    pub fn recent_views(&self, limit: usize) -> Result<Vec<i64>> {
        facts::recent_views(&self.connection, limit)
            .with_context(|| format!("Reading recent views from {}", self.path.display()))
    }

    pub fn checkpoints(&self) -> Result<HashMap<i64, Checkpoint>> {
        facts::checkpoints(&self.connection)
            .with_context(|| format!("Reading Review checkpoints from {}", self.path.display()))
    }

    pub fn review_exclusions(&self) -> Result<HashSet<i64>> {
        facts::review_exclusions(&self.connection).context("Reading permanent Review exclusions")
    }

    pub fn review_sequence(&self) -> Result<i64> {
        facts::review_sequence(&self.connection)
            .with_context(|| format!("Reading Review sequence from {}", self.path.display()))
    }

    #[cfg(test)]
    pub fn create_gallery(
        &mut self,
        session: &str,
        facet: &str,
        filter: &str,
        model_id: &str,
        requested: usize,
        cards: &[Card],
        exhausted: bool,
    ) -> Result<()> {
        self.commit_gallery(
            GalleryOwner {
                session,
                facet,
                filter,
                capacity: requested,
                issued: 0,
            },
            model_id,
            cards,
            cards.len() == requested || exhausted,
            exhausted,
            None,
        )
    }

    /// Create, append, or close a cohort atomically, optionally retiring its
    /// predecessor in the same transaction. Existing offer/header bytes are immutable.
    pub(crate) fn commit_gallery(
        &mut self,
        owner: GalleryOwner<'_>,
        model_id: &str,
        cards: &[Card],
        completed: bool,
        exhausted: bool,
        previous: Option<(GalleryOwner<'_>, bool)>,
    ) -> Result<()> {
        let offers = capture_offers(
            owner.session,
            owner.issued,
            cards,
            crate::selection::GALLERY_POLICY,
        )?;
        self.commit_generated(Event::GalleryCommitted {
            owner: owner.into(),
            model_id: model_id.into(),
            offers,
            completed,
            exhausted,
            previous: previous.map(|(owner, completed)| (owner.into(), completed)),
        })
    }

    pub(crate) fn retire_gallery(&mut self, owner: GalleryOwner<'_>) -> Result<()> {
        self.commit_generated(Event::GalleryRetired {
            owner: owner.into(),
        })
    }

    pub fn create_auto(
        &mut self,
        session: &str,
        preset: &str,
        filter: &str,
        model_id: &str,
        seed: i32,
        cards: &[Card],
    ) -> Result<()> {
        self.commit_generated(Event::AutoCreated {
            session: session.into(),
            preset: preset.into(),
            filter: filter.into(),
            model_id: model_id.into(),
            seed,
            offers: capture_offers(session, 0, cards, crate::selection::AUTO_POLICY)?,
        })
    }

    pub fn append_auto(&mut self, session: &str, start: usize, cards: &[Card]) -> Result<()> {
        self.commit_generated(Event::AutoAppended {
            session: session.into(),
            start,
            offers: capture_offers(session, start, cards, crate::selection::AUTO_POLICY)?,
        })
    }

    pub fn context(&self, session: &str, post_id: i64) -> Result<RatingContext> {
        facts::context(&self.connection, session, post_id)
    }

    /// Record the user's offered-context rating, optionally with a genuine
    /// model forecast. Retry identity excludes the current serving prediction.
    pub fn rate(
        &mut self,
        event_id: &str,
        post_id: i64,
        grade: Grade,
        session: &str,
        prediction: Option<RatingPrediction>,
    ) -> Result<RatingResult> {
        let (value, retried) = self.commit_event(
            event_id,
            Event::RatingRecorded {
                post_id,
                grade,
                session: session.into(),
                prediction,
            },
        )?;
        Ok(RatingResult {
            post_id,
            grade,
            revision: value["revision"]
                .as_i64()
                .context("Rating event lacks revision")?,
            changed: !retried
                && value["changed"]
                    .as_bool()
                    .context("Rating event lacks result")?,
        })
    }

    pub fn viewed(&mut self, event_id: &str, post_id: i64, session: &str) -> Result<()> {
        self.commit_event(
            event_id,
            Event::Viewed {
                post_id,
                session: session.into(),
            },
        )?;
        Ok(())
    }

    pub fn contains_event(&self, event_id: &str) -> Result<bool> {
        Ok(self.connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM interaction_events WHERE event_id=?1)",
            [event_id],
            |row| row.get(0),
        )?)
    }

    pub fn preferences(&self) -> Result<Option<UserSettings>> {
        facts::preferences(&self.connection)
    }

    pub fn save_preferences(&mut self, event_id: &str, settings: &UserSettings) -> Result<()> {
        self.commit_event(
            event_id,
            Event::PreferencesSaved {
                settings: settings.clone(),
            },
        )?;
        Ok(())
    }

    pub fn active_context(&self) -> Result<Option<BrowseContext>> {
        facts::active_context(&self.connection)
    }

    pub fn activate_context(
        &mut self,
        event_id: &str,
        mode: &str,
        filter: &str,
        fresh: bool,
    ) -> Result<BrowseContext> {
        let (result, _) = self.commit_event(
            event_id,
            Event::ContextActivated {
                mode: mode.into(),
                filter: filter.into(),
                fresh,
            },
        )?;
        // An old activation retry returns that context without activating it again.
        facts::read_context(
            &self.connection,
            result["context_id"]
                .as_str()
                .context("Activation lacks context ID")?,
        )
    }

    pub fn presented(
        &mut self,
        event_id: &str,
        context_id: &str,
        entry_id: Option<&str>,
        card: &Card,
    ) -> Result<()> {
        let mut card = card.clone();
        card.current_grade = None;
        self.commit_event(
            event_id,
            Event::Presented {
                context_id: context_id.into(),
                entry_id: entry_id.map(str::to_owned),
                card,
            },
        )?;
        Ok(())
    }

    /// Verify immutable history and derived state by replaying into an isolated store.
    /// This never repairs or rewrites facts in the profile.
    pub fn verify_history(&self) -> Result<()> {
        history::verify_history(&self.connection)
    }

    fn commit_generated(&mut self, event: Event) -> Result<()> {
        self.commit_event(&uuid::Uuid::new_v4().to_string(), event)?;
        Ok(())
    }

    fn commit_event(&mut self, event_id: &str, event: Event) -> Result<(Value, bool)> {
        required_text(event_id, "Interaction event ID")?;
        let payload = serde_json::to_string(&event)?;
        let request = serde_json::to_string(&event.request(&payload)?)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let previous: Option<(String, String)> = transaction
            .query_row(
                "SELECT request_json,result_json FROM interaction_events WHERE event_id=?1",
                [event_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((saved, result)) = previous {
            ensure!(
                saved == request,
                "Interaction event '{event_id}' conflicts with an earlier request"
            );
            let result = serde_json::from_str(&result)?;
            transaction.commit()?;
            return Ok((result, true));
        }
        let imported_view = facts::imported_view_reserved(&transaction, event_id)?;
        ensure!(
            !imported_view || matches!(&event, Event::Viewed { .. }),
            "Interaction event '{event_id}' is reserved by an imported view"
        );
        let sequence: i64 = transaction.query_row(
            "SELECT COALESCE(MAX(sequence),0)+1 FROM interaction_events",
            [],
            |row| row.get(0),
        )?;
        let timestamp = now()?;
        let result = facts::apply(&transaction, event_id, sequence, timestamp, &event)?;
        transaction.execute(
            "INSERT INTO interaction_events(sequence,event_id,occurred_at,request_json,payload_json,result_json) VALUES(?1,?2,?3,?4,?5,?6)",
            params![sequence, event_id, timestamp, request,
                payload, serde_json::to_string(&result)?],
        )?;
        transaction.commit()?;
        Ok((result, false))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Event {
    Initialized {
        facts: seed::InitialFacts,
    },
    GalleryCommitted {
        owner: OwnedGallery,
        model_id: String,
        offers: Vec<OfferFact>,
        completed: bool,
        exhausted: bool,
        previous: Option<(OwnedGallery, bool)>,
    },
    GalleryRetired {
        owner: OwnedGallery,
    },
    AutoCreated {
        session: String,
        preset: String,
        filter: String,
        model_id: String,
        seed: i32,
        offers: Vec<OfferFact>,
    },
    AutoAppended {
        session: String,
        start: usize,
        offers: Vec<OfferFact>,
    },
    RatingRecorded {
        post_id: i64,
        grade: Grade,
        session: String,
        prediction: Option<RatingPrediction>,
    },
    Viewed {
        post_id: i64,
        session: String,
    },
    PreferencesSaved {
        settings: UserSettings,
    },
    ContextActivated {
        mode: String,
        filter: String,
        fresh: bool,
    },
    Presented {
        context_id: String,
        entry_id: Option<String>,
        card: Card,
    },
}

impl Event {
    fn request(&self, payload: &str) -> Result<Value> {
        // Fingerprint the already-serialized facts instead of copying the entire
        // baseline or offer batch into a second JSON tree and database column.
        let rating = match self {
            // Frozen user-rating retry domain, independent of transport encoding.
            Self::RatingRecorded {
                post_id,
                grade,
                session,
                ..
            } => Some(serde_json::to_vec(
                // Frozen command hash domain: existing event IDs must remain retryable.
                &json!({"kind":"rated","post_id":post_id,"grade":grade,"session":session}),
            )?),
            _ => None,
        };
        let bytes = rating.as_deref().unwrap_or(payload.as_bytes());
        Ok(json!({"sha256": hex::encode(Sha256::digest(bytes))}))
    }
}

fn decode_event(payload: &str) -> Result<Event> {
    Ok(serde_json::from_str(payload)?)
}

/// Offer evidence is captured once, before entering the reducer.
/// Replay never reserializes a current Card or consults current selection policy.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OfferFact {
    post_id: i64,
    position: usize,
    facet: String,
    payload_json: String,
}

fn capture_offers(
    session: &str,
    start: usize,
    cards: &[Card],
    policy: &str,
) -> Result<Vec<OfferFact>> {
    validate_cards(session, start, cards)?;
    required_text(policy, "Offer selection policy")?;
    cards
        .iter()
        .map(|card| {
            Ok(OfferFact {
                post_id: card.post_id,
                position: card.position,
                facet: card.facet.clone(),
                payload_json: offered_payload(card, policy)?,
            })
        })
        .collect()
}

fn validate_offers(session: &str, start: usize, offers: &[OfferFact]) -> Result<()> {
    for (offset, offer) in offers.iter().enumerate() {
        ensure!(offer.post_id > 0, "Offered post ID must be positive");
        ensure!(
            Some(offer.position) == start.checked_add(offset),
            "Offered post {} has noncontiguous position",
            offer.post_id
        );
        let payload: Value = serde_json::from_str(&offer.payload_json)?;
        ensure!(
            payload["format_version"].as_u64() == Some(1),
            "Unsupported event offer format"
        );
        required_text(
            payload["selection_policy"]
                .as_str()
                .context("Offer lacks recorded policy")?,
            "Offer selection policy",
        )?;
        let card = &payload["card"];
        ensure!(
            card["session_id"].as_str() == Some(session)
                && card["post_id"].as_i64() == Some(offer.post_id)
                && card["position"].as_u64() == Some(u64::try_from(offer.position)?)
                && card["facet"].as_str() == Some(offer.facet.as_str()),
            "Recorded offer identity disagrees with its event"
        );
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedGallery {
    session: String,
    facet: String,
    filter: String,
    capacity: usize,
    issued: usize,
}

impl From<GalleryOwner<'_>> for OwnedGallery {
    fn from(owner: GalleryOwner<'_>) -> Self {
        Self {
            session: owner.session.into(),
            facet: owner.facet.into(),
            filter: owner.filter.into(),
            capacity: owner.capacity,
            issued: owner.issued,
        }
    }
}

impl OwnedGallery {
    fn borrowed(&self) -> GalleryOwner<'_> {
        GalleryOwner {
            session: &self.session,
            facet: &self.facet,
            filter: &self.filter,
            capacity: self.capacity,
            issued: self.issued,
        }
    }
}

fn required_text(value: &str, name: &str) -> Result<()> {
    ensure!(!value.trim().is_empty(), "{name} is required");
    Ok(())
}

fn validate_cards(session: &str, start: usize, cards: &[Card]) -> Result<()> {
    for (offset, card) in cards.iter().enumerate() {
        ensure!(card.post_id > 0, "Offered post ID must be positive");
        ensure!(
            card.session_id == session,
            "Offered post {} belongs to another session",
            card.post_id
        );
        ensure!(
            Some(card.position) == start.checked_add(offset),
            "Offered post {} has noncontiguous position",
            card.post_id
        );
        ensure!(
            card.selection_score.is_finite(),
            "Offered post {} has nonfinite selection score",
            card.post_id
        );
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct OfferedCard<'a> {
    format_version: u8,
    selection_policy: &'a str,
    card: &'a Card,
}

fn offered_payload(card: &Card, selection_policy: &str) -> serde_json::Result<String> {
    serde_json::to_string(&OfferedCard {
        format_version: 1,
        selection_policy,
        card,
    })
}

#[derive(Default)]
struct OfferedIdentity {
    md5: Option<String>,
    extension: Option<String>,
    model_id: Option<String>,
    selection_policy: Option<String>,
}

fn valid_md5(value: &str) -> bool {
    value.len() == 32 && value.bytes().all(|c| c.is_ascii_hexdigit())
}

fn valid_extension(value: &str) -> bool {
    !value.is_empty() && value.len() <= 10 && value.bytes().all(|c| c.is_ascii_alphanumeric())
}

fn payload_identity(payload: &str) -> Result<OfferedIdentity> {
    let Ok(value) = serde_json::from_str::<Value>(payload) else {
        return Ok(OfferedIdentity::default());
    };
    if let Some(version) = value.get("format_version") {
        ensure!(
            version.as_u64() == Some(1),
            "Unsupported native offered-card format"
        );
        let card = value.get("card").context("Native offer lacks its card")?;
        let md5 = card
            .get("md5")
            .and_then(Value::as_str)
            .filter(|value| valid_md5(value))
            .context("Native offer has invalid media MD5")?;
        let extension = card
            .get("file_extension")
            .and_then(Value::as_str)
            .filter(|value| valid_extension(value))
            .context("Native offer has invalid media extension")?;
        let model_id = card
            .get("model_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .context("Native offer lacks its model identity")?;
        let policy = value
            .get("selection_policy")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .context("Native offer lacks its selection policy")?;
        return Ok(OfferedIdentity {
            md5: Some(md5.to_ascii_lowercase()),
            extension: Some(extension.to_ascii_lowercase()),
            model_id: Some(model_id.to_owned()),
            selection_policy: Some(policy.to_owned()),
        });
    }
    Ok(historical_payload_identity(&value))
}

// Historical offers are immutable evidence, not current Card/model transport.
// Extract only recorded provenance; never fabricate media or a selection policy.
fn historical_payload_identity(value: &Value) -> OfferedIdentity {
    let root = value
        .get("card")
        .filter(|v| v.is_object())
        .or_else(|| value.get("Card").filter(|v| v.is_object()))
        .unwrap_or(value);
    let text = |name, alias| {
        root.get(name)
            .or_else(|| root.get(alias))
            .and_then(Value::as_str)
    };
    OfferedIdentity {
        md5: text("Md5", "md5")
            .filter(|value| valid_md5(value))
            .map(str::to_ascii_lowercase),
        extension: text("FileExtension", "file_ext")
            .filter(|value| valid_extension(value))
            .map(str::to_ascii_lowercase),
        model_id: text("ModelId", "model_id")
            .filter(|value| !value.trim().is_empty())
            .map(str::to_owned),
        selection_policy: None,
    }
}

fn now() -> Result<f64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("System clock precedes Unix epoch")?
        .as_secs_f64())
}

#[cfg(test)]
#[path = "../tests/unit/store/mod.rs"]
mod tests;

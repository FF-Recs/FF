//! Synthetic current-schema journals initialized with native immutable facts.
use super::*;

pub(super) fn initialized(path: &Path, facts: seed::InitialFacts) -> Result<Store> {
    let connection = Connection::open(path)?;
    connection.execute_batch(SCHEMA)?;
    connection.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    connection.pragma_update(None, "foreign_keys", true)?;
    let mut store = Store {
        path: path.to_path_buf(),
        connection,
    };
    store.commit_event("fixture-initialized", Event::Initialized { facts })?;
    drop(store);
    Store::open(path)
}

pub(super) fn seeded(
    path: &Path,
    populate: impl FnOnce(&Connection) -> Result<()>,
) -> Result<Store> {
    let connection = Connection::open_in_memory()?;
    connection.pragma_update(None, "foreign_keys", false)?;
    connection.execute_batch(SCHEMA)?;
    populate(&connection)?;
    initialized(path, seed::capture(&connection)?)
}

pub(crate) fn review_history(path: &Path, post_ids: &[i64], review_sequence: i64) -> Result<Store> {
    seeded(path, |connection| {
        for (ordinal, post_id) in post_ids.iter().enumerate() {
            connection.execute(
                "INSERT INTO rating_facts VALUES(1,?1,'seed',?2,'like',1.0,1.0,1,'{}',NULL,NULL,0,'like',1,'seed-model',0,1.0)",
                params![ordinal as i64, post_id],
            )?;
        }
        connection.execute(
            "INSERT INTO review_facts VALUES(1,0,'counter_seed',NULL,NULL,?1,NULL,NULL)",
            [review_sequence],
        )?;
        Ok(())
    })
}

// Boundary setup retains native envelopes and reducer results, but pays for one
// FULL-durability commit rather than two commits for every preparatory vote.
pub(crate) fn review_counter_seed(store: &mut Store, count: usize, card: Card) -> Result<()> {
    let transaction = store
        .connection
        .transaction_with_behavior(TransactionBehavior::Immediate)?;
    let first: i64 = transaction.query_row(
        "SELECT MAX(sequence)+1 FROM interaction_events",
        [],
        |row| row.get(0),
    )?;
    for index in 0..count {
        let session = format!("boundary-review-{first}-{index}");
        let mut offered = card.clone();
        offered.session_id = session.clone();
        offered.facet = "rating_review".into();
        offered.position = 0;
        let owner = GalleryOwner {
            session: &session,
            facet: "rating_review",
            filter: "",
            capacity: 1,
            issued: 0,
        };
        let offers = capture_offers(&session, 0, &[offered], crate::selection::GALLERY_POLICY)?;
        append_boundary_event(
            &transaction,
            &format!("{session}-offer"),
            Event::GalleryCommitted {
                owner: owner.into(),
                model_id: card.model_id.clone(),
                offers,
                completed: true,
                exhausted: false,
                previous: None,
            },
        )?;
        append_boundary_event(
            &transaction,
            &format!("{session}-vote"),
            Event::RatingRecorded {
                post_id: card.post_id,
                grade: Grade::Love,
                session,
                prediction: None,
            },
        )?;
    }
    transaction.commit()?;
    store.verify_history()
}

// Offers must already exist. Only preparation is batched; the caller's final
// view still crosses the recency boundary through Store::viewed normally.
pub(crate) fn recent_view_seed(store: &mut Store, cards: &[Card]) -> Result<()> {
    let transaction = store
        .connection
        .transaction_with_behavior(TransactionBehavior::Immediate)?;
    let first: i64 = transaction.query_row(
        "SELECT MAX(sequence)+1 FROM interaction_events",
        [],
        |row| row.get(0),
    )?;
    for (index, card) in cards.iter().enumerate() {
        append_boundary_event(
            &transaction,
            &format!("boundary-view-{first}-{index}"),
            Event::Viewed {
                post_id: card.post_id,
                session: card.session_id.clone(),
            },
        )?;
    }
    transaction.commit()?;
    store.verify_history()
}

fn append_boundary_event(connection: &Connection, id: &str, event: Event) -> Result<()> {
    let sequence: i64 = connection.query_row(
        "SELECT MAX(sequence)+1 FROM interaction_events",
        [],
        |row| row.get(0),
    )?;
    let timestamp = now()?;
    let payload = serde_json::to_string(&event)?;
    let request = serde_json::to_string(&event.request(&payload)?)?;
    let result = facts::apply(connection, id, sequence, timestamp, &event)?;
    connection.execute(
        "INSERT INTO interaction_events(sequence,event_id,occurred_at,request_json,payload_json,result_json) VALUES(?1,?2,?3,?4,?5,?6)",
        params![sequence, id, timestamp, request, payload, serde_json::to_string(&result)?],
    )?;
    Ok(())
}

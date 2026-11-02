use super::*;

pub(super) const FACT_TABLES: &[&str] = &[
    "interaction_events",
    "selection_facts",
    "offer_facts",
    "rating_facts",
    "review_facts",
    "presentation_facts",
    "activation_facts",
    "preference_facts",
    "legacy_delivery_facts",
];

pub(super) fn fixture() -> Result<tempfile::TempDir> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.workdir/ff-rs/store");
    fs::create_dir_all(&root)?;
    Ok(tempfile::Builder::new()
        .prefix("store-test-")
        .tempdir_in(root)?)
}

pub(super) fn prediction(predicted_grade: u8, model_id: &str) -> RatingPrediction {
    RatingPrediction {
        predicted_grade,
        model_id: model_id.into(),
    }
}

pub(super) fn card(session: &str, facet: &str, post_id: i64, position: usize) -> Card {
    Card {
        post_id,
        md5: "0123456789abcdef0123456789abcdef".into(),
        file_extension: "png".into(),
        content_rating: "s".into(),
        selection_score: 1.5,
        current_grade: None,
        facet: facet.into(),
        session_id: session.into(),
        epoch: 0,
        position,
        universe_size: 3,
        model_id: "offered-model".into(),
    }
}

pub(super) fn scalar_i64(connection: &Connection, sql: &str) -> Result<i64> {
    Ok(connection.query_row(sql, [], |row| row.get(0))?)
}

pub(super) fn raw_rows(connection: &Connection, sql: &str) -> Result<Vec<Vec<SqliteValue>>> {
    let mut statement = connection.prepare(sql)?;
    let count = statement.column_count();
    let rows = statement
        .query_map([], |row| {
            (0..count)
                .map(|column| row.get::<_, SqliteValue>(column))
                .collect::<rusqlite::Result<Vec<_>>>()
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

pub(super) fn durable_rows(connection: &Connection) -> Result<Vec<Vec<Vec<SqliteValue>>>> {
    FACT_TABLES
        .iter()
        .map(|table| {
            raw_rows(
                connection,
                &format!("SELECT * FROM main.{table} ORDER BY 1,2"),
            )
        })
        .collect()
}

pub(super) fn event_count(store: &Store) -> Result<i64> {
    scalar_i64(&store.connection, "SELECT COUNT(*) FROM interaction_events")
}

// A deterministic native append, including the production reducer and deferred envelope FK.
pub(super) fn append_at(store: &mut Store, id: &str, timestamp: f64, event: Event) -> Result<()> {
    let payload = format!("\n{}\n", serde_json::to_string_pretty(&event)?);
    let request = serde_json::to_string(&event.request(&payload)?)?;
    let transaction = store
        .connection
        .transaction_with_behavior(TransactionBehavior::Immediate)?;
    let sequence = scalar_i64(
        &transaction,
        "SELECT COALESCE(MAX(sequence),0)+1 FROM interaction_events",
    )?;
    let result = facts::apply(&transaction, id, sequence, timestamp, &event)?;
    transaction.execute("INSERT INTO interaction_events(sequence,event_id,occurred_at,request_json,payload_json,result_json)
        VALUES(?1,?2,?3,?4,?5,?6)", params![sequence, id, timestamp, request, payload, format!(" {} ", serde_json::to_string(&result)?)])?;
    transaction.commit()?;
    Ok(())
}

pub(super) fn populated_current_profile(path: &Path) -> Result<Store> {
    let mut store = fixtures::seeded(path, |connection| {
        connection.execute_batch(
            "INSERT INTO legacy_delivery_facts VALUES(1,0,'gallery_delivery',99,'orphan',1.25)",
        )?;
        Ok(())
    })?;
    let offered = card("review", "rating_review", 1, 0);
    store.create_gallery(
        "review",
        "rating_review",
        "",
        "model",
        1,
        &[offered.clone()],
        false,
    )?;
    let context = store.activate_context("activate", "rating_review", "", false)?;
    store.presented("present", &context.context_id, None, &offered)?;
    append_at(
        &mut store,
        "old-vote",
        9.25,
        Event::RatingRecorded {
            post_id: 1,
            grade: Grade::Love,
            session: "review".into(),
            prediction: Some(prediction(2, "old-judgment")),
        },
    )?;
    store.rate("new-vote", 1, Grade::Like, "review", None)?;
    store.save_preferences("settings", &UserSettings::default())?;
    for table in FACT_TABLES {
        assert!(
            scalar_i64(
                &store.connection,
                &format!("SELECT COUNT(*) FROM main.{table}")
            )? > 0
        );
    }
    Ok(store)
}

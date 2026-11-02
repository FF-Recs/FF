use super::*;

#[test]
fn stable_retries_never_reapply_old_ratings_cursors_preferences_or_activations() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("retry.sqlite3");
    let mut store = Store::open(&path)?;
    let cards = [
        card("gallery", "discover", 1, 0),
        card("gallery", "discover", 2, 1),
    ];
    store.create_gallery("gallery", "discover", "", "model", 2, &cards, false)?;
    let context = store.activate_context("activate-old", "discover", "", false)?;
    store.presented("show-first", &context.context_id, None, &cards[0])?;
    store.presented("show-second", &context.context_id, None, &cards[1])?;
    let first = store.rate(
        "vote-old",
        1,
        Grade::Love,
        "gallery",
        Some(prediction(2, "first-model")),
    )?;
    // Accepted receipt identity is a protocol, not the current encoder's oracle.
    let frozen_rating_request =
        br#"{"grade":"love","kind":"rated","post_id":1,"session":"gallery"}"#;
    let saved_request: String = store.connection.query_row(
        "SELECT request_json FROM interaction_events WHERE event_id='vote-old'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(
        saved_request,
        serde_json::to_string(
            &json!({"sha256": hex::encode(Sha256::digest(frozen_rating_request))})
        )?
    );
    store.rate(
        "vote-new",
        1,
        Grade::Dislike,
        "gallery",
        Some(prediction(1, "later-model")),
    )?;
    let original_preferences = UserSettings::default();
    store.save_preferences("preferences-old", &original_preferences)?;
    let newer_preferences = UserSettings {
        blacklist: vec!["fox".into()],
        ..Default::default()
    };
    store.save_preferences("preferences-new", &newer_preferences)?;
    let newer = store.activate_context("activate-new", "discover", "", true)?;
    assert_ne!(newer.context_id, context.context_id);
    let before = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    let count = event_count(&store)?;
    drop(store);
    let mut store = Store::open(&path)?;
    let retried = store.rate(
        "vote-old",
        1,
        Grade::Love,
        "gallery",
        Some(prediction(255, "")),
    )?;
    assert!(!retried.changed);
    assert_eq!(retried.revision, first.revision);
    let mut current_grade = cards[0].clone();
    current_grade.current_grade = Some(Grade::Dislike);
    store.presented("show-first", &context.context_id, None, &current_grade)?;
    let retried_context = store.activate_context("activate-old", "discover", "", false)?;
    assert_eq!(retried_context.context_id, context.context_id);
    store.save_preferences("preferences-old", &original_preferences)?;
    assert_eq!(before, durable_rows(&store.connection)?);
    assert_eq!(count, event_count(&store)?);
    assert!(
        store
            .rate(
                "vote-old",
                1,
                Grade::Like,
                "gallery",
                Some(prediction(2, "model"))
            )
            .is_err()
    );
    assert!(store.viewed("vote-old", 1, "gallery").is_err());
    assert!(
        store
            .activate_context("activate-old", "discover", "", true)
            .is_err()
    );
    assert!(
        store
            .presented("show-first", &context.context_id, None, &cards[1])
            .is_err()
    );
    assert!(
        store
            .save_preferences("preferences-old", &newer_preferences)
            .is_err()
    );
    assert_eq!(before, durable_rows(&store.connection)?);
    store.verify_history()?;
    assert_eq!(before, durable_rows(&store.connection)?);
    Ok(())
}

#[test]
fn failed_envelopes_leave_journal_facts_and_readouts_unchanged() -> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("rollback.sqlite3"))?;
    let offered = card("gallery", "discover", 1, 0);
    store.create_gallery(
        "gallery",
        "discover",
        "",
        "model",
        1,
        &[offered.clone()],
        false,
    )?;
    let context = store.activate_context("activation", "discover", "", false)?;
    store.save_preferences("settings", &UserSettings::default())?;
    let before = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    let count = event_count(&store)?;
    // Failure after reducer SQL still rolls back all derived writes.
    store.connection.execute_batch(
        "CREATE TRIGGER reject_event BEFORE INSERT ON interaction_events
         WHEN NEW.event_id='fail' BEGIN SELECT RAISE(ABORT,'journal insert failure'); END;",
    )?;
    assert!(
        store
            .rate(
                "fail",
                1,
                Grade::Love,
                "gallery",
                Some(prediction(2, "judge"))
            )
            .is_err()
    );
    assert!(
        store
            .presented("fail", &context.context_id, None, &offered)
            .is_err()
    );
    assert!(
        store
            .activate_context("fail", "balanced", "", false)
            .is_err()
    );
    let settings = UserSettings {
        blacklist: vec!["wolf".into()],
        ..Default::default()
    };
    assert!(store.save_preferences("fail", &settings).is_err());
    assert_eq!(before, durable_rows(&store.connection)?);
    assert_eq!(count, event_count(&store)?);
    assert!(!store.contains_event("fail")?);
    store
        .connection
        .execute_batch("DROP TRIGGER reject_event")?;
    let mut forged = offered.clone();
    forged.model_id = "forged-model".into();
    assert!(
        store
            .presented("forged", &context.context_id, None, &forged)
            .is_err()
    );
    assert!(
        store
            .presented(
                "unknown-entry",
                &context.context_id,
                Some("missing"),
                &offered
            )
            .is_err()
    );
    assert_eq!(before, durable_rows(&store.connection)?);
    assert_eq!(count, event_count(&store)?);
    store.rate(
        "valid-vote",
        1,
        Grade::Love,
        "gallery",
        Some(prediction(2, "judge")),
    )?;
    let replayed = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    store.verify_history()?;
    assert_eq!(replayed, durable_rows(&store.connection)?);
    Ok(())
}

#[test]
fn rejected_detail_insert_rolls_back_all_fact_families_and_envelope() -> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("detail-atomic.sqlite3"))?;
    let offered = card("gallery", "discover", 1, 0);
    store.create_gallery(
        "gallery",
        "discover",
        "",
        "model",
        1,
        &[offered.clone()],
        false,
    )?;
    let context = store.activate_context("activate", "discover", "", false)?;
    let before = durable_rows(&store.connection)?;
    store.connection.execute_batch(
        "CREATE TRIGGER reject_presentation BEFORE INSERT ON presentation_facts
        BEGIN SELECT RAISE(ABORT,'fixture presentation failure'); END;",
    )?;
    assert!(
        store
            .presented("failed", &context.context_id, None, &offered)
            .is_err()
    );
    assert_eq!(before, durable_rows(&store.connection)?);
    assert!(store.active_context()?.unwrap().entries.is_empty());
    assert!(!store.contains_event("failed")?);
    store
        .connection
        .execute_batch("DROP TRIGGER reject_presentation")?;
    store.presented("failed", &context.context_id, None, &offered)?;
    assert_eq!(store.active_context()?.unwrap().entries.len(), 1);
    store.verify_history()?;
    Ok(())
}

#[test]
fn predicted_and_unpredicted_rating_retries_preserve_exact_saved_envelopes() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("wire-history.sqlite3");
    let mut store = populated_current_profile(&path)?;
    let frozen_unpredicted_request =
        br#"{"grade":"like","kind":"rated","post_id":1,"session":"review"}"#;
    let saved_request: String = store.connection.query_row(
        "SELECT request_json FROM interaction_events WHERE event_id='new-vote'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(
        saved_request,
        serde_json::to_string(
            &json!({"sha256": hex::encode(Sha256::digest(frozen_unpredicted_request))})
        )?
    );
    let journal = raw_rows(
        &store.connection,
        "SELECT * FROM interaction_events ORDER BY sequence",
    )?;
    let saved_payload: String = store.connection.query_row(
        "SELECT payload_json FROM interaction_events WHERE event_id='old-vote'",
        [],
        |row| row.get(0),
    )?;
    assert!(matches!(
        decode_event(&saved_payload)?,
        Event::RatingRecorded {
            prediction: Some(_),
            ..
        }
    ));
    assert_eq!(store.ratings()?[0].revision, 2);
    assert!(store.checkpoints()?.is_empty());
    let retry = store.rate(
        "old-vote",
        1,
        Grade::Love,
        "review",
        Some(prediction(255, "")),
    )?;
    assert!(!retry.changed);
    assert_eq!(retry.revision, 1);
    let unpredicted_retry = store.rate("new-vote", 1, Grade::Like, "review", None)?;
    assert!(!unpredicted_retry.changed);
    assert_eq!(unpredicted_retry.revision, 2);
    assert!(
        store
            .rate("old-vote", 1, Grade::Meh, "review", None)
            .is_err()
    );
    let facts = durable_rows(&store.connection)?;
    store.verify_history()?;
    assert_eq!(facts, durable_rows(&store.connection)?);
    assert_eq!(
        journal,
        raw_rows(
            &store.connection,
            "SELECT * FROM interaction_events ORDER BY sequence"
        )?
    );
    drop(store);
    let reopened = Store::open(&path)?;
    reopened.verify_history()?;
    assert_eq!(facts, durable_rows(&reopened.connection)?);
    Ok(())
}

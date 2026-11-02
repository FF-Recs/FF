use super::*;

#[test]
fn optional_forecasts_track_only_changed_predicted_ratings_and_retry_without_forecasts()
-> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("optional-forecasts.sqlite3");
    let mut store = Store::open(&path)?;
    store.create_gallery(
        "starter",
        "discover",
        "",
        "starter",
        1,
        &[card("starter", "discover", 1, 0)],
        false,
    )?;
    let first = store.rate("unpredicted", 1, Grade::Like, "starter", None)?;
    assert!(first.changed);
    assert_eq!(first.revision, 1);
    assert!(store.checkpoints()?.is_empty());
    assert!(
        !store
            .rate(
                "same-with-forecast",
                1,
                Grade::Like,
                "starter",
                Some(prediction(3, "ensemble-a"))
            )?
            .changed
    );
    assert!(store.checkpoints()?.is_empty());
    store.rate(
        "predicted-zero",
        1,
        Grade::Love,
        "starter",
        Some(prediction(0, "ensemble-a")),
    )?;
    assert_eq!(store.checkpoints()?[&1].predicted_grade, 0);
    let checkpoint_time = store.checkpoints()?[&1].updated_at;
    assert!(
        !store
            .rate("same-without-forecast", 1, Grade::Love, "starter", None)?
            .changed
    );
    assert_eq!(store.checkpoints()?[&1].updated_at, checkpoint_time);
    store.rate(
        "predicted-three",
        1,
        Grade::Meh,
        "starter",
        Some(prediction(3, "ensemble-b")),
    )?;
    assert_eq!(store.checkpoints()?[&1].predicted_grade, 3);
    let facts = raw_rows(
        &store.connection,
        "SELECT * FROM interaction_events ORDER BY sequence",
    )?;
    let before = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    for invalid in [prediction(4, "ensemble"), prediction(3, " \t")] {
        assert!(
            store
                .rate("invalid", 1, Grade::Dislike, "starter", Some(invalid))
                .is_err()
        );
        assert_eq!(before, durable_rows(&store.connection)?);
        assert_eq!(
            facts,
            raw_rows(
                &store.connection,
                "SELECT * FROM interaction_events ORDER BY sequence"
            )?
        );
    }
    let cleared = store.rate("clear-forecast", 1, Grade::Dislike, "starter", None)?;
    assert!(cleared.changed);
    assert_eq!(cleared.revision, 4);
    assert!(store.checkpoints()?.is_empty());
    let retry = store.rate("predicted-three", 1, Grade::Meh, "starter", None)?;
    assert!(!retry.changed);
    assert_eq!(retry.revision, 3);
    assert!(store.checkpoints()?.is_empty());
    let retry = store.rate(
        "clear-forecast",
        1,
        Grade::Dislike,
        "starter",
        Some(prediction(255, "")),
    )?;
    assert!(!retry.changed);
    assert!(store.checkpoints()?.is_empty());
    let expected = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    let journal = raw_rows(
        &store.connection,
        "SELECT * FROM interaction_events ORDER BY sequence",
    )?;
    let rating_payload: String = store.connection.query_row(
        "SELECT payload_json FROM interaction_events WHERE event_id='unpredicted'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(
        serde_json::from_str::<Value>(&rating_payload)?["prediction"],
        Value::Null
    );
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM interaction_events WHERE json_extract(payload_json,'$.kind')='rating_recorded'"
        )?,
        6
    );
    store.verify_history()?;
    assert_eq!(expected, durable_rows(&store.connection)?);
    assert_eq!(
        journal,
        raw_rows(
            &store.connection,
            "SELECT * FROM interaction_events ORDER BY sequence"
        )?
    );
    drop(store);
    let reopened = Store::open(&path)?;
    assert!(reopened.checkpoints()?.is_empty());
    assert_eq!(expected, durable_rows(&reopened.connection)?);
    Ok(())
}

#[test]
fn retained_review_without_forecast_preserves_vote_and_real_offer_exclusion_provenance()
-> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("starter-retained-review.sqlite3"))?;
    store.create_gallery(
        "review",
        "rating_review",
        "",
        "header-model",
        1,
        &[card("review", "rating_review", 1, 0)],
        false,
    )?;
    store.rate(
        "predicted-review",
        1,
        Grade::Like,
        "review",
        Some(prediction(2, "judgment-model")),
    )?;
    assert_eq!(store.review_sequence()?, 1);
    store.rate("starter-review", 1, Grade::Dislike, "review", None)?;
    assert!(store.checkpoints()?.is_empty());
    assert_eq!(store.review_sequence()?, 1);
    assert!(store.review_exclusions()?.contains(&1));
    let provenance: String = store.connection.query_row(
        "SELECT model_id FROM review_facts WHERE post_id=1 AND kind IN ('exclusion','exclusion_seed') ORDER BY sequence DESC,ordinal DESC LIMIT 1",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(provenance, "offered-model");
    store.rate("starter-correction", 1, Grade::Love, "review", None)?;
    assert!(store.review_exclusions()?.contains(&1));
    assert!(store.checkpoints()?.is_empty());
    let before = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    store.verify_history()?;
    assert_eq!(before, durable_rows(&store.connection)?);
    Ok(())
}

#[test]
fn atomic_review_revisions_retry_cooldown_and_exclusion_provenance() -> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("reviews.sqlite3"))?;
    store.create_gallery(
        "initial",
        "discover",
        "",
        "initial-model",
        1,
        &[card("initial", "discover", 1, 0)],
        false,
    )?;
    let first = store.rate(
        "initial-vote",
        1,
        Grade::Love,
        "initial",
        Some(prediction(1, "model-a")),
    )?;
    let first_time = store.ratings()?[0].first_rated_at;
    assert_eq!(first.revision, 1);
    assert_eq!(store.review_sequence()?, 0);
    store.create_gallery(
        "review",
        "rating_review",
        "",
        "review-model",
        2,
        &[
            card("review", "rating_review", 1, 0),
            card("review", "rating_review", 2, 1),
        ],
        false,
    )?;
    let reviewed = store.rate(
        "review-vote",
        1,
        Grade::Love,
        "review",
        Some(prediction(2, "model-b")),
    )?;
    assert!(reviewed.changed); // Same grade, different offered context.
    assert_eq!(reviewed.revision, 2);
    assert_eq!(store.review_sequence()?, 1);
    let checkpoint = store.checkpoints()?.remove(&1).unwrap();
    store.rate(
        "second-vote",
        2,
        Grade::Like,
        "review",
        Some(prediction(1, "model-b")),
    )?;
    assert_eq!(store.review_sequence()?, 2);
    let retried = store.rate(
        "review-vote",
        1,
        Grade::Love,
        "review",
        Some(prediction(0, "newer-live-model")),
    )?;
    assert!(!retried.changed);
    assert_eq!(retried.revision, 2);
    let after_retry = store.checkpoints()?.remove(&1).unwrap();
    assert_eq!(after_retry.review_sequence, checkpoint.review_sequence);
    assert_eq!(after_retry.updated_at, checkpoint.updated_at);
    assert_eq!(after_retry.model_id, "model-b");
    assert_eq!(after_retry.predicted_grade, 2);
    let correction = store.rate(
        "correction",
        1,
        Grade::Dislike,
        "review",
        Some(prediction(0, "model-c")),
    )?;
    assert_eq!(correction.revision, 3);
    assert_eq!(store.review_sequence()?, 2);
    assert!(store.review_exclusions()?.contains(&1));
    let checkpoint = store.checkpoints()?.remove(&1).unwrap();
    assert_eq!(checkpoint.recorded_grade, Grade::Dislike);
    assert_eq!(checkpoint.rating_revision, 3);
    assert_eq!(checkpoint.review_sequence, 2);
    assert_eq!(checkpoint.model_id, "model-c");
    assert_eq!(store.ratings()?[0].first_rated_at, first_time);
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM review_facts WHERE kind IN ('vote','vote_seed')"
        )?,
        2
    );
    assert!(store.recent_views(500)?.is_empty());
    // A later correction must not undo a permanent exclusion.
    store.rate(
        "later-correction",
        1,
        Grade::Like,
        "review",
        Some(prediction(2, "model-d")),
    )?;
    assert!(store.review_exclusions()?.contains(&1));
    assert_eq!(store.review_sequence()?, 2);
    Ok(())
}

#[test]
fn failed_judgment_rolls_back_rating_gallery_and_review_state() -> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("atomic.sqlite3"))?;
    store.create_gallery(
        "review",
        "rating_review",
        "",
        "model",
        1,
        &[card("review", "rating_review", 1, 0)],
        false,
    )?;
    let before = durable_rows(&store.connection)?;
    store.connection.execute_batch(
        "CREATE TRIGGER reject_checkpoint BEFORE INSERT ON review_facts
         BEGIN SELECT RAISE(ABORT,'fixture checkpoint failure'); END;",
    )?;
    let error = store
        .rate(
            "failed-vote",
            1,
            Grade::Dislike,
            "review",
            Some(prediction(1, "model")),
        )
        .unwrap_err();
    assert!(format!("{error:#}").contains("fixture checkpoint failure"));
    assert!(store.ratings()?.is_empty());
    assert_eq!(store.review_sequence()?, 0);
    assert!(store.review_exclusions()?.is_empty());
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM review_facts WHERE kind IN ('vote','vote_seed')"
        )?,
        0
    );
    assert_eq!(
        scalar_i64(&store.connection, "SELECT COUNT(*) FROM rating_facts")?,
        0
    );
    assert_eq!(before, durable_rows(&store.connection)?);
    assert!(!store.contains_event("failed-vote")?);
    store
        .connection
        .execute_batch("DROP TRIGGER reject_checkpoint")?;
    let retried = store.rate(
        "failed-vote",
        1,
        Grade::Dislike,
        "review",
        Some(prediction(1, "model")),
    )?;
    assert!(retried.changed);
    assert_eq!(retried.revision, 1);
    assert_eq!(store.review_sequence()?, 1);
    assert!(store.review_exclusions()?.contains(&1));
    store.verify_history()?;
    Ok(())
}

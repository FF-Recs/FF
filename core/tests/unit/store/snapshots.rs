use super::*;

#[test]
fn snapshot_reads_typed_ordered_wal_ratings_and_releases_reader() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("snapshot.sqlite3");
    let mut store = Store::open(&path)?;
    store
        .connection
        .pragma_update(None, "wal_autocheckpoint", 0)?;
    let cards: Vec<_> = (1..=4)
        .map(|post| card("gallery", "discover", post, post as usize - 1))
        .collect();
    store.create_gallery("gallery", "discover", "", "model", 4, &cards, false)?;
    let expected = [
        (2, Grade::Dislike, 1.0),
        (3, Grade::Meh, 1.0),
        (1, Grade::Like, 2.0),
        (4, Grade::Love, 3.0),
    ];
    for (post_id, grade, timestamp) in expected.into_iter().rev() {
        append_at(
            &mut store,
            &format!("vote-{post_id}"),
            timestamp,
            Event::RatingRecorded {
                post_id,
                grade,
                session: "gallery".into(),
                prediction: None,
            },
        )?;
    }
    let snapshot = Store::snapshot_ratings_at(&path)?;
    assert_eq!(snapshot.len(), expected.len());
    for (rating, (post, grade, timestamp)) in snapshot.iter().zip(expected) {
        assert_eq!(rating.post_id, post);
        assert_eq!(rating.grade, grade);
        assert_eq!(rating.first_rated_at, timestamp);
        assert_eq!(rating.updated_at, timestamp);
        assert_eq!(rating.revision, 1);
        assert_eq!(rating.context_json, store.context("gallery", post)?.json);
    }
    append_at(
        &mut store,
        "correction",
        4.0,
        Event::RatingRecorded {
            post_id: 2,
            grade: Grade::Love,
            session: "gallery".into(),
            prediction: None,
        },
    )?;
    assert_eq!(snapshot[0].grade, Grade::Dislike);
    assert_eq!(snapshot[0].revision, 1);
    assert_eq!(Store::snapshot_ratings_at(&path)?[0].revision, 2);
    store.connection.busy_timeout(Duration::ZERO)?;
    assert_eq!(
        scalar_i64(&store.connection, "PRAGMA wal_checkpoint(TRUNCATE)")?,
        0
    );
    store.verify_history()?;
    Ok(())
}

#[test]
fn ratings_decoder_keeps_one_read_view_across_append_commits() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("snapshot-writer.sqlite3");
    let mut store = Store::open(&path)?;
    store.create_gallery(
        "gallery",
        "discover",
        "",
        "model",
        2,
        &[
            card("gallery", "discover", 1, 0),
            card("gallery", "discover", 2, 1),
        ],
        false,
    )?;
    append_at(
        &mut store,
        "before",
        2.0,
        Event::RatingRecorded {
            post_id: 1,
            grade: Grade::Love,
            session: "gallery".into(),
            prediction: None,
        },
    )?;
    let mut source = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let transaction = source.transaction_with_behavior(TransactionBehavior::Deferred)?;
    let before = facts::ratings(&transaction)?;
    append_at(
        &mut store,
        "after",
        3.0,
        Event::RatingRecorded {
            post_id: 1,
            grade: Grade::Meh,
            session: "gallery".into(),
            prediction: None,
        },
    )?;
    append_at(
        &mut store,
        "second",
        3.0,
        Event::RatingRecorded {
            post_id: 2,
            grade: Grade::Like,
            session: "gallery".into(),
            prediction: None,
        },
    )?;
    let during = facts::ratings(&transaction)?;
    assert_eq!(during.len(), 1);
    assert_eq!(during[0].post_id, 1);
    assert_eq!(during[0].grade, Grade::Love);
    assert_eq!(during[0].revision, 1);
    assert_eq!(during[0].first_rated_at, 2.0);
    assert_eq!(during[0].updated_at, 2.0);
    assert_eq!(during[0].context_json, before[0].context_json);
    transaction.commit()?;
    drop(source);
    let after = Store::snapshot_ratings_at(&path)?;
    assert_eq!(after.len(), 2);
    assert_eq!(after[0].grade, Grade::Meh);
    assert_eq!(after[0].updated_at, 3.0);
    assert_eq!(after[0].revision, 2);
    assert_eq!(after[1].post_id, 2);
    assert_eq!(after[1].grade, Grade::Like);
    Ok(())
}

#[test]
fn snapshot_decode_failure_releases_reader_for_writer_and_checkpoint() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("snapshot-invalid.sqlite3");
    let mut store = Store::open(&path)?;
    store
        .connection
        .pragma_update(None, "wal_autocheckpoint", 0)?;
    store.create_gallery(
        "gallery",
        "discover",
        "",
        "model",
        1,
        &[card("gallery", "discover", 1, 0)],
        false,
    )?;
    store.rate("vote", 1, Grade::Love, "gallery", None)?;
    let view: String = store.connection.query_row(
        "SELECT sql FROM sqlite_schema WHERE name='q_ratings'",
        [],
        |row| row.get(0),
    )?;
    // Fault the read boundary, not immutable history. Restore its exact definition below.
    store.connection.execute_batch("DROP VIEW q_ratings;
        CREATE VIEW q_ratings AS SELECT 1 post_id,'invalid' action,1.0 first_rated_at,2.0 updated_at,1 revision,'{}' context_json;")?;
    assert!(Store::snapshot_ratings_at(&path).is_err());
    store.connection.busy_timeout(Duration::ZERO)?;
    store.connection.execute_batch("DROP VIEW q_ratings")?;
    store.connection.execute_batch(&view)?;
    store.rate("correction", 1, Grade::Like, "gallery", None)?;
    assert_eq!(
        scalar_i64(&store.connection, "PRAGMA wal_checkpoint(TRUNCATE)")?,
        0
    );
    let recovered = Store::snapshot_ratings_at(&path)?;
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].grade, Grade::Like);
    assert_eq!(recovered[0].revision, 2);
    store.verify_history()?;
    Ok(())
}

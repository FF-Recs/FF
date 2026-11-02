use super::*;

fn corrupt_fact(connection: &Connection, table: &str, sql: &str) -> Result<()> {
    let name = format!("{table}_no_update");
    let trigger: String = connection.query_row(
        "SELECT sql FROM sqlite_schema WHERE type='trigger' AND name=?1",
        [&name],
        |row| row.get(0),
    )?;
    connection.execute_batch(&format!("DROP TRIGGER {name}; {sql}; {trigger}"))?;
    Ok(())
}

#[test]
fn unsupported_envelope_is_rejected_without_rewriting_existing_history() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("unsupported-event.sqlite3");
    let mut store = Store::open(&path)?;
    store.save_preferences("preferences", &UserSettings::default())?;
    store.connection.execute_batch(
        "INSERT INTO interaction_events(sequence,event_id,occurred_at,request_json,payload_json,result_json)
         VALUES(999,'future',1,'{}','{\"kind\":\"future\"}','null');",
    )?;
    let before = durable_rows(&store.connection)?;
    assert!(store.verify_history().is_err());
    assert_eq!(before, durable_rows(&store.connection)?);
    drop(store);
    assert!(Store::open(&path).is_err());
    Ok(())
}

#[test]
fn current_history_verification_is_read_only_with_temp_shadows_and_caller_transaction() -> Result<()>
{
    let fixture = fixture()?;
    let path = fixture.path().join("read-only-history.sqlite3");
    let store = populated_current_profile(&path)?;
    store.connection.execute_batch(
        "CREATE TABLE private_evidence(note TEXT,payload BLOB);
         INSERT INTO private_evidence VALUES('retained',X'00FEFF');
         CREATE INDEX private_note ON private_evidence(note);
         CREATE VIEW private_view AS SELECT * FROM private_evidence;
         CREATE TRIGGER private_event_insert AFTER INSERT ON interaction_events
         BEGIN INSERT INTO private_evidence VALUES(NEW.event_id,X'00'); END;",
    )?;
    for table in FACT_TABLES {
        store.connection.execute_batch(&format!(
            "CREATE TEMP TABLE {table}(shadow BLOB);
             INSERT INTO temp.{table} VALUES(X'00FF');"
        ))?;
    }
    let main_schema_sql =
        "SELECT type,name,tbl_name,sql FROM main.sqlite_schema ORDER BY type,name";
    let temp_schema_sql =
        "SELECT type,name,tbl_name,sql FROM temp.sqlite_schema ORDER BY type,name";
    let main_schema = raw_rows(&store.connection, main_schema_sql)?;
    let temp_schema = raw_rows(&store.connection, temp_schema_sql)?;
    let expected = durable_rows(&store.connection)?;
    let custom = raw_rows(&store.connection, "SELECT * FROM main.private_evidence")?;
    let changes = store.connection.total_changes();
    store.connection.execute_batch("PRAGMA query_only=ON")?;
    for caller_transaction in [false, true] {
        if caller_transaction {
            store.connection.execute_batch("BEGIN")?;
        }
        history::verify_history(&store.connection)?;
        assert_eq!(store.connection.is_autocommit(), !caller_transaction);
        assert_eq!(changes, store.connection.total_changes());
        assert_eq!(expected, durable_rows(&store.connection)?);
        assert_eq!(main_schema, raw_rows(&store.connection, main_schema_sql)?);
        assert_eq!(temp_schema, raw_rows(&store.connection, temp_schema_sql)?);
        assert_eq!(
            custom,
            raw_rows(&store.connection, "SELECT * FROM main.private_evidence")?
        );
        for table in FACT_TABLES {
            assert_eq!(
                raw_rows(&store.connection, &format!("SELECT * FROM temp.{table}"))?,
                vec![vec![SqliteValue::Blob(vec![0, 255])]],
            );
        }
        if caller_transaction {
            store.connection.execute_batch("ROLLBACK")?;
        }
    }
    drop(store);
    let source = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    history::verify_history(&source)?;
    assert!(source.is_autocommit());
    assert_eq!(expected, durable_rows(&source)?);
    drop(source);
    let reopened = Store::open(&path)?;
    assert_eq!(expected, durable_rows(&reopened.connection)?);
    assert_eq!(
        custom,
        raw_rows(&reopened.connection, "SELECT * FROM private_evidence")?
    );
    Ok(())
}

#[test]
fn verification_detects_corruption_in_every_fact_family_and_saved_results_without_writes()
-> Result<()> {
    let fixture = fixture()?;
    for (index, (table, corruption)) in [
        ("interaction_events", "UPDATE interaction_events SET result_json='{\"revision\":999,\"changed\":true}' WHERE event_id='old-vote'"),
        ("selection_facts", "UPDATE selection_facts SET model_id='corrupted'"),
        ("offer_facts", "UPDATE offer_facts SET payload_json='corrupted'"),
        ("rating_facts", "UPDATE rating_facts SET context_json='corrupted'"),
        ("review_facts", "UPDATE review_facts SET review_sequence=999"),
        ("presentation_facts", "UPDATE presentation_facts SET viewed_at=999"),
        ("activation_facts", "UPDATE activation_facts SET filter_text='corrupted'"),
        ("preference_facts", "UPDATE preference_facts SET settings_json='{}'"),
        ("legacy_delivery_facts", "UPDATE legacy_delivery_facts SET occurred_at=999"),
    ].iter().enumerate() {
        let path = fixture.path().join(format!("corrupt-fact-{index}.sqlite3"));
        let store = populated_current_profile(&path)?;
        corrupt_fact(&store.connection, table, corruption)?;
        let expected = durable_rows(&store.connection)?;
        let changes = store.connection.total_changes();
        store.connection.execute_batch("PRAGMA query_only=ON; BEGIN")?;
        assert!(store.verify_history().is_err(), "{table}");
        assert!(!store.connection.is_autocommit());
        assert_eq!(changes, store.connection.total_changes());
        assert_eq!(expected, durable_rows(&store.connection)?);
        store.connection.execute_batch("ROLLBACK")?;
        drop(store);
        // Opening admits a valid current layout without reducer replay.
        let reopened = Store::open(&path)?;
        assert!(reopened.verify_history().is_err(), "{table}");
        assert_eq!(expected, durable_rows(&reopened.connection)?);
    }
    Ok(())
}

#[test]
fn invalid_current_layout_foreign_keys_and_headers_are_rejected_untouched() -> Result<()> {
    let fixture = fixture()?;
    for (index, corruption) in [
        "DROP VIEW q_gallery_sessions",
        "DROP VIEW q_gallery_sessions; CREATE VIEW q_gallery_sessions AS SELECT 1 AS session_id",
        "DROP TRIGGER offer_facts_no_delete",
        "DROP INDEX selection_identity",
        "PRAGMA foreign_keys=OFF; CREATE TABLE invalid_reference(event_id TEXT REFERENCES interaction_events(event_id)); INSERT INTO invalid_reference VALUES('missing')",
        "PRAGMA foreign_keys=OFF; INSERT INTO preference_facts VALUES(999,0,'{}')",
        "INSERT INTO interaction_events VALUES(999,'future',1,'{}','{\"kind\":\"future\"}','null')",
        "INSERT INTO interaction_events VALUES(999,'invalid-payload',1,'{}','not-json','null')",
        "INSERT INTO interaction_events VALUES(999,'second-initialization',1,'{}','{\"kind\":\"initialized\",\"facts\":{\"tables\":[]}}','null')",
        "INSERT INTO interaction_events VALUES(999,'missing-facts',1,'{}','{\"kind\":\"initialized\"}','null')",
        "INSERT INTO interaction_events VALUES(999,'wrong-request',1,'{}','{\"kind\":\"preferences_saved\",\"settings\":{}}','null')",
    ].iter().enumerate() {
        let path = fixture.path().join(format!("bad-current-{index}.sqlite3"));
        let store = Store::open(&path)?;
        store.connection.execute_batch(corruption)?;
        let schema_sql = "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name";
        let schema = raw_rows(&store.connection, schema_sql)?;
        let rows = durable_rows(&store.connection)?;
        drop(store);
        let bytes = fs::read(&path)?;
        let file_count = fs::read_dir(fixture.path())?.count();
        assert!(Store::open(&path).is_err(), "{corruption}");
        assert_eq!(bytes, fs::read(&path)?, "{corruption}");
        assert_eq!(file_count, fs::read_dir(fixture.path())?.count());
        let original = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        assert_eq!(scalar_i64(&original, "PRAGMA user_version")?, 13);
        assert_eq!(schema, raw_rows(&original, schema_sql)?);
        assert_eq!(rows, durable_rows(&original)?);
    }
    Ok(())
}

#[test]
fn current_open_reads_committed_wal_and_preserves_custom_objects() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("current-wal.sqlite3");
    let mut source = Store::open(&path)?;
    source.connection.execute_batch(
        "CREATE TABLE private_wal(note TEXT,payload BLOB);
         PRAGMA wal_autocheckpoint=0;
         PRAGMA wal_checkpoint(TRUNCATE);",
    )?;
    source.create_gallery(
        "wal",
        "discover",
        "",
        "model",
        1,
        &[card("wal", "discover", 77, 0)],
        false,
    )?;
    source.rate("wal-vote", 77, Grade::Like, "wal", None)?;
    source
        .connection
        .execute_batch("INSERT INTO private_wal VALUES('committed in WAL',X'00FEFF')")?;
    let main_only = Connection::open_with_flags(
        format!("file:{}?immutable=1", path.display()),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    assert_eq!(
        scalar_i64(&main_only, "SELECT COUNT(*) FROM rating_facts")?,
        0
    );
    assert_eq!(
        scalar_i64(&main_only, "SELECT COUNT(*) FROM private_wal")?,
        0
    );
    drop(main_only);
    let expected = durable_rows(&source.connection)?;
    let custom = raw_rows(&source.connection, "SELECT * FROM private_wal")?;
    let store = Store::open(&path)?;
    assert_eq!(expected, durable_rows(&store.connection)?);
    assert_eq!(
        custom,
        raw_rows(&store.connection, "SELECT * FROM private_wal")?
    );
    assert_eq!(store.ratings()?[0].post_id, 77);
    store.verify_history()?;
    Ok(())
}

#[test]
fn current_layout_requires_a_single_first_initial_event() -> Result<()> {
    let fixture = fixture()?;
    for (index, envelope) in [
        None,
        Some((
            2,
            Event::Initialized {
                facts: seed::InitialFacts::default(),
            },
        )),
        Some((
            1,
            Event::PreferencesSaved {
                settings: UserSettings::default(),
            },
        )),
    ]
    .into_iter()
    .enumerate()
    {
        let path = fixture
            .path()
            .join(format!("missing-initial-{index}.sqlite3"));
        let connection = Connection::open(&path)?;
        connection.execute_batch(SCHEMA)?;
        connection.pragma_update(None, "user_version", 13)?;
        if let Some((sequence, event)) = envelope {
            let payload = serde_json::to_string(&event)?;
            let request = serde_json::to_string(&event.request(&payload)?)?;
            connection.execute(
                "INSERT INTO interaction_events VALUES(?1,'not-initial',1,?2,?3,'null')",
                params![sequence, request, payload],
            )?;
        }
        drop(connection);
        let bytes = fs::read(&path)?;
        assert!(Store::open(&path).is_err());
        assert_eq!(bytes, fs::read(&path)?);
        assert_eq!(fs::read_dir(fixture.path())?.count(), index + 1);
    }
    Ok(())
}

#[test]
fn current_journal_rejects_old_initialization_and_rating_payloads_without_rewriting() -> Result<()>
{
    let fixture = fixture()?;
    for (index, payload) in [
        r#"{"kind":"initialized"}"#,
        r#"{"kind":"imported_baseline","tables":[]}"#,
        r#"{"kind":"rated","post_id":1,"grade":"love","session":"old","predicted_grade":2,"model_id":"old-model"}"#,
    ].into_iter().enumerate() {
        assert!(decode_event(payload).is_err(), "{payload}");
        let path = fixture.path().join(format!("old-payload-{index}.sqlite3"));
        let store = Store::open(&path)?;
        store.connection.execute(
            "INSERT INTO interaction_events VALUES(2,'old-payload',1,'{}',?1,'null')",
            [payload],
        )?;
        let expected = durable_rows(&store.connection)?;
        assert!(store.verify_history().is_err(), "{payload}");
        assert_eq!(expected, durable_rows(&store.connection)?);
        drop(store);
        let bytes = fs::read(&path)?;
        assert!(Store::open(&path).is_err(), "{payload}");
        assert_eq!(bytes, fs::read(&path)?);
    }
    Ok(())
}

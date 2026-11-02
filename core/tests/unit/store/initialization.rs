use super::*;

fn assert_runtime_views(connection: &Connection) -> Result<()> {
    let expected = [
        "q_auto_sessions",
        "q_browse_contexts",
        "q_browse_entries",
        "q_browse_state",
        "q_gallery_sessions",
        "q_rating_prediction_checkpoints",
        "q_rating_review_state",
        "q_ratings",
        "q_user_preferences",
    ];
    let names = raw_rows(
        connection,
        "SELECT name FROM sqlite_schema WHERE type='view' AND name GLOB 'q_*' ORDER BY name",
    )?;
    assert_eq!(
        names,
        expected
            .iter()
            .map(|name| vec![SqliteValue::Text((*name).into())])
            .collect::<Vec<_>>()
    );
    for (view, columns) in [
        (
            "q_gallery_sessions",
            &[
                "session_id",
                "status",
                "facet",
                "filter_text",
                "model_id",
                "requested_size",
                "returned_size",
            ][..],
        ),
        (
            "q_auto_sessions",
            &[
                "session_id",
                "status",
                "model_id",
                "preset",
                "filter_text",
                "seed",
                "returned_size",
            ][..],
        ),
    ] {
        let actual = raw_rows(
            connection,
            &format!("SELECT name FROM pragma_table_info('{view}') ORDER BY cid"),
        )?;
        assert_eq!(
            actual,
            columns
                .iter()
                .map(|name| vec![SqliteValue::Text((*name).into())])
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(scalar_i64(connection, "PRAGMA user_version")?, 13);
    assert_eq!(
        raw_rows(
            connection,
            "SELECT name FROM pragma_table_info('interaction_events') ORDER BY cid",
        )?,
        [
            "sequence",
            "event_id",
            "occurred_at",
            "request_json",
            "payload_json",
            "result_json"
        ]
        .into_iter()
        .map(|name| vec![SqliteValue::Text(name.into())])
        .collect::<Vec<_>>(),
    );
    Ok(())
}

#[test]
fn rejected_schemas_leave_file_bytes_version_schema_and_data_unchanged() -> Result<()> {
    let fixture = fixture()?;
    for (index, version) in [0, 1, 6, 7, 8, 9, 10, 11, 12, 999].into_iter().enumerate() {
        let path = fixture.path().join(format!("rejected-{version}.sqlite3"));
        let connection = Connection::open(&path)?;
        connection.execute_batch(
            "CREATE TABLE retained(note TEXT, payload BLOB);
             INSERT INTO retained VALUES('untouched',X'00FEFF');",
        )?;
        connection.pragma_update(None, "user_version", version)?;
        let schema_sql = "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name";
        let schema = raw_rows(&connection, schema_sql)?;
        let rows = raw_rows(&connection, "SELECT * FROM retained")?;
        drop(connection);
        let before = fs::read(&path)?;
        assert!(Store::open(&path).is_err(), "version {version}");
        assert_eq!(before, fs::read(&path)?, "version {version}");
        let connection = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        assert_eq!(scalar_i64(&connection, "PRAGMA user_version")?, version);
        assert_eq!(schema, raw_rows(&connection, schema_sql)?);
        assert_eq!(rows, raw_rows(&connection, "SELECT * FROM retained")?);
        let journal: String =
            connection.pragma_query_value(None, "journal_mode", |row| row.get(0))?;
        assert_eq!(journal, "delete");
        assert_eq!(fs::read_dir(fixture.path())?.count(), index + 1);
    }
    Ok(())
}

#[test]
fn fresh_profile_has_no_fabricated_baseline_and_reopens_without_backup() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("fresh.sqlite3");
    let store = Store::open(&path)?;
    let (payload, result): (String, String) = store.connection.query_row(
        "SELECT payload_json,result_json FROM interaction_events WHERE sequence=1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    assert!(matches!(
        serde_json::from_str::<Event>(&payload)?,
        Event::Initialized { facts } if facts == seed::InitialFacts::default()
    ));
    assert_eq!(serde_json::from_str::<Value>(&result)?, Value::Null);
    assert!(store.ratings()?.is_empty());
    assert_eq!(store.review_sequence()?, 0);
    assert!(store.active_context()?.is_none());
    assert!(store.preferences()?.is_none());
    store.verify_history()?;
    assert_runtime_views(&store.connection)?;
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM sqlite_schema WHERE type='table'"
        )?,
        9
    );
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM sqlite_schema WHERE type='trigger'"
        )?,
        27
    );
    drop(store);
    Store::open(&path)?.verify_history()?;
    assert_eq!(fs::read_dir(fixture.path())?.count(), 1);
    Ok(())
}

#[test]
fn native_seed_preserves_known_state_and_continues_native_history() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("historical.sqlite3");
    use SqliteValue::Text as T;
    let mut store = fixtures::seeded(&path, |connection| {
        connection.execute_batch(
            "INSERT INTO rating_facts VALUES
             (1,0,'seed',1,'love',1.25,2.5,7,' {\"historical\": true} ',NULL,NULL,2,'love',7,'old-checkpoint',8,8.25),
             (1,1,'seed',2,'dislike',3.25,4.5,2,'not-current-transport',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
             INSERT INTO selection_facts VALUES
             (1,0,'gallery','old-gallery','seed',5.25,6.5,'completed','rating_review','+wolf','old-model',2,2,1,0,NULL),
             (1,1,'auto','old-auto','seed',10.25,11.5,'abandoned','balanced','','old-auto-model',NULL,1,0,NULL,17);
             INSERT INTO offer_facts VALUES
             (1,0,'gallery','old-gallery',0,1,'rating_review',' {\"old\":\"bytes\"} ',NULL,NULL,'rated',5.5),
             (1,1,'gallery','old-gallery',1,2,'rating_review','not-json',NULL,NULL,'released',NULL),
             (1,2,'auto','old-auto',0,4,'play','{\"opaque\":\"offer\"}',NULL,NULL,NULL,NULL);
             INSERT INTO legacy_delivery_facts VALUES
             (1,0,'gallery_delivery',3,'missing-old-owner',7.25),
             (1,1,'auto_play',4,'old-auto',12.25);
             INSERT INTO review_facts VALUES
             (1,0,'exclusion_seed',2,'old-gallery',NULL,'old-exclusion',9.5),
             (1,1,'counter_seed',NULL,NULL,8,NULL,NULL),
             (1,2,'vote_seed',1,'old-gallery',8,NULL,NULL);
             INSERT INTO presentation_facts VALUES
             (1,0,'view_seed','historical-view','old-auto',4,13.5,NULL,NULL,0,NULL,NULL,NULL,NULL,NULL,1);",
        )?;
        Ok(())
    })?;
    assert_eq!(store.ratings()?[0].context_json, " {\"historical\": true} ");
    assert_eq!(store.ratings()?[1].context_json, "not-current-transport");
    assert_eq!(store.review_exclusions()?, HashSet::from([2]));
    assert_eq!(event_count(&store)?, 1);
    let payload: String =
        store
            .connection
            .query_row("SELECT payload_json FROM interaction_events", [], |row| {
                row.get(0)
            })?;
    assert!(matches!(
        serde_json::from_str::<Event>(&payload)?,
        Event::Initialized { .. }
    ));
    assert_eq!(store.checkpoints()?[&1].updated_at, 8.25);
    assert_eq!(store.checkpoints()?[&1].review_sequence, 8);
    assert_eq!(store.review_sequence()?, 8);
    assert!(store.active_context()?.is_none());
    assert!(store.preferences()?.is_none());
    assert!(
        store
            .save_preferences("historical-view", &UserSettings::default())
            .is_err()
    );
    store.viewed("historical-view", 4, "old-auto")?;
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM presentation_facts WHERE is_view=1"
        )?,
        1
    );
    let first = card("review-new", "rating_review", 1, 0);
    store.create_gallery(
        "review-new",
        "rating_review",
        "+wolf",
        "new-model",
        2,
        &[first.clone(), card("review-new", "rating_review", 5, 1)],
        false,
    )?;
    let context = store.activate_context("activation", "rating_review", "+wolf", false)?;
    store.presented("presentation", &context.context_id, None, &first)?;
    assert_eq!(
        store
            .rate(
                "vote",
                1,
                Grade::Dislike,
                "review-new",
                Some(prediction(1, "new-judge"))
            )?
            .revision,
        8
    );
    assert_eq!(store.ratings()?[0].first_rated_at, 1.25);
    assert_eq!(store.review_sequence()?, 9);
    store.create_auto(
        "auto-new",
        "fresh",
        "",
        "auto-model",
        42,
        &[card("auto-new", "play", 6, 0)],
    )?;
    store.append_auto("auto-new", 1, &[card("auto-new", "play", 7, 1)])?;
    store.viewed("auto-view", 7, "auto-new")?;
    let settings = UserSettings {
        blacklist: vec!["wolf".into()],
        ..Default::default()
    };
    store.save_preferences("preferences", &settings)?;
    let final_state = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    store.verify_history()?;
    assert_eq!(final_state, durable_rows(&store.connection)?);
    assert_eq!(store.preferences()?, Some(settings));
    assert_eq!(scalar_i64(&store.connection, "PRAGMA synchronous")?, 2);
    let mode: String = store
        .connection
        .pragma_query_value(None, "journal_mode", |row| row.get(0))?;
    assert_eq!(mode, "wal");
    assert_eq!(
        raw_rows(
            &store.connection,
            "SELECT payload_json FROM offer_facts WHERE session_id='old-gallery' ORDER BY position"
        )?,
        vec![
            vec![T(" {\"old\":\"bytes\"} ".into())],
            vec![T("not-json".into())]
        ],
    );
    drop(store);
    let reopened = Store::open(&path)?;
    assert_eq!(final_state, durable_rows(&reopened.connection)?);
    reopened.verify_history()?;
    Ok(())
}

#[test]
fn bulk_boundary_fixtures_preserve_existing_history_and_native_results() -> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("boundary-setup.sqlite3"))?;
    store.save_preferences(
        "preferences",
        &UserSettings {
            video_muted: false,
            ..Default::default()
        },
    )?;
    let original = durable_rows(&store.connection)?;
    let count = event_count(&store)?;
    fixtures::review_counter_seed(&mut store, 2, card("unused", "rating_review", 99, 0))?;
    assert_eq!(event_count(&store)?, count + 4);
    assert_eq!(store.review_sequence()?, 2);
    assert_eq!(store.ratings()?[0].post_id, 99);
    assert_eq!(store.ratings()?[0].revision, 2);
    for (before, after) in original.iter().zip(durable_rows(&store.connection)?) {
        assert!(after.starts_with(before));
    }

    let later: Vec<_> = (0..3)
        .map(|position| card("later", "discover", 10 + position as i64, position))
        .collect();
    store.create_gallery("later", "discover", "", "model", 3, &later, false)?;
    let original = durable_rows(&store.connection)?;
    let count = event_count(&store)?;
    fixtures::recent_view_seed(&mut store, &later[..2])?;
    assert_eq!(event_count(&store)?, count + 2);
    assert_eq!(
        store.recent_views(10)?.into_iter().collect::<HashSet<_>>(),
        HashSet::from([10, 11]),
    );
    for (before, after) in original.iter().zip(durable_rows(&store.connection)?) {
        assert!(after.starts_with(before));
    }
    store.viewed("final-view", 12, "later")?;
    assert_eq!(
        store.recent_views(10)?.into_iter().collect::<HashSet<_>>(),
        HashSet::from([10, 11, 12]),
    );
    store.verify_history()?;
    drop(store);
    let reopened = Store::open(fixture.path().join("boundary-setup.sqlite3"))?;
    assert_eq!(reopened.review_sequence()?, 2);
    assert_eq!(reopened.ratings()?[0].revision, 2);
    assert!(!reopened.preferences()?.unwrap().video_muted);
    reopened.verify_history()?;
    Ok(())
}

#[test]
fn review_history_fixture_retains_offset_and_checkpoint_semantics_without_votes() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("review-history.sqlite3");
    let store = fixtures::review_history(&path, &[3, 1], 500)?;
    assert_eq!(store.review_sequence()?, 500);
    assert_eq!(
        store
            .ratings()?
            .iter()
            .map(|rating| rating.post_id)
            .collect::<Vec<_>>(),
        vec![1, 3]
    );
    let checkpoints = store.checkpoints()?;
    for rating in store.ratings()? {
        assert_eq!(rating.grade, Grade::Like);
        assert_eq!(rating.revision, 1);
        assert_eq!((rating.first_rated_at, rating.updated_at), (1.0, 1.0));
        assert_eq!(rating.context_json, "{}");
        let checkpoint = &checkpoints[&rating.post_id];
        assert_eq!(checkpoint.predicted_grade, 0);
        assert_eq!(checkpoint.recorded_grade, Grade::Like);
        assert_eq!(checkpoint.rating_revision, 1);
        assert_eq!(checkpoint.model_id, "seed-model");
        assert_eq!(checkpoint.review_sequence, 0);
        assert_eq!(checkpoint.updated_at, 1.0);
    }
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM review_facts WHERE kind IN ('vote','vote_seed')"
        )?,
        0
    );
    store.verify_history()?;
    Ok(())
}

#[test]
fn native_initialization_preserves_sql_scalar_storage_and_exact_real_bits() -> Result<()> {
    use seed::{FactTable, InitialFacts, InitialTable, SqlValue};
    let fixture = fixture()?;
    let path = fixture.path().join("native-scalars.sqlite3");
    let fractional = f64::from_bits(0x3fb999999999999b);
    let facts = InitialFacts {
        tables: vec![InitialTable {
            table: FactTable::SelectionFacts,
            rows: vec![vec![
                SqlValue::Integer(1),
                SqlValue::Integer(0),
                SqlValue::Text("gallery".into()),
                SqlValue::Text("scalar-fixture".into()),
                SqlValue::Text("seed".into()),
                SqlValue::Null,
                SqlValue::RealBits(fractional.to_bits()),
                SqlValue::Null,
                SqlValue::Blob(vec![0, 128, 255]),
                SqlValue::Text("opaque\0text".into()),
                SqlValue::Text("model".into()),
                SqlValue::Integer(i64::MIN),
                SqlValue::Integer(i64::MAX),
                SqlValue::Integer(-1),
                SqlValue::Null,
                SqlValue::Null,
            ]],
        }],
    };
    let encoded = serde_json::to_string(&facts)?;
    assert_eq!(serde_json::from_str::<InitialFacts>(&encoded)?, facts);
    let store = fixtures::initialized(&path, facts)?;
    let actual: f64 =
        store
            .connection
            .query_row("SELECT updated_at FROM selection_facts", [], |row| {
                row.get(0)
            })?;
    assert_eq!(actual.to_bits(), fractional.to_bits());
    assert_eq!(
        raw_rows(
            &store.connection,
            "SELECT created_at,typeof(created_at),mode,typeof(mode),filter_text,requested_size,returned_size FROM selection_facts"
        )?,
        vec![vec![
            SqliteValue::Null,
            SqliteValue::Text("null".into()),
            SqliteValue::Blob(vec![0, 128, 255]),
            SqliteValue::Text("blob".into()),
            SqliteValue::Text("opaque\0text".into()),
            SqliteValue::Integer(i64::MIN),
            SqliteValue::Integer(i64::MAX),
        ]],
    );
    assert_eq!(
        serde_json::to_string(&seed::capture(&store.connection)?)?,
        encoded
    );
    let expected = durable_rows(&store.connection)?;
    store.verify_history()?;
    assert_eq!(expected, durable_rows(&store.connection)?);
    drop(store);
    let reopened = Store::open(&path)?;
    reopened.verify_history()?;
    assert_eq!(expected, durable_rows(&reopened.connection)?);
    Ok(())
}

#[test]
fn invalid_native_seed_families_rows_and_constraints_roll_back_initialization() -> Result<()> {
    use seed::{FactTable, InitialFacts, InitialTable, SqlValue};
    let fixture = fixture()?;
    for (index, facts) in [
        InitialFacts {
            tables: vec![
                InitialTable {
                    table: FactTable::PreferenceFacts,
                    rows: Vec::new(),
                },
                InitialTable {
                    table: FactTable::PreferenceFacts,
                    rows: Vec::new(),
                },
            ],
        },
        InitialFacts {
            tables: vec![InitialTable {
                table: FactTable::PreferenceFacts,
                rows: vec![vec![SqlValue::Integer(1), SqlValue::Integer(0)]],
            }],
        },
        InitialFacts {
            tables: vec![InitialTable {
                table: FactTable::PreferenceFacts,
                rows: vec![vec![
                    SqlValue::Integer(2),
                    SqlValue::Integer(0),
                    SqlValue::Text("{}".into()),
                ]],
            }],
        },
        InitialFacts {
            tables: vec![InitialTable {
                table: FactTable::PreferenceFacts,
                rows: vec![vec![
                    SqlValue::Integer(1),
                    SqlValue::Integer(0),
                    SqlValue::Null,
                ]],
            }],
        },
        InitialFacts {
            tables: vec![InitialTable {
                table: FactTable::PreferenceFacts,
                rows: vec![
                    vec![
                        SqlValue::Integer(1),
                        SqlValue::Integer(0),
                        SqlValue::Text("{}".into()),
                    ],
                    vec![
                        SqlValue::Integer(1),
                        SqlValue::Integer(0),
                        SqlValue::Text("{}".into()),
                    ],
                ],
            }],
        },
    ]
    .into_iter()
    .enumerate()
    {
        let path = fixture.path().join(format!("invalid-seed-{index}.sqlite3"));
        assert!(fixtures::initialized(&path, facts).is_err(), "case {index}");
        let connection = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        for table in FACT_TABLES {
            assert_eq!(
                scalar_i64(&connection, &format!("SELECT COUNT(*) FROM {table}"))?,
                0,
                "case {index} left {table} rows",
            );
        }
    }
    for payload in [
        r#"{"tables":[{"table":"ratings","rows":[]}]}"#,
        r#"{"tables":[{"table":"interaction_events","rows":[]}]}"#,
        r#"{"tables":[],"extra":true}"#,
        r#"{"tables":[{"table":"rating_facts","rows":[],"extra":true}]}"#,
    ] {
        assert!(
            serde_json::from_str::<InitialFacts>(payload).is_err(),
            "{payload}"
        );
    }
    Ok(())
}

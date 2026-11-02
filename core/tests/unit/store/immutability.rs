use super::*;

#[test]
fn every_persisted_fact_rejects_update_delete_and_replace_with_recursive_triggers_off() -> Result<()>
{
    let fixture = fixture()?;
    let path = fixture.path().join("immutable.sqlite3");
    let mut store = fixtures::seeded(&path, |connection| {
        connection.execute_batch(
            "INSERT INTO legacy_delivery_facts VALUES(1,0,'gallery_delivery',99,'legacy',1.0)",
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
    store.rate(
        "rate",
        1,
        Grade::Dislike,
        "review",
        Some(prediction(2, "model")),
    )?;
    store.save_preferences("preferences", &UserSettings::default())?;
    store
        .connection
        .pragma_update(None, "recursive_triggers", false)?;
    let before = durable_rows(&store.connection)?;
    for table in FACT_TABLES {
        assert!(
            scalar_i64(&store.connection, &format!("SELECT COUNT(*) FROM {table}"))? > 0,
            "Guard fixture left {table} empty"
        );
        assert!(
            store
                .connection
                .execute(&format!("UPDATE {table} SET sequence=sequence"), [])
                .is_err(),
            "{table} allows UPDATE"
        );
        assert!(
            store
                .connection
                .execute(&format!("DELETE FROM {table}"), [])
                .is_err(),
            "{table} allows DELETE"
        );
        assert!(
            store
                .connection
                .execute(
                    &format!("INSERT OR REPLACE INTO {table} SELECT * FROM {table} LIMIT 1"),
                    []
                )
                .is_err(),
            "{table} allows REPLACE"
        );
        assert_eq!(before, durable_rows(&store.connection)?);
        if *table == "preference_facts" {
            let changed = UserSettings {
                video_muted: false,
                ..Default::default()
            };
            for alias in ["rowid", "_rowid_", "oid"] {
                let replacement = format!(
                    "INSERT OR REPLACE INTO preference_facts({alias},sequence,ordinal,settings_json) \
                     SELECT {alias},sequence,999,?1 FROM preference_facts ORDER BY sequence DESC LIMIT 1"
                );
                assert!(
                    store
                        .connection
                        .execute(&replacement, [serde_json::to_string(&changed)?])
                        .is_err(),
                    "Hidden {alias} replaced a preference without a command"
                );
                assert_eq!(store.preferences()?, Some(UserSettings::default()));
                assert_eq!(before, durable_rows(&store.connection)?);
            }
        }
    }
    store.verify_history()?;
    Ok(())
}

#[test]
fn secondary_identity_replacements_require_immutable_guards_not_other_constraints() -> Result<()> {
    let fixture = fixture()?;
    let mut store = fixtures::seeded(
        &fixture.path().join("secondary-identities.sqlite3"),
        |connection| {
            connection.execute_batch(
            "INSERT INTO legacy_delivery_facts VALUES(1,0,'gallery_delivery',99,'legacy',1.0);
             INSERT INTO activation_facts VALUES(1,0,'seed','seed-context','discover','',1,NULL,0);",
        )?;
            Ok(())
        },
    )?;
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
    store.rate(
        "rate",
        1,
        Grade::Dislike,
        "review",
        Some(prediction(2, "model")),
    )?;
    store.save_preferences("preferences", &UserSettings::default())?;
    store
        .connection
        .pragma_update(None, "recursive_triggers", false)?;
    let before = durable_rows(&store.connection)?;

    // Each SELECT keeps exactly one protected identity and changes the competing
    // identities. In particular, rating_latest necessarily keeps sequence while
    // changing ordinal; preference_facts has no secondary identity at all.
    let cases = [
        (
            "interaction_events",
            "event_id",
            "SELECT ?1+1,event_id,occurred_at,request_json,payload_json,result_json FROM interaction_events WHERE sequence=?1",
        ),
        (
            "selection_facts",
            "initial family/session",
            "SELECT ?1,999,family,session_id,operation,created_at,updated_at,status,mode,filter_text,model_id,requested_size,returned_size,cursor,exhausted,seed FROM selection_facts WHERE operation IN ('seed','create') LIMIT 1",
        ),
        (
            "offer_facts",
            "sequence/ordinal",
            "SELECT sequence,ordinal,family,session_id,999,999,facet,payload_json,card_json,selection_policy,initial_state,initial_delivered_at FROM offer_facts WHERE ?1>sequence LIMIT 1",
        ),
        (
            "offer_facts",
            "family/session/position",
            "SELECT ?1,999,family,session_id,position,999,facet,payload_json,card_json,selection_policy,initial_state,initial_delivered_at FROM offer_facts LIMIT 1",
        ),
        (
            "offer_facts",
            "family/session/post",
            "SELECT ?1,999,family,session_id,999,post_id,facet,payload_json,card_json,selection_policy,initial_state,initial_delivered_at FROM offer_facts LIMIT 1",
        ),
        (
            "rating_facts",
            "post/sequence",
            "SELECT sequence,999,origin,post_id,action,first_rated_at,updated_at,revision,context_json,family,session_id,predicted_grade,recorded_action,rating_revision,judgment_model_id,checkpoint_review_sequence,checkpoint_updated_at FROM rating_facts WHERE ?1>sequence LIMIT 1",
        ),
        (
            "review_facts",
            "vote session/post",
            "SELECT ?1,999,kind,post_id,session_id,999,model_id,excluded_at FROM review_facts WHERE kind IN ('vote_seed','vote') LIMIT 1",
        ),
        (
            "review_facts",
            "vote sequence",
            "SELECT ?1,999,kind,999,'fresh-session',review_sequence,model_id,excluded_at FROM review_facts WHERE kind IN ('vote_seed','vote') LIMIT 1",
        ),
        (
            "presentation_facts",
            "view event",
            "SELECT ?1,999,kind,event_id,session_id,post_id,viewed_at,context_id,NULL,0,NULL,offer_family,offer_position,view_family,seed_card_json,1 FROM presentation_facts WHERE is_view=1 LIMIT 1",
        ),
        (
            "presentation_facts",
            "history entry",
            "SELECT ?1,999,kind,NULL,session_id,post_id,viewed_at,context_id,entry_id,1,999,offer_family,offer_position,view_family,seed_card_json,0 FROM presentation_facts WHERE new_entry=1 LIMIT 1",
        ),
        (
            "presentation_facts",
            "history context/position",
            "SELECT ?1,999,kind,NULL,session_id,post_id,viewed_at,context_id,'fresh-entry',1,entry_position,offer_family,offer_position,view_family,seed_card_json,0 FROM presentation_facts WHERE new_entry=1 LIMIT 1",
        ),
        (
            "activation_facts",
            "seed context",
            "SELECT ?1,999,origin,context_id,mode,filter_text,activated_sequence,seed_cursor,seed_active FROM activation_facts WHERE origin='seed' LIMIT 1",
        ),
        (
            "legacy_delivery_facts",
            "kind/post",
            "SELECT ?1,999,kind,post_id,session_id,occurred_at FROM legacy_delivery_facts LIMIT 1",
        ),
    ];
    for (table, identity, select) in cases {
        store
            .connection
            .execute_batch("SAVEPOINT replacement_attack")?;
        let fresh = scalar_i64(
            &store.connection,
            "SELECT MAX(sequence)+1 FROM interaction_events",
        )?;
        // A controlled, deferred-FK transaction supplies a well-formed envelope.
        // The journal attack targets this unreferenced header, not an event whose
        // replacement could be rejected later because it orphaned existing facts.
        store.connection.execute(
            "INSERT INTO interaction_events SELECT ?1,'replacement-support',occurred_at,request_json,payload_json,result_json FROM interaction_events ORDER BY sequence DESC LIMIT 1",
            [fresh],
        )?;
        let supported = durable_rows(&store.connection)?;
        let count = scalar_i64(&store.connection, &format!("SELECT COUNT(*) FROM {table}"))?;
        let sql = format!("INSERT OR REPLACE INTO {table} {select}");
        let error = store.connection.execute(&sql, [fresh]).expect_err(identity);
        match error {
            rusqlite::Error::SqliteFailure(code, _) => {
                assert_eq!(
                    code.extended_code,
                    rusqlite::ffi::SQLITE_CONSTRAINT_TRIGGER,
                    "{table}: {identity}",
                );
            }
            other => panic!("{table}: {identity}: expected trigger rejection, got {other:?}"),
        }
        assert_eq!(
            supported,
            durable_rows(&store.connection)?,
            "{table}: {identity}"
        );

        // Prove the input actually reaches destructive REPLACE when this guard
        // is absent: no FK, other unique key, or missing rowid error may stand in
        // for the intended rejection. DDL and the replacement are rolled back.
        store
            .connection
            .execute_batch(&format!("DROP TRIGGER {table}_no_replace"))?;
        assert_eq!(
            store.connection.execute(&sql, [fresh])?,
            1,
            "{table}: {identity}"
        );
        assert_eq!(
            scalar_i64(&store.connection, &format!("SELECT COUNT(*) FROM {table}"))?,
            count,
            "{table}: {identity} must replace, not append",
        );
        assert_ne!(
            supported,
            durable_rows(&store.connection)?,
            "{table}: {identity}"
        );
        assert!(raw_rows(&store.connection, "PRAGMA foreign_key_check")?.is_empty());
        store
            .connection
            .execute_batch("ROLLBACK TO replacement_attack; RELEASE replacement_attack")?;
        assert_eq!(
            before,
            durable_rows(&store.connection)?,
            "{table}: {identity}"
        );
    }
    store.verify_history()?;
    Ok(())
}

#[test]
fn current_profile_with_disabled_immutable_guard_is_rejected_without_new_backup() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("missing-guard.sqlite3");
    let store = Store::open(&path)?;
    store
        .connection
        .execute_batch("DROP TRIGGER rating_facts_no_delete")?;
    drop(store);
    let before = fs::read(&path)?;
    assert!(Store::open(&path).is_err());
    assert_eq!(before, fs::read(&path)?);
    assert_eq!(fs::read_dir(fixture.path())?.count(), 1);
    Ok(())
}

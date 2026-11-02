use super::*;

#[test]
fn explicit_views_are_independent_idempotent_and_owner_checked() -> Result<()> {
    let fixture = fixture()?;
    for auto in [false, true] {
        let mut store = Store::open(fixture.path().join(format!("views-{auto}.sqlite3")))?;
        let facet = if auto { "auto" } else { "discover" };
        let cards: Vec<_> = (0..3)
            .map(|i| card("session", facet, i + 1, i as usize))
            .collect();
        if auto {
            store.create_auto("session", "balanced", "", "session-model", 42, &cards)?;
        } else {
            store.create_gallery("session", facet, "", "session-model", 3, &cards, false)?;
        }
        assert!(store.unavailable()?.is_empty());
        assert!(store.recent_views(500)?.is_empty());
        store.rate(
            "vote",
            1,
            Grade::Love,
            "session",
            Some(prediction(2, "judgment-model")),
        )?;
        assert!(store.recent_views(500)?.is_empty());
        store.viewed("event", 2, "session")?;
        let first_time: f64 = store.connection.query_row(
            "SELECT viewed_at FROM presentation_facts WHERE is_view=1",
            [],
            |row| row.get(0),
        )?;
        store.viewed("event", 2, "session")?;
        assert_eq!(
            first_time,
            store.connection.query_row(
                "SELECT viewed_at FROM presentation_facts WHERE is_view=1",
                [],
                |row| row.get::<_, f64>(0)
            )?
        );
        assert_eq!(store.recent_views(500)?, [2]);
        assert!(store.viewed("event", 3, "session").is_err());
        assert!(store.viewed("missing", 999, "session").is_err());
        let other = card("other", facet, 2, 0);
        if auto {
            store.create_auto("other", "balanced", "", "session-model", 42, &[other])?;
        } else {
            store.create_gallery("other", facet, "", "session-model", 1, &[other], false)?;
        }
        assert!(store.viewed("event", 2, "other").is_err());
        store.viewed("back", 2, "session")?;
        assert_eq!(
            scalar_i64(
                &store.connection,
                "SELECT COUNT(*) FROM presentation_facts WHERE is_view=1"
            )?,
            2
        );
        assert_eq!(store.unavailable()?, HashSet::from([1, 2]));
        assert_eq!(
            scalar_i64(
                &store.connection,
                "SELECT COUNT(*) FROM legacy_delivery_facts WHERE kind='auto_play'"
            )?,
            0
        );
        assert_eq!(
            scalar_i64(
                &store.connection,
                "SELECT COUNT(*) FROM presentation_facts WHERE is_view=1 AND post_id=3",
            )?,
            0
        );
    }
    Ok(())
}

#[test]
fn history_is_linear_across_cohorts_back_rating_and_resume_without_persisting_lookahead()
-> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("history.sqlite3"))?;
    let first = card("first-cohort", "discover", 1, 0);
    let second = card("second-cohort", "discover", 2, 0);
    store.create_gallery(
        "first-cohort",
        "discover",
        "",
        "m0",
        1,
        &[first.clone()],
        false,
    )?;
    store.create_gallery(
        "second-cohort",
        "discover",
        "",
        "m1",
        2,
        &[second.clone(), card("second-cohort", "discover", 3, 1)],
        false,
    )?;
    let context = store.activate_context("context", "discover", "", false)?;
    store.presented("entry-one", &context.context_id, None, &first)?;
    store.presented("entry-two", &context.context_id, None, &second)?;
    store.presented("back", &context.context_id, Some("entry-one"), &first)?;
    store.rate(
        "rate-back",
        1,
        Grade::Love,
        "first-cohort",
        Some(prediction(2, "judge")),
    )?;
    let active = store.active_context()?.unwrap();
    assert_eq!(active.entries.len(), 2);
    assert_eq!(active.cursor.as_deref(), Some("entry-one"));
    assert_eq!(active.entries[1].card.session_id, "second-cohort");
    store.activate_context("other-mode", "balanced", "", false)?;
    let resumed = store.activate_context("resume", "discover", "", false)?;
    assert_eq!(resumed.context_id, context.context_id);
    assert_eq!(resumed.entries.len(), 2);
    assert_eq!(resumed.cursor.as_deref(), Some("entry-one"));
    store.presented("forward", &context.context_id, Some("entry-two"), &second)?;
    store.presented("back", &context.context_id, Some("entry-one"), &first)?;
    assert_eq!(
        store.active_context()?.unwrap().cursor.as_deref(),
        Some("entry-two")
    );
    let fresh = store.activate_context("fresh", "discover", "", true)?;
    assert!(fresh.entries.is_empty());
    assert_eq!(
        store
            .activate_context("resume-fresh", "discover", "", false)?
            .context_id,
        fresh.context_id
    );
    assert_eq!(
        facts::read_context(&store.connection, &context.context_id)?
            .entries
            .len(),
        2
    );
    store.verify_history()?;
    assert_eq!(
        store.active_context()?.unwrap().context_id,
        fresh.context_id
    );
    Ok(())
}

#[test]
fn fractional_offer_scores_keep_bits_through_revisit_restart_and_verification() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("fractional-history.sqlite3");
    let mut store = Store::open(&path)?;
    let mut first = card("gallery", "discover", 1, 0);
    first.selection_score = 0.1f32;
    let mut second = card("gallery", "discover", 2, 1);
    second.selection_score = 0.3f32;
    store.create_gallery(
        "gallery",
        "discover",
        "",
        "model",
        2,
        &[first.clone(), second.clone()],
        false,
    )?;
    let original_payload: String = store.connection.query_row(
        "SELECT payload_json FROM offer_facts WHERE family='gallery' AND post_id=1",
        [],
        |row| row.get(0),
    )?;
    let context = store.activate_context("context", "discover", "", false)?;
    store.presented("first", &context.context_id, None, &first)?;
    store.presented("second", &context.context_id, None, &second)?;
    let mut first_latest_grade = first.clone();
    first_latest_grade.current_grade = Some(Grade::Like);
    store.presented(
        "back",
        &context.context_id,
        Some("first"),
        &first_latest_grade,
    )?;
    assert_eq!(
        store.active_context()?.unwrap().cursor.as_deref(),
        Some("first")
    );
    drop(store);
    let mut store = Store::open(&path)?;
    let restored = store.active_context()?.unwrap();
    assert_eq!(
        restored.entries[0].card.selection_score.to_bits(),
        first.selection_score.to_bits()
    );
    assert_eq!(
        restored.entries[1].card.selection_score.to_bits(),
        second.selection_score.to_bits()
    );
    store.presented("forward", &context.context_id, Some("second"), &second)?;
    let before = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    store.verify_history()?;
    assert_eq!(before, durable_rows(&store.connection)?);
    store.presented(
        "back-after-rebuild",
        &context.context_id,
        Some("first"),
        &first_latest_grade,
    )?;
    let retained_payload: String = store.connection.query_row(
        "SELECT payload_json FROM offer_facts WHERE family='gallery' AND post_id=1",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(retained_payload, original_payload);
    let retained: Value = serde_json::from_str(&retained_payload)?;
    let retained_card: Card = serde_json::from_value(retained["card"].clone())?;
    assert_eq!(
        retained_card.selection_score.to_bits(),
        first.selection_score.to_bits()
    );
    assert_eq!(
        store.active_context()?.unwrap().entries[0]
            .card
            .selection_score
            .to_bits(),
        first.selection_score.to_bits()
    );
    Ok(())
}

#[test]
fn presentations_require_active_specification_and_exact_existing_offer_identity() -> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("presentation-ownership.sqlite3"))?;
    let first = card("gallery", "discover", 1, 0);
    let second = card("gallery", "discover", 2, 1);
    store.create_gallery(
        "gallery",
        "discover",
        "+wolf",
        "model",
        2,
        &[first.clone(), second.clone()],
        false,
    )?;
    let own = store.activate_context("own", "discover", "+wolf", false)?;
    store.presented("entry", &own.context_id, None, &first)?;
    let wrong_filter = store.activate_context("wrong-filter", "discover", "-wolf", false)?;
    let before = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    let count = event_count(&store)?;
    assert!(
        store
            .presented("inactive", &own.context_id, Some("entry"), &first)
            .is_err()
    );
    assert!(
        store
            .presented("wrong-filter-view", &wrong_filter.context_id, None, &first)
            .is_err()
    );
    assert_eq!(before, durable_rows(&store.connection)?);
    assert_eq!(count, event_count(&store)?);
    let wrong_mode = store.activate_context("wrong-mode", "develop", "+wolf", false)?;
    assert!(
        store
            .presented("wrong-mode-view", &wrong_mode.context_id, None, &first)
            .is_err()
    );
    store.activate_context("resume", "discover", "+wolf", false)?;
    assert!(
        store
            .presented("wrong-offer", &own.context_id, Some("entry"), &second)
            .is_err()
    );
    let mut changed_grade = first.clone();
    changed_grade.current_grade = Some(Grade::Love);
    store.presented(
        "existing-entry",
        &own.context_id,
        Some("entry"),
        &changed_grade,
    )?;
    let expected = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    store.verify_history()?;
    assert_eq!(expected, durable_rows(&store.connection)?);
    assert_eq!(store.active_context()?.unwrap().entries.len(), 1);
    Ok(())
}

#[test]
fn recent_views_follow_max_timestamp_not_append_order_and_delivery_is_first_occurrence()
-> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("recency.sqlite3"))?;
    store.create_gallery(
        "gallery",
        "discover",
        "",
        "model",
        3,
        &[
            card("gallery", "discover", 1, 0),
            card("gallery", "discover", 2, 1),
            card("gallery", "discover", 3, 2),
        ],
        false,
    )?;
    for (id, post_id, timestamp) in [
        ("a", 2, 20.0),
        ("b", 1, 20.0),
        ("c", 3, 10.0),
        ("d", 2, 1.0),
    ] {
        append_at(
            &mut store,
            id,
            timestamp,
            Event::Viewed {
                post_id,
                session: "gallery".into(),
            },
        )?;
    }
    assert_eq!(store.recent_views(2)?, [1, 2]);
    assert_eq!(store.recent_views(3)?, [1, 2, 3]);
    assert!(store.recent_views(0).is_err());
    assert_eq!(
        raw_rows(
            &store.connection,
            "SELECT viewed_at FROM presentation_facts WHERE is_view=1 AND post_id=2 ORDER BY sequence LIMIT 1",
        )?,
        vec![vec![SqliteValue::Real(20.0)]],
    );
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM presentation_facts WHERE is_view=1"
        )?,
        4
    );
    store.verify_history()?;
    Ok(())
}

#[test]
fn seeded_browse_history_preferences_and_card_bytes_survive_replay_and_resume() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("imported-browse.sqlite3");
    let first = card("retained-gallery", "discover", 11, 0);
    let second = card("retained-gallery", "discover", 12, 1);
    let first_json = serde_json::to_string(&first)?;
    let second_json = serde_json::to_string(&second)?;
    let settings = UserSettings {
        blacklist: vec!["wolf".into()],
        ..Default::default()
    };
    let settings_json = format!(" {} ", serde_json::to_string(&settings)?);
    let mut store = fixtures::seeded(&path, |connection| {
        connection.execute(
            "INSERT INTO selection_facts VALUES(1,0,'gallery','retained-gallery','seed',1.0,2.0,'active','discover','+wolf','historical-model',2,2,-1,0,NULL)",
            [],
        )?;
        for (position, card) in [&first, &second].into_iter().enumerate() {
            connection.execute(
                "INSERT INTO offer_facts VALUES(1,?1,'gallery','retained-gallery',?1,?2,'discover',?3,?4,'retained-policy','selected',NULL)",
                params![position as i64, card.post_id, offered_payload(card, "retained-policy")?, serde_json::to_string(card)?],
            )?;
        }
        connection.execute_batch(
            "INSERT INTO activation_facts VALUES(1,0,'seed','retained-context','discover','+wolf',17,'retained-first',1)",
        )?;
        for (position, (entry, card_json)) in [
            ("retained-first", &first_json),
            ("retained-second", &second_json),
        ]
        .into_iter()
        .enumerate()
        {
            connection.execute(
                "INSERT INTO presentation_facts(sequence,ordinal,kind,entry_id,context_id,entry_position,seed_card_json,new_entry,is_view) VALUES(1,?1,'entry_seed',?2,'retained-context',?1,?3,1,0)",
                params![position as i64, entry, card_json],
            )?;
        }
        connection.execute(
            "INSERT INTO preference_facts VALUES(1,0,?1)",
            [&settings_json],
        )?;
        Ok(())
    })?;
    let active = store.active_context()?.unwrap();
    assert_eq!(active.cursor.as_deref(), Some("retained-first"));
    assert_eq!(active.entries.len(), 2);
    assert_eq!(active.entries[0].card.post_id, 11);
    assert_eq!(active.entries[1].card.post_id, 12);
    assert_eq!(store.preferences()?, Some(settings));
    assert!(store.recent_views(10)?.is_empty());
    let expected = durable_rows(&store.connection)?;
    store.verify_history()?;
    assert_eq!(expected, durable_rows(&store.connection)?);
    assert_eq!(
        raw_rows(
            &store.connection,
            "SELECT seed_card_json FROM presentation_facts WHERE kind='entry_seed' ORDER BY entry_position"
        )?,
        vec![
            vec![SqliteValue::Text(first_json)],
            vec![SqliteValue::Text(second_json)]
        ],
    );
    assert_eq!(
        raw_rows(
            &store.connection,
            "SELECT settings_json FROM preference_facts"
        )?,
        vec![vec![SqliteValue::Text(settings_json)]],
    );
    store.presented(
        "resume-forward",
        &active.context_id,
        Some("retained-second"),
        &second,
    )?;
    assert_eq!(
        store.active_context()?.unwrap().cursor.as_deref(),
        Some("retained-second")
    );
    assert_eq!(store.active_context()?.unwrap().entries.len(), 2);
    let continued = durable_rows(&store.connection)?;
    drop(store);
    let reopened = Store::open(&path)?;
    reopened.verify_history()?;
    assert_eq!(continued, durable_rows(&reopened.connection)?);
    assert_eq!(
        reopened.active_context()?.unwrap().cursor.as_deref(),
        Some("retained-second")
    );
    Ok(())
}

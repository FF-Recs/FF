use super::*;

fn gallery_owner(session: &str, capacity: usize, issued: usize) -> GalleryOwner<'_> {
    GalleryOwner {
        session,
        facet: "discover",
        filter: "+wolf",
        capacity,
        issued,
    }
}

#[test]
fn replay_preserves_captured_offer_policy_and_exact_bytes_for_create_and_append() -> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("captured-offers.sqlite3"))?;
    let gallery_policy = "gallery-retired-test-policy";
    let auto_policy = "auto-retired-test-policy";
    assert_ne!(gallery_policy, crate::selection::GALLERY_POLICY);
    assert_ne!(auto_policy, crate::selection::AUTO_POLICY);
    let capture =
        |session: &str, facet: &str, post_id, position, policy: &str| -> Result<Vec<OfferFact>> {
            let mut facts = capture_offers(
                session,
                position,
                &[card(session, facet, post_id, position)],
                policy,
            )?;
            // Preserve transport bytes, not just semantically equivalent JSON.
            let value: Value = serde_json::from_str(&facts[0].payload_json)?;
            facts[0].payload_json = format!("\n{}\n", serde_json::to_string_pretty(&value)?);
            Ok(facts)
        };
    store.commit_generated(Event::GalleryCommitted {
        owner: gallery_owner("gallery", 2, 0).into(),
        model_id: "gallery-header".into(),
        offers: capture("gallery", "discover", 1, 0, gallery_policy)?,
        completed: false,
        exhausted: false,
        previous: None,
    })?;
    store.commit_generated(Event::GalleryCommitted {
        owner: gallery_owner("gallery", 2, 1).into(),
        model_id: "gallery-append".into(),
        offers: capture("gallery", "discover", 2, 1, gallery_policy)?,
        completed: true,
        exhausted: false,
        previous: None,
    })?;
    store.commit_generated(Event::AutoCreated {
        session: "auto".into(),
        preset: "balanced".into(),
        filter: "".into(),
        model_id: "auto-header".into(),
        seed: 42,
        offers: capture("auto", "play", 3, 0, auto_policy)?,
    })?;
    store.commit_generated(Event::AutoAppended {
        session: "auto".into(),
        start: 1,
        offers: capture("auto", "play", 4, 1, auto_policy)?,
    })?;
    let before = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    store.verify_history()?;
    assert_eq!(before, durable_rows(&store.connection)?);
    for (session, ids, expected_policy) in [
        ("gallery", [1, 2], gallery_policy),
        ("auto", [3, 4], auto_policy),
    ] {
        for post_id in ids {
            let context: Value = serde_json::from_str(&store.context(session, post_id)?.json)?;
            assert_eq!(context["selection_policy"], expected_policy);
            let bytes: String = store.connection.query_row(
                "SELECT payload_json FROM offer_facts WHERE session_id=?1 AND post_id=?2",
                params![session, post_id],
                |row| row.get(0),
            )?;
            assert!(bytes.starts_with('\n') && bytes.ends_with('\n'));
            assert!(bytes.contains("\n  \""));
        }
    }
    Ok(())
}

#[test]
fn gallery_append_preserves_v2_payload_retry_and_initial_model_with_new_offer_identity()
-> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("gallery-append.sqlite3"))?;
    let mut first = card("cohort", "discover", 1, 0);
    first.universe_size = 1;
    first.model_id = "M0".into();
    let old_payload = offered_payload(&first, "gallery-rust-v2")?;
    store.commit_generated(Event::GalleryCommitted {
        owner: gallery_owner("cohort", 3, 0).into(),
        model_id: "M0".into(),
        offers: capture_offers("cohort", 0, &[first.clone()], "gallery-rust-v2")?,
        completed: false,
        exhausted: false,
        previous: None,
    })?;
    let old_context = store.context("cohort", 1)?.json;
    assert!(
        store
            .rate(
                "first-vote",
                1,
                Grade::Love,
                "cohort",
                Some(prediction(2, "judge-M0"))
            )?
            .changed
    );
    let mut second = card("cohort", "discover", 2, 1);
    second.model_id = "M1".into();
    second.universe_size = 2;
    store.commit_gallery(
        gallery_owner("cohort", 3, 1),
        "M1",
        &[second],
        false,
        false,
        None,
    )?;
    assert_eq!(
        store.connection.query_row(
            "SELECT payload_json FROM offer_facts WHERE family='gallery' AND position=0",
            [],
            |row| row.get::<_, String>(0)
        )?,
        old_payload
    );
    assert_eq!(
        store
            .connection
            .query_row("SELECT model_id FROM q_gallery_sessions", [], |row| row
                .get::<_, String>(
                0
            ))?,
        "M0"
    );
    assert_eq!(store.context("cohort", 1)?.json, old_context);
    assert!(
        !store
            .rate(
                "first-vote",
                1,
                Grade::Love,
                "cohort",
                Some(prediction(0, "new-judge"))
            )?
            .changed
    );
    let context: Value = serde_json::from_str(&store.context("cohort", 2)?.json)?;
    assert_eq!(context["model_id"], "M1");
    assert_eq!(context["rank"], 1);
    assert_eq!(
        context["selection_policy"],
        crate::selection::GALLERY_POLICY
    );
    assert_eq!(context["source"], "gallery-rust-v1");
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT returned_size FROM q_gallery_sessions"
        )?,
        2
    );
    let mut last = card("cohort", "discover", 3, 2);
    last.model_id = "M1".into();
    store.commit_gallery(
        gallery_owner("cohort", 3, 2),
        "M1",
        &[last],
        true,
        false,
        None,
    )?;
    let (status, exhausted): (String, bool) = store.connection.query_row(
        "SELECT status,(SELECT exhausted FROM selection_facts WHERE family='gallery' AND exhausted IS NOT NULL ORDER BY sequence DESC,ordinal DESC LIMIT 1) FROM q_gallery_sessions",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    assert_eq!(status, "completed");
    assert!(!exhausted, "A full group is not global exhaustion");
    Ok(())
}

#[test]
fn gallery_append_rejects_wrong_owner_stale_position_capacity_duplicates_and_closed_groups()
-> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("gallery-owner.sqlite3"))?;
    store.commit_gallery(
        gallery_owner("cohort", 3, 0),
        "M0",
        &[card("cohort", "discover", 1, 0)],
        false,
        false,
        None,
    )?;
    let second = card("cohort", "discover", 2, 1);
    for (facet, filter, capacity, issued) in [
        ("develop", "+wolf", 3, 1),
        ("discover", "-wolf", 3, 1),
        ("discover", "+wolf", 4, 1),
        ("discover", "+wolf", 3, 2),
    ] {
        assert!(
            store
                .commit_gallery(
                    GalleryOwner {
                        session: "cohort",
                        facet,
                        filter,
                        capacity,
                        issued
                    },
                    "M1",
                    &[second.clone()],
                    false,
                    false,
                    None
                )
                .is_err()
        );
    }
    let mut wrong_session = second.clone();
    wrong_session.session_id = "elsewhere".into();
    assert!(
        store
            .commit_gallery(
                gallery_owner("cohort", 3, 1),
                "M1",
                &[wrong_session],
                false,
                false,
                None
            )
            .is_err()
    );
    let mut duplicate = second.clone();
    duplicate.post_id = 1;
    assert!(
        store
            .commit_gallery(
                gallery_owner("cohort", 3, 1),
                "M1",
                &[second.clone(), card("cohort", "discover", 1, 2)],
                true,
                false,
                None
            )
            .is_err()
    );
    assert!(
        store
            .commit_gallery(
                gallery_owner("cohort", 3, 1),
                "M1",
                &[duplicate],
                false,
                false,
                None
            )
            .is_err()
    );
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM offer_facts WHERE family='gallery'"
        )?,
        1
    );
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT returned_size FROM q_gallery_sessions"
        )?,
        1
    );
    assert!(
        store
            .commit_gallery(
                gallery_owner("cohort", 3, 1),
                "M1",
                &[
                    second.clone(),
                    card("cohort", "discover", 3, 2),
                    card("cohort", "discover", 4, 3)
                ],
                true,
                false,
                None
            )
            .is_err()
    );
    store.commit_gallery(gallery_owner("cohort", 3, 1), "M1", &[], true, false, None)?;
    assert!(
        store
            .commit_gallery(
                gallery_owner("cohort", 3, 1),
                "M1",
                &[second],
                false,
                false,
                None
            )
            .is_err()
    );
    store.commit_gallery(gallery_owner("cohort", 3, 1), "M1", &[], true, true, None)?;
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT exhausted FROM selection_facts WHERE family='gallery' AND exhausted IS NOT NULL ORDER BY sequence DESC,ordinal DESC LIMIT 1"
        )?,
        1
    );
    Ok(())
}

#[test]
fn gallery_boundary_and_restart_retirement_rollback_with_new_session_and_items() -> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("gallery-transition.sqlite3"))?;
    store.commit_gallery(
        gallery_owner("old", 3, 0),
        "M0",
        &[card("old", "discover", 1, 0)],
        false,
        false,
        None,
    )?;
    let immutable = durable_rows(&store.connection)?;
    store.connection.execute_batch(
        "CREATE TRIGGER fail_retire BEFORE INSERT ON selection_facts WHEN NEW.operation='retire' AND NEW.session_id='old'
         BEGIN SELECT RAISE(ABORT,'retirement failure'); END;",
    )?;
    assert!(
        store
            .commit_gallery(
                gallery_owner("new", 3, 0),
                "M1",
                &[card("new", "discover", 2, 0)],
                false,
                false,
                Some((gallery_owner("old", 3, 1), true))
            )
            .is_err()
    );
    assert_eq!(event_count(&store)?, 2);
    assert_eq!(immutable, durable_rows(&store.connection)?);
    assert_eq!(
        scalar_i64(&store.connection, "SELECT COUNT(*) FROM q_gallery_sessions")?,
        1
    );
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM offer_facts WHERE family='gallery'"
        )?,
        1
    );
    assert_eq!(
        store
            .connection
            .query_row("SELECT status FROM q_gallery_sessions", [], |row| row
                .get::<_, String>(0))?,
        "active"
    );
    store.connection.execute_batch("DROP TRIGGER fail_retire")?;
    store.commit_gallery(
        gallery_owner("new", 3, 0),
        "M1",
        &[card("new", "discover", 2, 0)],
        false,
        false,
        Some((gallery_owner("old", 3, 1), true)),
    )?;
    assert_eq!(
        store.connection.query_row(
            "SELECT status FROM q_gallery_sessions WHERE session_id='old'",
            [],
            |row| row.get::<_, String>(0)
        )?,
        "completed"
    );
    store.commit_gallery(
        gallery_owner("restart", 3, 0),
        "M1",
        &[card("restart", "discover", 3, 0)],
        false,
        false,
        Some((gallery_owner("new", 3, 1), false)),
    )?;
    assert_eq!(
        store.connection.query_row(
            "SELECT status FROM q_gallery_sessions WHERE session_id='new'",
            [],
            |row| row.get::<_, String>(0)
        )?,
        "abandoned"
    );
    assert!(store.context("old", 1).is_ok());
    assert!(store.context("new", 2).is_ok());
    store.commit_gallery(
        gallery_owner("restart", 3, 1),
        "M2",
        &[card("restart", "discover", 4, 1)],
        false,
        false,
        None,
    )?;
    store.retire_gallery(gallery_owner("restart", 3, 2))?;
    let expected = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    store.verify_history()?;
    assert_eq!(expected, durable_rows(&store.connection)?);
    Ok(())
}

#[test]
fn auto_append_keeps_generation_provenance_and_rolls_back_duplicates() -> Result<()> {
    let fixture = fixture()?;
    let mut store = Store::open(fixture.path().join("auto.sqlite3"))?;
    store.create_auto(
        "auto",
        "balanced",
        "",
        "first-generation",
        42,
        &[card("auto", "auto", 1, 0)],
    )?;
    let mut next = card("auto", "auto", 2, 1);
    next.model_id = "second-generation".into();
    store.append_auto("auto", 1, &[next])?;
    let context: Value = serde_json::from_str(&store.context("auto", 2)?.json)?;
    assert_eq!(context["model_id"], "second-generation");
    assert_eq!(context["rank"], 1);
    let duplicate = [card("auto", "auto", 3, 2), card("auto", "auto", 2, 3)];
    assert!(store.append_auto("auto", 2, &duplicate).is_err());
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM offer_facts WHERE family='auto'"
        )?,
        2
    );
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT returned_size FROM q_auto_sessions"
        )?,
        2
    );
    assert!(
        store
            .append_auto("auto", 7, &[card("auto", "auto", 4, 7)])
            .is_err()
    );
    assert!(store.unavailable()?.is_empty());
    Ok(())
}

#[test]
fn native_auto_policy_cutover_preserves_old_payload_and_exact_vote_retry() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("auto-policy-cutover.sqlite3");
    let mut store = Store::open(&path)?;
    let old_card = card("old-auto", "play", 1, 0);
    let old_payload = serde_json::to_string(&json!({
        "format_version": 1, "selection_policy": "auto-rust-v1", "card": old_card,
    }))?;
    store.commit_generated(Event::AutoCreated {
        session: "old-auto".into(),
        preset: "balanced".into(),
        filter: "".into(),
        model_id: "session-old".into(),
        seed: 7,
        offers: vec![OfferFact {
            post_id: 1,
            position: 0,
            facet: "play".into(),
            payload_json: old_payload.clone(),
        }],
    })?;
    let first = store.rate(
        "old-vote",
        1,
        Grade::Like,
        "old-auto",
        Some(prediction(2, "serving-before")),
    )?;
    let old_rating = store.ratings()?.remove(0);
    let old_context: Value = serde_json::from_str(&old_rating.context_json)?;
    assert_eq!(old_context["selection_policy"], "auto-rust-v1");
    store.create_auto(
        "new-auto",
        "balanced",
        "",
        "session-new",
        8,
        &[card("new-auto", "play", 2, 0)],
    )?;
    let new_context: Value = serde_json::from_str(&store.context("new-auto", 2)?.json)?;
    assert_eq!(
        new_context["selection_policy"],
        crate::selection::AUTO_POLICY
    );
    drop(store);
    let mut reopened = Store::open(&path)?;
    let retry = reopened.rate(
        "old-vote",
        1,
        Grade::Like,
        "old-auto",
        Some(prediction(0, "serving-after")),
    )?;
    assert!(!retry.changed);
    assert_eq!(retry.revision, first.revision);
    let retained = reopened.ratings()?.remove(0);
    assert_eq!(retained.context_json, old_rating.context_json);
    assert_eq!(retained.updated_at, old_rating.updated_at);
    let payload: String = reopened.connection.query_row(
        "SELECT payload_json FROM offer_facts WHERE family='auto' AND session_id='old-auto'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(payload, old_payload);
    Ok(())
}

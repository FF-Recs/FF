use super::*;

fn seed_historical_offer(
    connection: &Connection,
    session: &str,
    auto: bool,
    payloads: &[String],
) -> Result<()> {
    let family = if auto { "auto" } else { "gallery" };
    let facet = if auto { "play" } else { "discover" };
    let ordinal = scalar_i64(connection, "SELECT COUNT(*) FROM selection_facts")?;
    connection.execute(
        "INSERT INTO selection_facts VALUES(1,?1,?2,?3,'seed',1.0,2.0,'active',?4,'+wolf','historical-model',?5,?6,-1,?7,?8)",
        params![ordinal, family, session, if auto { "fresh" } else { facet },
            if auto { None } else { Some(payloads.len() as i64) },
            payloads.len() as i64, if auto { None } else { Some(0i64) },
            if auto { Some(23i64) } else { None }],
    )?;
    let first_ordinal = scalar_i64(connection, "SELECT COUNT(*) FROM offer_facts")?;
    for (position, payload) in payloads.iter().enumerate() {
        #[derive(Deserialize)]
        struct NativeOffer {
            format_version: u8,
            selection_policy: String,
            card: Card,
        }
        let (card_json, policy) = match serde_json::from_str::<NativeOffer>(payload) {
            Ok(offer) if offer.format_version == 1 => (
                Some(serde_json::to_string(&offer.card)?),
                Some(offer.selection_policy),
            ),
            _ => (None, None),
        };
        connection.execute(
            "INSERT INTO offer_facts VALUES(1,?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,NULL)",
            params![
                first_ordinal + position as i64,
                family,
                session,
                position as i64,
                position as i64 + 1,
                facet,
                payload,
                card_json,
                policy,
                if auto { None } else { Some("selected") }
            ],
        )?;
    }
    Ok(())
}

#[test]
fn historical_offer_context_keeps_opaque_bytes_and_per_card_generation() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("historical-context.sqlite3");
    let hash = "0123456789abcdef0123456789abcdef";
    let cases = vec![
        (
            format!(
                r#"{{"Md5":"{}","FileExtension":"PNG","ModelId":"card-model"}}"#,
                hash.to_ascii_uppercase()
            ),
            Some(hash),
            Some("png"),
            "card-model",
        ),
        (
            format!(r#"{{"md5":"{hash}","file_ext":"webm"}}"#),
            Some(hash),
            Some("webm"),
            "historical-model",
        ),
        (
            format!(r#"{{"Card":{{"Md5":"{hash}","FileExtension":"gif","ModelId":"nested"}}}}"#),
            Some(hash),
            Some("gif"),
            "nested",
        ),
        ("{}".into(), None, None, "historical-model"),
        ("not-json".into(), None, None, "historical-model"),
        ("[]".into(), None, None, "historical-model"),
        (
            r#"{"Md5":"invalid","FileExtension":"../png","ModelId":""}"#.into(),
            None,
            None,
            "historical-model",
        ),
        (
            format!(r#"{{"Md5":"{hash}","FileExtension":null,"ModelId":5}}"#),
            Some(hash),
            None,
            "historical-model",
        ),
    ];
    let payloads: Vec<_> = cases.iter().map(|case| case.0.clone()).collect();
    let store = fixtures::seeded(&path, |connection| {
        seed_historical_offer(connection, "gallery", false, &payloads)?;
        seed_historical_offer(connection, "auto", true, &payloads)
    })?;
    let expected = {
        let connection: &Connection = &store.connection;
        durable_rows(connection)
    }?;
    for session in ["gallery", "auto"] {
        for (index, (_, md5, extension, model)) in cases.iter().enumerate() {
            let context = store.context(session, index as i64 + 1)?.json;
            let value: Value = serde_json::from_str(&context)?;
            assert_eq!(value["media_md5"].as_str(), *md5);
            assert_eq!(value["media_file_extension"].as_str(), *extension);
            assert_eq!(value["model_id"], *model);
            assert_eq!(value["filter"], "+wolf");
            assert_eq!(value["rank"], index);
            assert!(value.get("selection_policy").is_none());
        }
    }
    store.verify_history()?;
    assert_eq!(expected, durable_rows(&store.connection)?);
    Ok(())
}

#[test]
fn historical_rating_noop_preserves_exact_context_timestamps_and_revision() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("historical-retry.sqlite3");
    let payload = r#"{"Md5":"0123456789abcdef0123456789abcdef","FileExtension":"png","ModelId":"original-model"}"#;
    let historical_context = serde_json::to_string(&json!({
        "source": "gallery-rust-v1", "media_md5": "0123456789abcdef0123456789abcdef",
        "media_file_extension": "png", "gallery_session_id": "old", "facet": "discover",
        "filter": "+wolf", "model_id": "original-model", "rank": 0,
    }))?;
    let mut store = fixtures::seeded(&path, |connection| {
        seed_historical_offer(connection, "old", false, &[payload.into()])?;
        connection.execute(
            "INSERT INTO rating_facts VALUES(1,0,'seed',1,'like',1.0,2.0,7,?1,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)",
            [&historical_context],
        )?;
        Ok(())
    })?;
    let result = store.rate(
        "historical-noop",
        1,
        Grade::Like,
        "old",
        Some(prediction(2, "current-model")),
    )?;
    assert!(!result.changed);
    assert_eq!(result.revision, 7);
    let rating = &store.ratings()?[0];
    assert_eq!(rating.context_json, historical_context);
    assert_eq!((rating.first_rated_at, rating.updated_at), (1.0, 2.0));
    assert!(store.checkpoints()?.is_empty());
    let retained: String = store.connection.query_row(
        "SELECT payload_json FROM offer_facts WHERE family='gallery'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(retained, payload);
    store.verify_history()?;
    Ok(())
}

#[test]
fn unknown_native_offer_version_is_not_treated_as_historical() -> Result<()> {
    let fixture = fixture()?;
    let path = fixture.path().join("future-offer.sqlite3");
    let mut store = fixtures::seeded(&path, |connection| {
        seed_historical_offer(
            connection,
            "future",
            false,
            &[r#"{"format_version":2,"card":{}}"#.into()],
        )
    })?;
    let before = durable_rows(&store.connection)?;
    assert!(
        store
            .rate(
                "future-vote",
                1,
                Grade::Love,
                "future",
                Some(prediction(3, "model"))
            )
            .is_err()
    );
    assert!(store.ratings()?.is_empty());
    assert!(store.checkpoints()?.is_empty());
    assert_eq!(before, durable_rows(&store.connection)?);
    Ok(())
}

#[test]
fn overlapping_seeded_offer_families_keep_gallery_presentation_and_auto_view_identity() -> Result<()>
{
    let fixture = fixture()?;
    let path = fixture.path().join("overlapping-families.sqlite3");
    let gallery = card("shared", "discover", 1, 0);
    let mut auto = card("shared", "play", 1, 0);
    auto.model_id = "auto-card-model".into();
    let mut store = fixtures::seeded(&path, |connection| {
        seed_historical_offer(
            connection,
            "shared",
            false,
            &[offered_payload(&gallery, "gallery-historical")?],
        )?;
        seed_historical_offer(
            connection,
            "shared",
            true,
            &[offered_payload(&auto, "auto-historical")?],
        )
    })?;
    let context: Value = serde_json::from_str(&store.context("shared", 1)?.json)?;
    assert_eq!(context["model_id"], "auto-card-model");
    assert_eq!(context["selection_policy"], "auto-historical");
    let browse = store.activate_context("activate", "discover", "+wolf", false)?;
    store.presented("show", &browse.context_id, None, &gallery)?;
    let restored = store.active_context()?.unwrap();
    assert_eq!(restored.entries.len(), 1);
    assert_eq!(restored.entries[0].card.model_id, gallery.model_id);
    assert_eq!(store.recent_views(10)?, [1]);
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM presentation_facts WHERE is_view=1 AND view_family='gallery'",
        )?,
        0,
        "Auto-first actual view must retain its Auto offer identity",
    );
    store.verify_history()?;
    Ok(())
}

#[test]
fn seeded_orphan_delivery_stays_immutable_without_inventing_views_after_native_rating() -> Result<()>
{
    let fixture = fixture()?;
    let path = fixture.path().join("orphan-delivery.sqlite3");
    let mut store = fixtures::seeded(&path, |connection| {
        connection.execute_batch(
            "INSERT INTO legacy_delivery_facts VALUES(1,0,'gallery_delivery',3,'missing-old-owner',7.25)",
        )?;
        Ok(())
    })?;
    assert!(store.unavailable()?.is_empty());
    assert!(store.recent_views(5)?.is_empty());
    store.create_gallery(
        "new",
        "discover",
        "",
        "new-model",
        1,
        &[card("new", "discover", 3, 0)],
        false,
    )?;
    store.rate("vote", 3, Grade::Like, "new", None)?;
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM legacy_delivery_facts WHERE post_id=3"
        )?,
        1,
    );
    assert_eq!(store.unavailable()?, HashSet::from([3]));
    assert!(store.recent_views(5)?.is_empty());
    store.verify_history()?;
    Ok(())
}

#[test]
fn matching_seeded_view_retry_keeps_original_occurrence_and_reserves_command_identity() -> Result<()>
{
    let fixture = fixture()?;
    let path = fixture.path().join("imported-view.sqlite3");
    let mut store = fixtures::seeded(&path, |connection| {
        seed_historical_offer(connection, "old", true, &["{}".into(), "{}".into()])?;
        connection.execute_batch(
            "INSERT INTO presentation_facts VALUES(1,0,'view_seed','historical','old',1,12.25,NULL,NULL,0,NULL,NULL,NULL,NULL,NULL,1)",
        )?;
        Ok(())
    })?;
    let imported = durable_rows(&store.connection)?;
    assert!(
        store
            .save_preferences("historical", &UserSettings::default())
            .is_err()
    );
    assert!(store.viewed("historical", 2, "old").is_err());
    assert_eq!(imported, durable_rows(&store.connection)?);
    store.viewed("historical", 1, "old")?;
    let after_first = durable_rows(&store.connection)?;
    store.viewed("historical", 1, "old")?;
    assert_eq!(after_first, durable_rows(&store.connection)?);
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM presentation_facts WHERE is_view=1"
        )?,
        1
    );
    let timestamp: f64 = store.connection.query_row(
        "SELECT viewed_at FROM presentation_facts WHERE is_view=1 AND event_id='historical'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(timestamp, 12.25);
    assert!(store.viewed("historical", 2, "old").is_err());
    store.viewed("new-actual-occurrence", 1, "old")?;
    assert_eq!(
        scalar_i64(
            &store.connection,
            "SELECT COUNT(*) FROM presentation_facts WHERE is_view=1"
        )?,
        2
    );
    store.verify_history()?;
    Ok(())
}

use super::*;
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_startup_imports_once_and_restart_restores_journal_authority() -> Result<()> {
    let (temporary, mut config) = support::fixture(8)?;
    ff_core::corpus::verify_corpus(&config.corpus_path)?;
    let import = b"version=1\n[input]\nleft_hand_mode=true\n[playback]\nmuted=false\n[auto]\nselection='fresh'\ndelay_seconds=11\n";
    std::fs::write(&config.settings_path, import)?;
    let app = AppHandle::open(config.clone()).await?;
    assert_eq!(
        app.health().await?.model_status.route,
        model::ModelRoute::Starter
    );
    let imported = app.settings().await?;
    assert!(imported.left_handed);
    assert!(!imported.video_muted);
    assert_eq!(
        (imported.auto_preset.as_str(), imported.auto_delay_seconds),
        ("fresh", 11)
    );
    assert_eq!(std::fs::read(&config.settings_path)?, import);
    let context = app
        .activate_context(
            "initial-context".into(),
            "discover".into(),
            "vulpine".into(),
            true,
        )
        .await?;
    let same_context = app
        .activate_context(
            "canonical-context".into(),
            "discover".into(),
            "fox".into(),
            false,
        )
        .await?;
    assert_eq!(same_context.context_id, context.context_id);
    let page = app
        .start_gallery(GalleryStartRequest {
            facet: "discover".into(),
            page_size: 8,
            filter: "vulpine".into(),
            exclude_post_ids: vec![],
        })
        .await?;
    assert_eq!(
        page.items
            .iter()
            .map(|card| card.post_id)
            .collect::<HashSet<_>>(),
        HashSet::from([2, 6])
    );
    for id in [2, 6] {
        let card = page
            .items
            .iter()
            .find(|card| card.post_id == id)
            .unwrap()
            .clone();
        app.presented(
            format!("present-{id}"),
            context.context_id.clone(),
            None,
            card.clone(),
        )
        .await?;
        let result = app.rate(format!("rate-{id}"), card, Grade::Like).await?;
        assert!(result.changed);
        assert_eq!(result.revision, 1);
    }
    let saved = UserSettings {
        left_handed: false,
        video_muted: true,
        auto_preset: "favorites".into(),
        auto_delay_seconds: 17,
        blacklist: vec!["artist_b".into()],
    };
    assert_eq!(
        app.save_settings("saved-settings".into(), saved.clone())
            .await?,
        saved
    );
    let before = app.restore_context().await?.context("Presented history")?;
    assert_eq!(
        before
            .entries
            .iter()
            .map(|entry| entry.card.post_id)
            .collect::<Vec<_>>(),
        [2, 6]
    );
    assert!(
        before
            .entries
            .iter()
            .all(|entry| entry.card.current_grade == Some(Grade::Like))
    );
    app.shutdown().await?;
    assert_eq!(std::fs::read(&config.settings_path)?, import);
    let preference_count = || -> Result<i64> {
        Ok(
            rusqlite::Connection::open(&config.interaction_path)?.query_row(
                "SELECT COUNT(*) FROM preference_facts",
                [],
                |row| row.get(0),
            )?,
        )
    };
    assert_eq!(preference_count()?, 2);

    // The next release lacks post 6, but its saved label and history remain
    // durable. An invalid import must never become a second authority.
    config.corpus_path = temporary.path().join("corpus-next");
    ff_test_support::corpus::write(&config.corpus_path, 4)?;
    let invalid_import = b"this is not valid TOML [[[";
    std::fs::write(&config.settings_path, invalid_import)?;
    let app = AppHandle::open(config.clone()).await?;
    let health = app.health().await?;
    assert_eq!(health.rating_count, 2);
    assert_eq!(health.model_status.route, model::ModelRoute::Starter);
    assert_eq!(app.settings().await?, saved);
    assert_eq!(preference_count()?, 2);
    assert_eq!(std::fs::read(&config.settings_path)?, invalid_import);
    let restored = app.restore_context().await?.context("Restored history")?;
    assert_eq!(restored.context_id, before.context_id);
    assert_eq!(restored.mode, before.mode);
    assert_eq!(restored.filter, before.filter);
    assert_eq!(restored.entries.len(), 1);
    assert_eq!(restored.entries[0].entry_id, before.entries[0].entry_id);
    assert_eq!(restored.entries[0].card.post_id, 2);
    assert_eq!(restored.entries[0].card.current_grade, Some(Grade::Like));
    assert_eq!(
        restored.cursor.as_deref(),
        Some(restored.entries[0].entry_id.as_str())
    );
    let unavailable = app
        .start_gallery(GalleryStartRequest {
            facet: "discover".into(),
            page_size: 8,
            filter: "vulpine".into(),
            exclude_post_ids: vec![],
        })
        .await?;
    assert!(unavailable.items.is_empty());
    let mut unblocked = saved;
    unblocked.blacklist.clear();
    app.save_settings("unblock-artists".into(), unblocked)
        .await?;
    let fresh = app
        .start_auto(AutoStartRequest {
            preset: "fresh".into(),
            page_size: 8,
            seed: 7,
            filter: String::new(),
            exclude_post_ids: vec![],
        })
        .await?;
    // Rating artist_a on post 2 makes its other work and coartists'
    // shared post ineligible for fresh. Only artist_b's solo work remains.
    assert_eq!(
        fresh
            .items
            .iter()
            .map(|card| card.post_id)
            .collect::<Vec<_>>(),
        [3]
    );
    app.shutdown().await?;
    let store = Store::open(&config.interaction_path)?;
    assert_eq!(
        store
            .ratings()?
            .iter()
            .map(|rating| (rating.post_id, rating.grade))
            .collect::<Vec<_>>(),
        [(2, Grade::Like), (6, Grade::Like)]
    );
    assert_eq!(store.active_context()?.unwrap().entries.len(), 2);
    Ok(())
}

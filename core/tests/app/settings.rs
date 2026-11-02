use super::*;
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn actor_failed_settings_save_keeps_live_policy_and_retries_same_id() -> Result<()> {
    for target in ["interaction_events", "preference_facts"] {
        let (_temporary, config) = support::fixture(8)?;
        let app = AppHandle::open(config.clone()).await?;
        let context = app
            .activate_context(
                "fault-context".into(),
                "discover".into(),
                "vulpine".into(),
                true,
            )
            .await?;
        let gallery = app
            .start_gallery(GalleryStartRequest {
                facet: "discover".into(),
                page_size: 8,
                filter: "vulpine".into(),
                exclude_post_ids: vec![],
            })
            .await?;
        assert_eq!(
            gallery
                .items
                .iter()
                .map(|card| card.post_id)
                .collect::<HashSet<_>>(),
            HashSet::from([2, 6])
        );
        let shown = gallery
            .items
            .iter()
            .find(|card| card.post_id == 2)
            .unwrap()
            .clone();
        app.presented("fault-presented".into(), context.context_id, None, shown)
            .await?;
        let before_context = app.restore_context().await?.context("Presented history")?;
        assert_eq!(before_context.entries.len(), 1);
        assert_eq!(before_context.entries[0].card.post_id, 2);
        assert_eq!(
            before_context.cursor.as_deref(),
            Some(before_context.entries[0].entry_id.as_str())
        );
        let original_context = serde_json::to_value(before_context)?;
        let original_settings = app.settings().await?;
        let original_health = app.health().await?;
        let failure = rusqlite::Connection::open(&config.interaction_path)?;
        type Envelope = (i64, String, u64, String, String, String);
        let journal = || -> Result<Vec<Envelope>> {
            Ok(failure.prepare("SELECT sequence,event_id,occurred_at,request_json,payload_json,result_json FROM interaction_events ORDER BY sequence")?
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get::<_, f64>(2)?.to_bits(), row.get(3)?, row.get(4)?, row.get(5)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?)
        };
        let preferences = || -> Result<Vec<(i64, i64, String)>> {
            Ok(failure.prepare("SELECT sequence,ordinal,settings_json FROM preference_facts ORDER BY sequence,ordinal")?
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?)
        };
        let before_journal = journal()?;
        let before_preferences = preferences()?;
        assert_eq!(before_preferences.len(), 1);
        failure.execute_batch(&format!("CREATE TRIGGER fail_settings BEFORE INSERT ON {target} BEGIN SELECT RAISE(ABORT,'injected settings write fault'); END;"))?;
        let speculative = UserSettings {
            left_handed: true,
            video_muted: false,
            auto_preset: "favorites".into(),
            auto_delay_seconds: 17,
            blacklist: vec!["vulpine".into()],
        };
        let rejected = app
            .save_settings("fault-settings".into(), speculative.clone())
            .await
            .expect_err("Injected preference write must fail");
        assert!(
            matches!(rejected.downcast_ref::<rusqlite::Error>(),
            Some(rusqlite::Error::SqliteFailure(code, _))
                if code.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_TRIGGER),
            "{target}: {rejected:?}"
        );
        assert_eq!(journal()?, before_journal);
        assert_eq!(preferences()?, before_preferences);
        assert_eq!(app.settings().await?, original_settings);
        assert_eq!(
            serde_json::to_value(app.restore_context().await?)?,
            original_context
        );
        let health = app.health().await?;
        assert_eq!(health.model_id, original_health.model_id);
        assert_eq!(health.rating_count, original_health.rating_count);
        assert!(!health.refreshing);
        assert!(health.refresh_error.is_none());
        // Selection itself appends an envelope. Removing its injected gate
        // must not publish the previously rejected speculative policy.
        failure.execute_batch("DROP TRIGGER fail_settings")?;
        let available = app
            .start_auto(AutoStartRequest {
                preset: "balanced".into(),
                page_size: 8,
                seed: 7,
                filter: "vulpine".into(),
                exclude_post_ids: vec![],
            })
            .await?;
        assert_eq!(
            available
                .items
                .iter()
                .map(|card| card.post_id)
                .collect::<Vec<_>>(),
            [6]
        );
        let mut committed = speculative.clone();
        committed.blacklist = vec!["fox".into()];
        assert_eq!(
            app.save_settings("fault-settings".into(), speculative)
                .await?,
            committed
        );
        assert_eq!(app.settings().await?, committed);
        let durable = Store::open(&config.interaction_path)?;
        assert_eq!(durable.preferences()?, Some(committed));
        assert_eq!(
            durable
                .active_context()?
                .unwrap()
                .entries
                .iter()
                .map(|entry| entry.card.post_id)
                .collect::<Vec<_>>(),
            [2]
        );
        drop(durable);
        let blocked_context = app
            .restore_context()
            .await?
            .context("Retained blocked history")?;
        assert!(blocked_context.entries.is_empty());
        assert!(blocked_context.cursor.is_none());
        let blocked = app
            .start_auto(AutoStartRequest {
                preset: "balanced".into(),
                page_size: 8,
                seed: 7,
                filter: "vulpine".into(),
                exclude_post_ids: vec![],
            })
            .await?;
        assert!(blocked.items.is_empty());
        assert!(blocked.exhausted);
        assert_eq!(preferences()?.len(), 2);
        assert_eq!(
            failure.query_row(
                "SELECT COUNT(*) FROM interaction_events WHERE event_id='fault-settings'",
                [],
                |row| row.get::<_, i64>(0)
            )?,
            1
        );
        assert!(!app.health().await?.refreshing);
        app.shutdown().await?;
    }
    Ok(())
}

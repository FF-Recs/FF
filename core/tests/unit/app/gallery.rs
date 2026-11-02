use super::support::*;
use super::*;

#[test]
fn gallery_empty_restart_retires_prior_context_and_rejects_its_continuation() -> Result<()> {
    let (_root, mut backend) = gallery_fixture(40)?;
    let first = gallery_select(&mut backend, None, "discover", 5)?;
    backend.rate(
        uuid::Uuid::new_v4().to_string(),
        &first.items[0],
        Grade::Love,
    )?;
    let sql = rusqlite::Connection::open(&backend.store.path)?;
    let payload: String = sql.query_row(
        "SELECT payload_json FROM offer_facts WHERE family='gallery' AND session_id=?1 AND position=0",
        [&first.session_id],
        |row| row.get(0),
    )?;
    let context = backend
        .store
        .context(&first.session_id, first.items[0].post_id)?
        .json;
    // Review has no eligible rows: the one judgment has a fresh checkpoint.
    let empty = gallery_select(&mut backend, None, "rating_review", 5)?;
    assert!(empty.items.is_empty() && empty.exhausted && empty.session_id.is_empty());
    assert_eq!(empty.facet, "rating_review");
    assert!(backend.gallery.is_none());
    assert!(
        gallery_select(
            &mut backend,
            Some(gallery_next(&first, 2, vec![])),
            "discover",
            2
        )
        .is_err()
    );
    let (status, exhausted, returned): (String, bool, i64) = sql.query_row(
        "SELECT status,(SELECT exhausted FROM selection_facts WHERE family='gallery' AND session_id=?1 AND exhausted IS NOT NULL ORDER BY sequence DESC,ordinal DESC LIMIT 1),returned_size FROM q_gallery_sessions WHERE session_id=?1",
        [&first.session_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    assert_eq!(status, "abandoned");
    assert!(
        !exhausted,
        "An empty new Review context says nothing about the old Discover universe"
    );
    assert_eq!(returned, 5);
    assert_eq!(
        sql.query_row("SELECT COUNT(*) FROM q_gallery_sessions", [], |row| row
            .get::<_, i64>(0))?,
        1
    );
    assert_eq!(
        sql.query_row(
            "SELECT payload_json FROM offer_facts WHERE family='gallery' AND session_id=?1 AND position=0",
            [&first.session_id],
            |row| row.get::<_, String>(0),
        )?,
        payload
    );
    assert_eq!(
        backend
            .store
            .context(&first.session_id, first.items[0].post_id)?
            .json,
        context
    );
    assert!(
        !backend
            .rate(
                uuid::Uuid::new_v4().to_string(),
                &first.items[0],
                Grade::Love
            )?
            .changed
    );
    Ok(())
}

#[test]
fn gallery_failed_empty_restart_retirement_keeps_original_continuation_valid() -> Result<()> {
    let (_root, mut backend) = gallery_fixture(40)?;
    let first = gallery_select(&mut backend, None, "discover", 5)?;
    let sql = rusqlite::Connection::open(&backend.store.path)?;
    let previous_time: f64 = sql.query_row(
        "SELECT updated_at FROM selection_facts WHERE family='gallery' AND session_id=?1 ORDER BY sequence DESC,ordinal DESC LIMIT 1",
        [&first.session_id],
        |row| row.get(0),
    )?;
    sql.execute_batch(&format!(
        "CREATE TRIGGER fail_empty_retirement BEFORE INSERT ON selection_facts
             WHEN NEW.family='gallery' AND NEW.operation='retire' AND NEW.session_id='{}'
             BEGIN SELECT RAISE(ABORT,'retirement failure'); END;",
        first.session_id.replace('\'', "''")
    ))?;
    assert!(gallery_select(&mut backend, None, "rating_review", 5).is_err());
    assert_eq!(backend.gallery.as_ref().unwrap().session, first.session_id);
    assert_eq!(backend.gallery.as_ref().unwrap().issued.len(), 5);
    let (status, exhausted, updated): (String, bool, f64) = sql.query_row(
        "SELECT status,(SELECT exhausted FROM selection_facts WHERE family='gallery' AND session_id=?1 AND exhausted IS NOT NULL ORDER BY sequence DESC,ordinal DESC LIMIT 1),(SELECT updated_at FROM selection_facts WHERE family='gallery' AND session_id=?1 ORDER BY sequence DESC,ordinal DESC LIMIT 1) FROM q_gallery_sessions WHERE session_id=?1",
        [&first.session_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    assert_eq!(status, "active");
    assert!(!exhausted);
    assert_eq!(updated, previous_time);
    assert_eq!(
        sql.query_row("SELECT COUNT(*) FROM q_gallery_sessions", [], |row| row
            .get::<_, i64>(0))?,
        1
    );
    assert_eq!(
        sql.query_row(
            "SELECT COUNT(*) FROM offer_facts WHERE family='gallery'",
            [],
            |row| row.get::<_, i64>(0)
        )?,
        5
    );
    sql.execute_batch("DROP TRIGGER fail_empty_retirement")?;
    let retry = gallery_select(
        &mut backend,
        Some(gallery_next(&first, 2, vec![])),
        "discover",
        2,
    )?;
    assert_eq!(retry.session_id, first.session_id);
    assert_eq!(retry.items.len(), 2);
    assert_eq!(retry.items[0].position, 5);
    Ok(())
}

#[test]
fn gallery_explicit_restarts_preserve_completed_prior_cohorts() -> Result<()> {
    for empty_restart in [false, true] {
        let (_root, mut backend) = gallery_fixture(40)?;
        let first = gallery_select(&mut backend, None, "discover", 30)?;
        let next = gallery_select(
            &mut backend,
            None,
            if empty_restart {
                "rating_review"
            } else {
                "q3_probe"
            },
            2,
        )?;
        assert_eq!(next.items.is_empty(), empty_restart);
        assert!(
            gallery_select(
                &mut backend,
                Some(gallery_next(&first, 2, vec![])),
                "discover",
                2
            )
            .is_err()
        );
        let sql = rusqlite::Connection::open(&backend.store.path)?;
        let (status, exhausted): (String, bool) = sql.query_row(
            "SELECT status,(SELECT exhausted FROM selection_facts WHERE family='gallery' AND session_id=?1 AND exhausted IS NOT NULL ORDER BY sequence DESC,ordinal DESC LIMIT 1) FROM q_gallery_sessions WHERE session_id=?1",
            [&first.session_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(status, "completed");
        assert!(!exhausted);
        assert_eq!(
            sql.query_row("SELECT COUNT(*) FROM q_gallery_sessions", [], |row| row
                .get::<_, i64>(0))?,
            if empty_restart { 1 } else { 2 }
        );
    }
    Ok(())
}

#[test]
fn gallery_fixed_inputs_reuse_exact_full_cohort_suffix_across_views() -> Result<()> {
    for facet in ["discover", "train_broadly", "q3_probe"] {
        let (_whole_root, mut whole) = gallery_fixture(40)?;
        let full = gallery_select(&mut whole, None, facet, 100)?;
        let (_split_root, mut split) = gallery_fixture(40)?;
        let first = gallery_select(&mut split, None, facet, 5)?;
        assert_eq!(first.items.len(), 5);
        assert_eq!(first.items[0].universe_size, 5);
        assert!(!first.exhausted);
        for card in &first.items {
            split.store.viewed(
                &format!("visit-{}", card.post_id),
                card.post_id,
                &card.session_id,
            )?;
        }
        let retained = first.items.iter().map(|card| card.post_id).collect();
        let next = gallery_select(
            &mut split,
            Some(gallery_next(&first, 2, retained)),
            facet,
            2,
        )?;
        assert_eq!(first.session_id, next.session_id);
        assert_eq!(next.items[0].position, 5);
        assert_eq!(next.items[0].universe_size, 7);
        let retained = first
            .items
            .iter()
            .chain(&next.items)
            .map(|card| card.post_id)
            .collect();
        let tail = gallery_select(
            &mut split,
            Some(gallery_next(&next, 100, retained)),
            facet,
            100,
        )?;
        let split_cards: Vec<_> = first
            .items
            .iter()
            .chain(&next.items)
            .chain(&tail.items)
            .map(|card| (card.post_id, card.selection_score))
            .collect();
        assert_eq!(
            split_cards,
            full.items
                .iter()
                .map(|card| (card.post_id, card.selection_score))
                .collect::<Vec<_>>()
        );
        assert!(
            !tail.exhausted,
            "A cohort boundary is not global exhaustion"
        );
    }
    Ok(())
}

#[test]
fn gallery_partial_cohort_appends_new_generation_without_rewriting_offers_or_group() -> Result<()> {
    let (_root, mut backend) = gallery_fixture(40)?;
    let first = gallery_select(&mut backend, None, "discover", 5)?;
    let sql = rusqlite::Connection::open(&backend.store.path)?;
    let old_payload: String = sql.query_row(
        "SELECT payload_json FROM offer_facts WHERE family='gallery' AND session_id=?1 AND position=0",
        [&first.session_id],
        |row| row.get(0),
    )?;
    let old_context = backend
        .store
        .context(&first.session_id, first.items[0].post_id)?
        .json;
    backend.rate(
        uuid::Uuid::new_v4().to_string(),
        &first.items[0],
        Grade::Love,
    )?;
    let old_generation = Arc::downgrade(&backend.generation);
    backend.generation = Arc::new(FitResult {
        model_id: "M1".into(),
        status: backend.generation.status.clone(),
        scores: Some(ScoreBuffers {
            not_dislike: vec![0.98; 40],
            like_or_love: vec![0.94; 40],
            love: vec![0.79; 40],
            expected_grade: vec![2.71; 40],
        }),
    });
    assert!(old_generation.upgrade().is_none());
    let next = gallery_select(
        &mut backend,
        Some(gallery_next(
            &first,
            2,
            first.items.iter().map(|card| card.post_id).collect(),
        )),
        "discover",
        2,
    )?;
    assert_eq!(next.session_id, first.session_id);
    assert_eq!(
        next.items
            .iter()
            .map(|card| card.position)
            .collect::<Vec<_>>(),
        vec![5, 6]
    );
    assert!(
        next.items
            .iter()
            .all(|card| card.model_id == "M1" && card.universe_size == 7)
    );
    assert_eq!(
        sql.query_row(
            "SELECT model_id FROM q_gallery_sessions WHERE session_id=?1",
            [&first.session_id],
            |row| row.get::<_, String>(0)
        )?,
        "M0"
    );
    assert_eq!(
        sql.query_row(
            "SELECT payload_json FROM offer_facts WHERE family='gallery' AND session_id=?1 AND position=0",
            [&first.session_id],
            |row| row.get::<_, String>(0)
        )?,
        old_payload
    );
    assert_eq!(
        backend
            .store
            .context(&first.session_id, first.items[0].post_id)?
            .json,
        old_context
    );
    assert!(
        !backend
            .rate(
                uuid::Uuid::new_v4().to_string(),
                &first.items[0],
                Grade::Love
            )?
            .changed
    );
    let training = labels(backend.store.ratings()?);
    assert_eq!(training[0].session_id, first.session_id);
    Ok(())
}

#[test]
fn gallery_short_boundary_rollover_and_true_empty_keep_distinct_meanings() -> Result<()> {
    let (_root, mut backend) = gallery_fixture(40)?;
    let first = gallery_select(&mut backend, None, "discover", 29)?;
    let ids: Vec<_> = first.items.iter().map(|card| card.post_id).collect();
    let short = gallery_select(
        &mut backend,
        Some(gallery_next(&first, 5, ids.clone())),
        "discover",
        5,
    )?;
    assert_eq!(short.items.len(), 1);
    assert_eq!(short.items[0].position, 29);
    assert_eq!(short.session_id, first.session_id);
    assert!(!short.exhausted);
    let mut held = ids;
    held.extend(short.items.iter().map(|card| card.post_id));
    let fresh = gallery_select(
        &mut backend,
        Some(gallery_next(&short, 100, held.clone())),
        "discover",
        100,
    )?;
    assert_ne!(fresh.session_id, short.session_id);
    assert_eq!(fresh.items.len(), 10);
    assert_eq!(fresh.items[0].position, 0);
    assert!(!fresh.exhausted);
    held.extend(fresh.items.iter().map(|card| card.post_id));
    let empty = gallery_select(
        &mut backend,
        Some(gallery_next(&fresh, 2, held)),
        "discover",
        2,
    )?;
    assert!(empty.exhausted && empty.items.is_empty());
    assert_eq!(empty.session_id, fresh.session_id);
    let sql = rusqlite::Connection::open(&backend.store.path)?;
    assert_eq!(
        sql.query_row("SELECT COUNT(*) FROM q_gallery_sessions", [], |row| row
            .get::<_, i64>(0))?,
        2
    );
    assert_eq!(
        sql.query_row(
            "SELECT COUNT(*) FROM offer_facts WHERE family='gallery'",
            [],
            |row| row.get::<_, i64>(0)
        )?,
        40
    );
    Ok(())
}

#[test]
fn gallery_failed_append_and_cancelled_or_failed_boundary_preserve_continuation() -> Result<()> {
    let (_root, mut backend) = gallery_fixture(40)?;
    let first = gallery_select(&mut backend, None, "discover", 5)?;
    let sql = rusqlite::Connection::open(&backend.store.path)?;
    sql.execute_batch("CREATE TRIGGER fail_gallery BEFORE INSERT ON offer_facts WHEN NEW.family='gallery' BEGIN SELECT RAISE(ABORT,'transient'); END;")?;
    assert!(
        gallery_select(
            &mut backend,
            Some(gallery_next(&first, 2, vec![])),
            "discover",
            2
        )
        .is_err()
    );
    assert!(gallery_select(&mut backend, None, "q3_probe", 2).is_err());
    assert_eq!(backend.gallery.as_ref().unwrap().session, first.session_id);
    assert_eq!(
        sql.query_row("SELECT COUNT(*) FROM q_gallery_sessions", [], |row| row
            .get::<_, i64>(0))?,
        1
    );
    assert_eq!(backend.gallery.as_ref().unwrap().issued.len(), 5);
    assert_eq!(
        sql.query_row("SELECT returned_size FROM q_gallery_sessions", [], |row| {
            row.get::<_, i64>(0)
        })?,
        5
    );
    sql.execute_batch("DROP TRIGGER fail_gallery")?;
    let (mut append_job, append_receive) = gallery_job(
        &mut backend,
        Some(gallery_next(&first, 2, vec![])),
        "discover",
        2,
    )?;
    append_job.plan(&backend.corpus, &backend.store.path)?;
    assert!(backend.revalidate(&append_job)?);
    drop(append_receive);
    assert!(append_job.task.is_closed());
    append_job
        .task
        .fail(&mut backend, anyhow!("Selection cancelled"));
    backend.planner = Some(append_job.state);
    assert_eq!(backend.gallery.as_ref().unwrap().issued.len(), 5);
    let last = gallery_select(
        &mut backend,
        Some(gallery_next(&first, 100, vec![])),
        "discover",
        100,
    )?;
    let (mut job, receive) = gallery_job(
        &mut backend,
        Some(gallery_next(&last, 2, vec![])),
        "discover",
        2,
    )?;
    job.plan(&backend.corpus, &backend.store.path)?;
    assert!(backend.revalidate(&job)?);
    drop(receive);
    assert!(job.task.is_closed());
    job.task.fail(&mut backend, anyhow!("Selection cancelled"));
    backend.planner = Some(job.state);
    assert_eq!(backend.gallery.as_ref().unwrap().session, first.session_id);
    assert_eq!(backend.gallery.as_ref().unwrap().issued.len(), 30);
    sql.execute_batch(&format!(
        "CREATE TRIGGER fail_boundary BEFORE INSERT ON selection_facts WHEN NEW.family='gallery' AND NEW.operation='retire' AND NEW.session_id='{}' BEGIN SELECT RAISE(ABORT,'transition'); END;",
        first.session_id.replace('\'', "''")
    ))?;
    assert!(
        gallery_select(
            &mut backend,
            Some(gallery_next(&last, 2, vec![])),
            "discover",
            2
        )
        .is_err()
    );
    assert_eq!(
        sql.query_row("SELECT COUNT(*) FROM q_gallery_sessions", [], |row| row
            .get::<_, i64>(0))?,
        1
    );
    assert_eq!(backend.gallery.as_ref().unwrap().session, first.session_id);
    sql.execute_batch("DROP TRIGGER fail_boundary")?;
    let retry = gallery_select(
        &mut backend,
        Some(gallery_next(&last, 2, vec![])),
        "discover",
        2,
    )?;
    assert_ne!(retry.session_id, first.session_id);
    assert_eq!(retry.items[0].position, 0);
    Ok(())
}

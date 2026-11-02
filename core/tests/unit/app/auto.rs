use super::support::*;
use super::*;

fn auto_select(
    backend: &mut Backend,
    next: Option<AutoNextRequest>,
    size: usize,
) -> Result<AutoPage> {
    let (mut job, mut receive) = auto_job(backend, next, size)?;
    let result = job.plan(&backend.corpus, &backend.store.path);
    backend.reclaim_changes(&mut job);
    let result = result.and_then(|_| {
        ensure!(backend.revalidate(&job)?, "Unexpected stale test plan");
        Ok(())
    });
    backend.planner = Some(job.state);
    match result {
        Ok(()) => backend.commit(job.task, &job.generation),
        Err(error) => job.task.fail(backend, error),
    }
    receive.try_recv().context("Auto test reply")?
}

fn continuation(page: &AutoPage, size: usize, exclude_post_ids: Vec<i64>) -> AutoNextRequest {
    AutoNextRequest {
        session_id: page.session_id.clone(),
        page_size: size,
        exclude_post_ids,
    }
}

fn publish_scores(backend: &mut Backend, probabilities: [f32; 4]) {
    backend.generation = Arc::new(FitResult {
        // Generation object identity, not this display ID, owns admission.
        model_id: backend.generation.model_id.clone(),
        status: backend.generation.status.clone(),
        scores: Some(ScoreBuffers {
            not_dislike: vec![0.99; 4],
            like_or_love: probabilities.to_vec(),
            love: vec![0.8; 4],
            expected_grade: vec![2.74; 4],
        }),
    });
}

#[test]
fn latest_generation_admits_new_rows_without_resetting_epoch_or_emission_history() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    publish_scores(&mut backend, [0.95, 0.0, 0.0, 0.0]);
    let old_generation = Arc::downgrade(&backend.generation);
    let first = auto_select(&mut backend, None, 1)?;
    assert_eq!(first.items[0].post_id, 1);
    assert!(first.exhausted);
    publish_scores(&mut backend, [0.95, 0.95, 0.0, 0.0]);
    assert!(old_generation.upgrade().is_none());
    let second = auto_select(&mut backend, Some(continuation(&first, 1, vec![])), 1)?;
    assert_eq!(second.items[0].post_id, 2);
    assert_eq!(second.session_id, first.session_id);
    assert_eq!(second.epoch, first.epoch);
    assert_eq!(second.emitted_count, 2);
    assert_eq!(second.items[0].position, 1);

    // Reordered admission cannot reinterpret old row IDs as new offsets.
    backend
        .auto
        .as_mut()
        .context("Missing Auto state")?
        .candidates
        .reverse();
    publish_scores(&mut backend, [0.0, 0.95, 0.95, 0.0]);
    let third = auto_select(&mut backend, Some(continuation(&second, 1, vec![])), 1)?;
    assert_eq!(third.items[0].post_id, 3);
    assert_eq!(third.session_id, first.session_id);
    assert_eq!(third.epoch, 0);
    assert_eq!(third.universe_size, 2);
    assert_eq!(third.emitted_count, 3);
    assert_eq!(third.items[0].position, 2);
    assert_eq!(
        backend.auto.as_ref().unwrap().emitted,
        HashSet::from([0, 1, 2])
    );
    Ok(())
}

#[test]
fn held_cards_do_not_roll_over_into_duplicate_offers_and_true_empty_keeps_history() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    let first = auto_select(&mut backend, None, 4)?;
    let held: Vec<_> = first.items.iter().map(|card| card.post_id).collect();
    assert!(first.exhausted);
    let empty = auto_select(&mut backend, Some(continuation(&first, 4, held.clone())), 4)?;
    assert!(empty.items.is_empty() && empty.exhausted);
    assert_eq!(empty.session_id, first.session_id);
    assert_eq!(empty.epoch, 0);
    assert_eq!(empty.emitted_count, 4);
    let held_subset = held[1..].to_vec();
    let replay = auto_select(&mut backend, Some(continuation(&empty, 4, held_subset)), 4)?;
    assert_eq!(replay.items.len(), 1);
    assert_eq!(replay.items[0].post_id, held[0]);
    assert_ne!(replay.session_id, first.session_id);
    assert_eq!(replay.epoch, 1);
    assert_eq!(replay.emitted_count, 1);
    assert_eq!(replay.items[0].position, 0);
    assert!(replay.exhausted);
    let persisted = rusqlite::Connection::open(&backend.store.path)?;
    let offers: i64 = persisted.query_row(
        "SELECT COUNT(*) FROM offer_facts WHERE family='auto'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(offers, 5);

    // Current eligibility, rather than an empty masked page, owns exhaustion.
    publish_scores(&mut backend, [0.0; 4]);
    let ineligible = auto_select(&mut backend, Some(continuation(&replay, 4, vec![])), 4)?;
    assert!(ineligible.items.is_empty() && ineligible.exhausted);
    assert_eq!(ineligible.session_id, replay.session_id);
    assert_eq!(ineligible.epoch, replay.epoch);
    assert_eq!(ineligible.emitted_count, 1);
    Ok(())
}

#[test]
fn retained_exclusions_do_not_poison_admission_cache() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    let first = auto_select(&mut backend, None, 1)?;
    let deferred = (1..=4).find(|id| *id != first.items[0].post_id).unwrap();
    let excluded = auto_select(
        &mut backend,
        Some(continuation(&first, 4, vec![deferred])),
        4,
    )?;
    assert_eq!(excluded.items.len(), 2);
    assert!(!excluded.items.iter().any(|card| card.post_id == deferred));
    assert!(!excluded.exhausted);
    let admitted = auto_select(&mut backend, Some(continuation(&excluded, 1, vec![])), 1)?;
    assert_eq!(admitted.items[0].post_id, deferred);
    assert_eq!(admitted.session_id, first.session_id);
    assert_eq!(admitted.epoch, 0);
    Ok(())
}

#[test]
fn grade_changes_readmit_rows_without_waiting_for_model_publication() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    let offered = refresh_offers(&mut backend)?;
    publish_scores(&mut backend, [0.95, 0.0, 0.0, 0.0]);
    let first = auto_select(&mut backend, None, 1)?;
    backend.rate(uuid::Uuid::new_v4().to_string(), &offered[1], Grade::Like)?;
    // A Gallery job can consume this delta before the next Auto request.
    let mut changes = backend.pending_grades.drain().collect();
    backend.planner.as_mut().unwrap().synchronize(
        &backend.corpus,
        &mut changes,
        &backend.store.path,
        &|| false,
    )?;
    let second = auto_select(&mut backend, Some(continuation(&first, 1, vec![])), 1)?;
    assert_eq!(second.items[0].post_id, 2);
    assert_eq!(second.session_id, first.session_id);
    assert_eq!(second.emitted_count, 2);
    Ok(())
}

#[test]
fn cancelled_completed_rollover_restores_original_continuation_and_emissions() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    let first = auto_select(&mut backend, None, 4)?;
    let (mut job, receive) = auto_job(&mut backend, Some(continuation(&first, 1, vec![])), 1)?;
    job.plan(&backend.corpus, &backend.store.path)?;
    assert!(backend.revalidate(&job)?);
    let PlanTask::Auto(selection) = &job.task else {
        return Err(anyhow!("Expected Auto task"));
    };
    assert_eq!(selection.lifecycle.proposed().unwrap().epoch, 1);
    drop(receive);
    assert!(job.task.is_closed());
    job.task.fail(&mut backend, anyhow!("Selection cancelled"));
    backend.planner = Some(job.state);
    let restored = backend.auto.as_ref().unwrap();
    assert_eq!(restored.session, first.session_id);
    assert_eq!(restored.emitted, HashSet::from([0, 1, 2, 3]));
    let retried = auto_select(
        &mut backend,
        Some(continuation(&first, 1, vec![1, 2, 3])),
        1,
    )?;
    assert_eq!(retried.items[0].post_id, 4);
    assert_eq!(retried.epoch, 1);
    assert_eq!(retried.emitted_count, 1);
    Ok(())
}

#[test]
fn failed_auto_append_does_not_advance_emissions_or_card_positions() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    let first = auto_select(&mut backend, None, 1)?;
    let failure = rusqlite::Connection::open(&backend.store.path)?;
    failure.execute_batch(&format!(
        "CREATE TRIGGER fail_append BEFORE INSERT ON offer_facts WHEN NEW.family='auto' AND NEW.session_id='{}' BEGIN SELECT RAISE(ABORT,'transient append failure'); END;",
        first.session_id.replace('\'', "''")
    ))?;
    assert!(auto_select(&mut backend, Some(continuation(&first, 1, vec![])), 1).is_err());
    assert_eq!(backend.auto.as_ref().unwrap().emitted.len(), 1);
    failure.execute_batch("DROP TRIGGER fail_append")?;
    let second = auto_select(&mut backend, Some(continuation(&first, 1, vec![])), 1)?;
    assert_eq!(second.session_id, first.session_id);
    assert_eq!(second.emitted_count, 2);
    assert_eq!(second.items[0].position, 1);
    assert_ne!(second.items[0].post_id, first.items[0].post_id);
    Ok(())
}

#[test]
fn recent_views_are_reapplied_without_removing_rows_from_admission() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    refresh_offers(&mut backend)?;
    backend.store.viewed("recent", 1, "earlier")?;
    publish_scores(&mut backend, [0.95, 0.95, 0.0, 0.0]);
    let first = auto_select(&mut backend, None, 2)?;
    assert_eq!(first.items.len(), 1);
    assert_eq!(first.items[0].post_id, 2);
    assert!(first.exhausted);
    assert!(backend.auto.as_ref().unwrap().candidates.contains(&0));
    // Batch only native boundary-minus-one preparation. The final explicit
    // view still commits through the ordinary durable Store path.
    let later: Vec<_> = (0..500)
        .map(|position| {
            let mut card = backend.card(
                &backend.generation,
                "discover",
                "later-views",
                0,
                position,
                500,
                0,
                1.0,
            );
            card.post_id = 1000 + position as i64;
            card
        })
        .collect();
    backend.store.create_gallery(
        "later-views",
        "discover",
        "",
        "test-generation",
        500,
        &later,
        false,
    )?;
    crate::store::fixtures::recent_view_seed(&mut backend.store, &later[..499])?;
    let recent = backend.store.recent_views(500)?;
    assert_eq!(recent.len(), 500);
    assert!(recent.contains(&1));
    let at_499 = auto_select(&mut backend, Some(continuation(&first, 1, vec![2])), 1)?;
    assert!(at_499.items.is_empty());
    assert_eq!(at_499.session_id, first.session_id);
    assert_eq!(at_499.epoch, 0);
    assert!(backend.auto.as_ref().unwrap().candidates.contains(&0));
    let last = &later[499];
    backend
        .store
        .viewed("later-view-final", last.post_id, &last.session_id)?;
    let recent = backend.store.recent_views(500)?;
    assert_eq!(recent.len(), 500);
    assert!(!recent.contains(&1));
    let second = auto_select(&mut backend, Some(continuation(&first, 1, vec![2])), 1)?;
    assert_eq!(second.items[0].post_id, 1);
    assert_eq!(second.session_id, first.session_id);
    assert_eq!(second.epoch, 0);
    Ok(())
}

#[test]
fn empty_auto_plan_reranks_after_publication_or_live_eligibility_change() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    let offered = refresh_offers(&mut backend)?;
    publish_scores(&mut backend, [0.0; 4]);
    let first = auto_select(&mut backend, None, 1)?;
    let (mut job, mut receive) = auto_job(&mut backend, Some(continuation(&first, 1, vec![])), 1)?;
    job.plan(&backend.corpus, &backend.store.path)?;
    assert!(backend.revalidate(&job)?);
    backend.rate(uuid::Uuid::new_v4().to_string(), &offered[1], Grade::Like)?;
    assert!(!backend.revalidate(&job)?);
    job.eligibility = backend.eligibility(&job.task)?;
    job.eligibility.changes = backend.pending_grades.drain().collect();
    job.plan(&backend.corpus, &backend.store.path)?;
    let PlanTask::Auto(selection) = &job.task else {
        return Err(anyhow!("Expected Auto plan"));
    };
    assert_eq!(selection.rows.as_ref().unwrap().rows, vec![1]);
    backend.store.viewed("during-plan", 1, "earlier")?;
    assert!(!backend.revalidate(&job)?);
    publish_scores(&mut backend, [0.95, 0.0, 0.0, 0.0]);
    Arc::get_mut(&mut backend.generation).unwrap().model_id = "latest-generation".into();
    assert!(!backend.revalidate(&job)?);
    job.generation = backend.generation.clone();
    job.eligibility = backend.eligibility(&job.task)?;
    job.plan(&backend.corpus, &backend.store.path)?;
    assert!(backend.revalidate(&job)?);
    backend.commit(job.task, &job.generation);
    let page = receive.try_recv()??;
    assert_eq!(page.session_id, first.session_id);
    assert_eq!(page.epoch, 0);
    assert_eq!(page.items[0].model_id, "latest-generation");
    backend.planner = Some(job.state);
    Ok(())
}

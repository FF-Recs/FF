use super::support::*;
use super::*;

#[tokio::test]
async fn actor_exact_rating_retry_after_completed_fit_does_not_request_refresh() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    let offered = refresh_offers(&mut backend)?;
    let path = backend.store.path.clone();
    // Paused Tokio time can expire before a native worker is scheduled.
    let release = block_compute(&backend.pool)
        .await
        .context("Native compute gate startup")?;
    let mut workers = launch_observed(backend)?;
    let app = workers.app.clone();
    let result = app
        .rate("actor-vote".into(), offered[0].clone(), Grade::Like)
        .await?;
    assert!(result.changed);
    assert_eq!(result.revision, 1);
    let start = workers.next().await.context("Actor debounce demand")?;
    assert!(matches!(start, Command::StartFit(1)));
    app.sender.send(start).await?;
    assert!(app.health().await?.refreshing);
    release.send(())?;
    let completion = workers.next().await.context("Native fit completion")?;
    assert!(matches!(&completion, Command::Fitted(1, Ok(Some(_)))));
    app.sender.send(completion).await?;
    // This reply is a barrier after publication of the actual worker result.
    let complete = app.health().await?;
    assert!(!complete.refreshing);
    assert_eq!(complete.rating_count, 1);
    tokio::time::pause();

    let retry = app
        .rate("actor-vote".into(), offered[0].clone(), Grade::Like)
        .await?;
    assert!(!retry.changed);
    assert_eq!(retry.revision, result.revision);
    let after = app.health().await?;
    assert!(!after.refreshing);
    assert_eq!(after.model_id, complete.model_id);
    assert_eq!(after.rating_count, 1);
    tokio::task::yield_now().await;
    tokio::time::advance(std::time::Duration::from_secs(1)).await;
    tokio::task::yield_now().await;
    assert!(!app.health().await?.refreshing);
    assert!(matches!(
        workers.receiver.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert_eq!(Store::snapshot_ratings_at(&path)?.len(), 1);
    app.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_snapshot_including_newer_demand_publishes_then_reuses_the_same_generation()
-> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    let offered = refresh_offers(&mut backend)?;
    let path = backend.store.path.clone();
    let release = block_compute(&backend.pool).await?;
    let mut workers = launch_observed(backend)?;
    let app = workers.app.clone();
    app.rate(
        uuid::Uuid::new_v4().to_string(),
        offered[0].clone(),
        Grade::Like,
    )
    .await?;
    let command = workers.next().await?;
    assert!(matches!(command, Command::StartFit(1)));
    app.sender.send(command).await?;
    assert!(app.health().await?.refreshing);
    app.sender
        .send(Command::Fitted(0, Err(anyhow!("stale completion"))))
        .await?;
    assert!(app.health().await?.refresh_error.is_none());
    // The admitted worker has not started reading yet. This second demand
    // will be in its actual snapshot, but is not covered by request ID 1.
    app.rate(
        uuid::Uuid::new_v4().to_string(),
        offered[1].clone(),
        Grade::Dislike,
    )
    .await?;
    assert_eq!(Store::snapshot_ratings_at(&path)?.len(), 2);
    app.sender.send(Command::StartFit(2)).await?;
    assert_eq!(app.health().await?.model_id, "test-generation");
    release.send(())?;
    let Command::Fitted(request, result) = workers.next().await? else {
        return Err(anyhow!("Expected first worker completion"));
    };
    assert_eq!(request, 1);
    let generation = result?.context("Changed input must fit")?;
    let serving = Arc::downgrade(&generation);
    assert!(matches!(
        generation.status.fallback_reason,
        Some(model::SupportReason::InsufficientLabels { actual: 2, .. })
    ));
    let model_id = generation.model_id.clone();
    let status = generation.status.clone();
    app.sender
        .send(Command::Fitted(request, Ok(Some(generation))))
        .await?;
    let health = app.health().await?;
    assert_eq!(health.model_id, model_id);
    assert_eq!(health.model_status, status);
    assert!(health.refreshing);
    assert!(health.refresh_error.is_none());

    // The later request still takes a snapshot, but identical prepared inputs
    // skip both fitting and scoring against the successful serving identity.
    let command = workers.next().await?;
    assert!(matches!(command, Command::StartFit(2)));
    app.sender.send(command).await?;
    let command = workers.next().await?;
    assert!(matches!(command, Command::Fitted(2, Ok(None))));
    app.sender.send(command).await?;
    let health = app.health().await?;
    assert!(!health.refreshing);
    assert_eq!(health.model_id, model_id);
    assert_eq!(health.model_status, status);
    assert!(
        serving.upgrade().is_some(),
        "Reuse must retain the original generation Arc"
    );
    app.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn demand_after_worker_snapshot_is_not_swallowed_by_prefit_reuse() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    let offered = refresh_offers(&mut backend)?;
    let mut workers = launch_observed(backend)?;
    let app = workers.app.clone();
    app.rate(
        uuid::Uuid::new_v4().to_string(),
        offered[0].clone(),
        Grade::Like,
    )
    .await?;
    let command = workers.next().await?;
    assert!(matches!(command, Command::StartFit(1)));
    app.sender.send(command).await?;
    // Holding this real completion proves the snapshot has already closed.
    let command = workers.next().await?;
    let Command::Fitted(1, Ok(Some(generation))) = &command else {
        return Err(anyhow!("Expected a genuinely fitted first generation"));
    };
    let intermediate_id = generation.model_id.clone();
    app.rate(
        uuid::Uuid::new_v4().to_string(),
        offered[1].clone(),
        Grade::Love,
    )
    .await?;
    app.sender.send(command).await?;
    let intermediate = app.health().await?;
    assert_eq!(intermediate.model_id, intermediate_id);
    assert!(intermediate.refreshing);
    let command = workers.next().await?;
    assert!(matches!(command, Command::StartFit(2)));
    app.sender.send(command).await?;
    let command = workers.next().await?;
    let Command::Fitted(2, Ok(Some(generation))) = &command else {
        return Err(anyhow!("Later changed prepared inputs must not be skipped"));
    };
    assert_ne!(generation.model_id, intermediate_id);
    assert!(matches!(
        generation.status.fallback_reason,
        Some(model::SupportReason::InsufficientLabels { actual: 2, .. })
    ));
    app.sender.send(command).await?;
    let complete = app.health().await?;
    assert!(!complete.refreshing);
    assert!(complete.refresh_error.is_none());
    assert_ne!(complete.model_id, intermediate_id);
    app.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_worker_snapshot_retains_generation_reports_error_and_releases_reads() -> Result<()>
{
    let (_temporary, mut backend) = fixture()?;
    let path = backend.store.path.clone();
    let offered = refresh_offers(&mut backend)?;
    backend.rate(uuid::Uuid::new_v4().to_string(), &offered[0], Grade::Like)?;
    let connection = rusqlite::Connection::open(&path)?;
    let canonical_ratings: String = connection.query_row(
        "SELECT sql FROM sqlite_schema WHERE type='view' AND name='q_ratings'",
        [],
        |row| row.get(0),
    )?;
    // Inject a bad readout, not a forbidden mutation of committed history.
    connection.execute_batch(
        "DROP VIEW q_ratings;
             CREATE VIEW q_ratings AS
             SELECT current.post_id,
                    CASE WHEN current.post_id=1 THEN 'invalid' ELSE current.action END AS action,
                    current.first_rated_at,current.updated_at,current.revision,current.context_json
             FROM rating_facts AS current
             JOIN (SELECT post_id,MAX(sequence) AS sequence FROM rating_facts GROUP BY post_id)
             AS latest USING(post_id,sequence);",
    )?;
    drop(connection);
    let mut workers = launch_observed(backend)?;
    let app = workers.app.clone();
    app.rate(
        uuid::Uuid::new_v4().to_string(),
        offered[2].clone(),
        Grade::Like,
    )
    .await?;
    let command = workers.next().await?;
    assert!(matches!(command, Command::StartFit(1)));
    app.sender.send(command).await?;
    let command = workers.next().await?;
    assert!(matches!(command, Command::Fitted(1, Err(_))));
    app.sender.send(command).await?;
    let health = app.health().await?;
    assert_eq!(health.model_id, "test-generation");
    assert_eq!(health.rating_count, 2);
    assert!(health.refresh_error.is_some());
    assert!(!health.refreshing);
    // Restore the canonical readout and force a checkpoint: an abandoned
    // failed read must not retain a WAL transaction or unusable connection.
    let recovery = rusqlite::Connection::open(&path)?;
    recovery.execute_batch("DROP VIEW q_ratings")?;
    recovery.execute_batch(&canonical_ratings)?;
    let busy: i64 = recovery.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(0))?;
    assert_eq!(busy, 0);
    assert_eq!(Store::snapshot_ratings_at(&path)?.len(), 2);
    // A delayed notification for the failed request cannot retry it, even
    // now that a snapshot would succeed. The prior serving/error stay put.
    app.sender.send(Command::StartFit(1)).await?;
    let after = app.health().await?;
    assert!(!after.refreshing);
    assert_eq!(after.model_id, health.model_id);
    assert_eq!(after.model_status, health.model_status);
    assert_eq!(after.refresh_error, health.refresh_error);
    assert!(matches!(
        workers.receiver.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    app.rate(
        uuid::Uuid::new_v4().to_string(),
        offered[3].clone(),
        Grade::Love,
    )
    .await?;
    let pending = app.health().await?;
    assert!(pending.refreshing);
    assert_eq!(pending.model_id, health.model_id);
    assert_eq!(pending.refresh_error, health.refresh_error);
    let command = workers.next().await?;
    assert!(matches!(command, Command::StartFit(2)));
    app.sender.send(command).await?;
    let command = workers.next().await?;
    assert!(matches!(command, Command::Fitted(2, Ok(Some(_)))));
    app.sender.send(command).await?;
    let recovered = app.health().await?;
    assert!(!recovered.refreshing);
    assert!(recovered.refresh_error.is_none());
    assert_eq!(recovered.rating_count, 3);
    assert_ne!(recovered.model_id, health.model_id);
    app.shutdown().await?;
    Ok(())
}

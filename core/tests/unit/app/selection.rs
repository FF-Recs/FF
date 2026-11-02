use super::support::*;
use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_auto_rollover_preserves_the_callers_session_for_retry() -> Result<()> {
    let (_temporary, backend) = fixture()?;
    let path = backend.store.path.clone();
    let app = launch(backend)?;
    let first = app
        .start_auto(AutoStartRequest {
            preset: "balanced".into(),
            page_size: 4,
            seed: 7,
            filter: String::new(),
            exclude_post_ids: vec![],
        })
        .await?;
    let failure = rusqlite::Connection::open(path)?;
    failure.execute_batch("CREATE TRIGGER fail_rollover BEFORE INSERT ON selection_facts WHEN NEW.family='auto' AND NEW.operation='create' BEGIN SELECT RAISE(ABORT,'transient write failure'); END;")?;
    assert!(
        app.next_auto(AutoNextRequest {
            session_id: first.session_id.clone(),
            page_size: 4,
            exclude_post_ids: vec![],
        })
        .await
        .is_err()
    );
    failure.execute_batch("DROP TRIGGER fail_rollover")?;
    let retried = app
        .next_auto(AutoNextRequest {
            session_id: first.session_id.clone(),
            page_size: 4,
            exclude_post_ids: vec![],
        })
        .await?;
    assert_ne!(retried.session_id, first.session_id);
    assert_eq!(retried.epoch, 1);
    assert_eq!(retried.items.len(), 4);
    assert!(
        retried
            .items
            .iter()
            .all(|card| card.session_id == retried.session_id)
    );
    app.shutdown().await?;
    Ok(())
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pending_selection_allows_durable_commands_and_revalidates_live_eligibility() -> Result<()>
{
    let (_temporary, mut backend) = fixture()?;
    let path = backend.store.path.clone();
    let offered: Vec<_> = (0..4)
        .map(|row| {
            backend.card(
                &backend.generation,
                "discover",
                "earlier",
                0,
                row,
                4,
                row,
                2.74,
            )
        })
        .collect();
    backend.store.create_gallery(
        "earlier",
        "discover",
        "",
        "test-generation",
        4,
        &offered,
        false,
    )?;
    // This fixture supplies the completion of an explicitly admitted fit.
    backend.refresh.in_flight = Some(0);
    let (release, wait) = std::sync::mpsc::sync_channel(1);
    backend.pool.spawn(move || {
        let _ = wait.recv();
    });
    let app = launch(backend)?;
    let permit = app.planning.clone().acquire_owned().await?;
    let (reply, result) = oneshot::channel();
    app.sender
        .send(Command::Plan(
            PlanRequest::Gallery(
                GalleryStartRequest {
                    facet: "discover".into(),
                    page_size: 3,
                    filter: String::new(),
                    exclude_post_ids: vec![],
                },
                reply,
            ),
            permit,
        ))
        .await
        .map_err(|_| anyhow!("actor stopped"))?;
    // FIFO puts the plan on the blocked compute worker before this health request.
    let health = tokio::time::timeout(std::time::Duration::from_secs(2), app.health()).await??;
    assert_eq!(health.rating_count, 0);
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        app.rate(
            uuid::Uuid::new_v4().to_string(),
            offered[0].clone(),
            Grade::Dislike,
        ),
    )
    .await??;
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        app.viewed("visit".into(), offered[2].clone()),
    )
    .await??;
    assert_eq!(app.health().await?.rating_count, 1);
    app.sender
        .send(Command::Fitted(
            0,
            Ok(Some(Arc::new(FitResult {
                model_id: "new-generation".into(),
                status: model::ModelStatus {
                    route: model::ModelRoute::Ensemble,
                    fallback_reason: None,
                },
                scores: Some(ScoreBuffers {
                    not_dislike: vec![0.99; 4],
                    like_or_love: vec![0.95; 4],
                    love: vec![0.45; 4],
                    expected_grade: vec![2.39; 4],
                }),
            }))),
        ))
        .await
        .map_err(|_| anyhow!("actor stopped"))?;
    assert_eq!(app.health().await?.model_id, "new-generation");
    assert_eq!(
        app.health().await?.model_status.route,
        model::ModelRoute::Ensemble
    );
    release.send(())?;
    let page = tokio::time::timeout(std::time::Duration::from_secs(5), result).await???;
    assert_eq!(
        page.items
            .iter()
            .map(|card| card.post_id)
            .collect::<Vec<_>>(),
        vec![4]
    );
    assert_eq!(page.items[0].model_id, "new-generation");
    assert_eq!(page.items[0].selection_score, 2.39);
    app.rate(
        uuid::Uuid::new_v4().to_string(),
        page.items[0].clone(),
        Grade::Like,
    )
    .await?;
    app.shutdown().await?;
    let store = Store::open(path)?;
    let rating = store
        .ratings()?
        .into_iter()
        .find(|rating| rating.post_id == 4)
        .context("missing new rating")?;
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&rating.context_json)?["model_id"],
        "new-generation"
    );
    assert_eq!(store.checkpoints()?[&4].model_id, "new-generation");
    Ok(())
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_auto_request_does_not_consume_row_emissions() -> Result<()> {
    let (_temporary, backend) = fixture()?;
    let pool = backend.pool.clone();
    let app = launch(backend)?;
    let first = app
        .start_auto(AutoStartRequest {
            preset: "balanced".into(),
            page_size: 1,
            seed: 7,
            filter: String::new(),
            exclude_post_ids: vec![],
        })
        .await?;
    let (release, wait) = std::sync::mpsc::sync_channel(1);
    pool.spawn(move || {
        let _ = wait.recv();
    });
    let permit = app.planning.clone().acquire_owned().await?;
    let (reply, result) = oneshot::channel();
    app.sender
        .send(Command::Plan(
            PlanRequest::NextAuto(
                AutoNextRequest {
                    session_id: first.session_id.clone(),
                    page_size: 1,
                    exclude_post_ids: vec![],
                },
                reply,
            ),
            permit,
        ))
        .await
        .map_err(|_| anyhow!("actor stopped"))?;
    app.health().await?;
    drop(result);
    release.send(())?;
    let second = app
        .next_auto(AutoNextRequest {
            session_id: first.session_id,
            page_size: 1,
            exclude_post_ids: vec![],
        })
        .await?;
    assert_eq!(second.emitted_count, 2);
    assert_eq!(second.items[0].position, 1);
    assert_ne!(second.items[0].post_id, first.items[0].post_id);
    app.shutdown().await?;
    assert!(app.health().await.is_err());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fresh_plan_rechecks_coartist_exposure_and_recent_views() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    // Row 1's second credit is the artist rated while planning is blocked.
    // Its first credit remains unseen; a low average must not admit it.
    backend.corpus = Arc::new(Corpus::fixture(
        &[1, 2, 3, 4],
        &[vec![1], vec![0, 1], vec![0], vec![2]],
        3,
        &[0, 1, 2],
        &[],
    ));
    backend.planner.as_mut().unwrap().familiarity = Familiarity::build(&backend.corpus, &[]);
    let offered: Vec<_> = (0..4)
        .map(|row| {
            backend.card(
                &backend.generation,
                "discover",
                "earlier",
                0,
                row,
                4,
                row,
                2.74,
            )
        })
        .collect();
    backend.store.create_gallery(
        "earlier",
        "discover",
        "",
        "test-generation",
        4,
        &offered,
        false,
    )?;
    let (release, wait) = std::sync::mpsc::sync_channel(1);
    backend.pool.spawn(move || {
        let _ = wait.recv();
    });
    let app = launch(backend)?;
    let permit = app.planning.clone().acquire_owned().await?;
    let (reply, result) = oneshot::channel();
    app.sender
        .send(Command::Plan(
            PlanRequest::StartAuto(
                AutoStartRequest {
                    preset: "fresh".into(),
                    page_size: 4,
                    seed: 7,
                    filter: String::new(),
                    exclude_post_ids: vec![],
                },
                reply,
            ),
            permit,
        ))
        .await
        .map_err(|_| anyhow!("actor stopped"))?;
    app.health().await?;
    app.rate(
        uuid::Uuid::new_v4().to_string(),
        offered[0].clone(),
        Grade::Like,
    )
    .await?;
    app.viewed("visit".into(), offered[2].clone()).await?;
    release.send(())?;
    let page = tokio::time::timeout(std::time::Duration::from_secs(5), result).await???;
    assert_eq!(
        page.items
            .iter()
            .map(|card| card.post_id)
            .collect::<Vec<_>>(),
        vec![4]
    );
    assert_eq!(page.emitted_count, 1);
    assert_eq!(page.items[0].position, 0);
    app.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn review_plan_rechecks_permanent_exclusions_and_checkpoint_cooldown() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    seed_review_history(&mut backend)?;
    let offered: Vec<_> = (0..4)
        .map(|row| {
            backend.card(
                &backend.generation,
                "discover",
                "earlier",
                0,
                row,
                4,
                row,
                2.74,
            )
        })
        .collect();
    backend.store.create_gallery(
        "earlier",
        "discover",
        "",
        "test-generation",
        4,
        &offered,
        false,
    )?;
    let review: Vec<_> = (0..4)
        .map(|row| {
            backend.card(
                &backend.generation,
                "rating_review",
                "previous-review",
                0,
                row,
                4,
                row,
                2.74,
            )
        })
        .collect();
    backend.store.create_gallery(
        "previous-review",
        "rating_review",
        "",
        "test-generation",
        4,
        &review,
        false,
    )?;
    let (release, wait) = std::sync::mpsc::sync_channel(1);
    backend.pool.spawn(move || {
        let _ = wait.recv();
    });
    let app = launch(backend)?;
    let permit = app.planning.clone().acquire_owned().await?;
    let (reply, result) = oneshot::channel();
    app.sender
        .send(Command::Plan(
            PlanRequest::Gallery(
                GalleryStartRequest {
                    facet: "rating_review".into(),
                    page_size: 4,
                    filter: String::new(),
                    exclude_post_ids: vec![],
                },
                reply,
            ),
            permit,
        ))
        .await
        .map_err(|_| anyhow!("actor stopped"))?;
    app.health().await?;
    app.rate(
        uuid::Uuid::new_v4().to_string(),
        review[0].clone(),
        Grade::Dislike,
    )
    .await?;
    app.rate(
        uuid::Uuid::new_v4().to_string(),
        review[1].clone(),
        Grade::Love,
    )
    .await?;
    release.send(())?;
    let page = tokio::time::timeout(std::time::Duration::from_secs(5), result).await???;
    let ids: HashSet<_> = page.items.iter().map(|card| card.post_id).collect();
    assert_eq!(ids, HashSet::from([3, 4]));
    app.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn published_generation_reranks_auto_and_rechecks_probability_gates() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    backend.refresh.in_flight = Some(0);
    let (release, wait) = std::sync::mpsc::sync_channel(1);
    backend.pool.spawn(move || {
        let _ = wait.recv();
    });
    let app = launch(backend)?;
    let permit = app.planning.clone().acquire_owned().await?;
    let (reply, result) = oneshot::channel();
    app.sender
        .send(Command::Plan(
            PlanRequest::StartAuto(
                AutoStartRequest {
                    preset: "balanced".into(),
                    page_size: 4,
                    seed: 7,
                    filter: String::new(),
                    exclude_post_ids: vec![],
                },
                reply,
            ),
            permit,
        ))
        .await
        .map_err(|_| anyhow!("actor stopped"))?;
    app.health().await?;
    app.sender
        .send(Command::Fitted(
            0,
            Ok(Some(Arc::new(FitResult {
                model_id: "new-generation".into(),
                status: model::ModelStatus {
                    route: model::ModelRoute::Ensemble,
                    fallback_reason: None,
                },
                scores: Some(ScoreBuffers {
                    not_dislike: vec![0.99; 4],
                    like_or_love: vec![0.1, 0.95, 0.95, 0.95],
                    love: vec![0.05, 0.45, 0.45, 0.45],
                    expected_grade: vec![1.14, 2.39, 2.39, 2.39],
                }),
            }))),
        ))
        .await
        .map_err(|_| anyhow!("actor stopped"))?;
    assert_eq!(app.health().await?.model_id, "new-generation");
    release.send(())?;
    let page = tokio::time::timeout(std::time::Duration::from_secs(5), result).await???;
    assert_eq!(page.items.len(), 3);
    assert!(
        page.items
            .iter()
            .all(|card| card.post_id != 1 && card.model_id == "new-generation")
    );
    assert_eq!(page.universe_size, 3);
    assert_eq!(page.emitted_count, 3);
    assert_eq!(
        page.items
            .iter()
            .map(|card| card.position)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    app.shutdown().await?;
    Ok(())
}

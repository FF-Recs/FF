use super::support::*;
use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_cancels_unadmitted_refresh_without_waiting_for_compute() -> Result<()> {
    let (_temporary, mut backend) = fixture()?;
    let offered = refresh_offers(&mut backend)?;
    let path = backend.store.path.clone();
    let release = block_compute(&backend.pool).await?;
    let workers = launch_observed(backend)?;
    workers
        .app
        .rate(
            uuid::Uuid::new_v4().to_string(),
            offered[0].clone(),
            Grade::Like,
        )
        .await?;
    assert!(workers.app.health().await?.refreshing);
    tokio::time::timeout(std::time::Duration::from_secs(5), workers.app.shutdown()).await??;
    assert!(workers.app.sender.send(Command::StartFit(1)).await.is_err());
    assert_eq!(Store::open(&path)?.ratings()?.len(), 1);
    release.send(())?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_waits_for_admitted_fit_and_does_not_start_pending_catchup() -> Result<()> {
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
    app.health().await?;
    app.rate(
        uuid::Uuid::new_v4().to_string(),
        offered[1].clone(),
        Grade::Dislike,
    )
    .await?;
    let (reply, mut stopped) = oneshot::channel();
    app.sender.send(Command::Shutdown(reply)).await?;
    assert!(app.health().await.is_err());
    assert!(matches!(
        stopped.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    release.send(())?;
    let command = workers.next().await?;
    assert!(matches!(command, Command::Fitted(1, Ok(_))));
    assert!(matches!(
        stopped.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    app.sender.send(command).await?;
    tokio::time::timeout(std::time::Duration::from_secs(5), stopped).await???;
    assert!(matches!(
        workers.receiver.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert_eq!(Store::open(&path)?.ratings()?.len(), 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_cancels_pending_selection_and_waits_for_its_completion() -> Result<()> {
    let (_temporary, backend) = fixture()?;
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
                    page_size: 1,
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
    let (reply, mut stopped) = oneshot::channel();
    app.sender
        .send(Command::Shutdown(reply))
        .await
        .map_err(|_| anyhow!("actor stopped"))?;
    assert!(app.health().await.is_err());
    assert!(matches!(
        stopped.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    release.send(())?;
    assert!(result.await?.is_err());
    tokio::time::timeout(std::time::Duration::from_secs(5), stopped).await???;
    assert!(
        app.start_auto(AutoStartRequest {
            preset: "balanced".into(),
            page_size: 1,
            seed: 7,
            filter: String::new(),
            exclude_post_ids: vec![],
        })
        .await
        .is_err()
    );
    Ok(())
}

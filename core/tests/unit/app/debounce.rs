use super::*;

#[tokio::test(start_paused = true)]
async fn trailing_debounce_coalesces_edits_and_rejects_already_queued_stale_demand() -> Result<()> {
    let runtime = tokio::runtime::Handle::current();
    let (sender, mut receiver) = mpsc::channel(8);
    let mut refresh = Refresh::default();
    refresh.request(&runtime, &sender.downgrade());
    tokio::task::yield_now().await;
    tokio::time::advance(std::time::Duration::from_millis(199)).await;
    refresh.request(&runtime, &sender.downgrade());
    tokio::task::yield_now().await;
    tokio::time::advance(std::time::Duration::from_millis(199)).await;
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    assert!(matches!(receiver.recv().await, Some(Command::StartFit(2))));

    // A notification that already escaped its timer is still stale after
    // a newer edit; rejecting it must not cancel the newer trailing timer.
    refresh.request(&runtime, &sender.downgrade());
    assert!(!refresh.admit(2));
    tokio::task::yield_now().await;
    tokio::time::advance(std::time::Duration::from_millis(199)).await;
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    assert!(matches!(receiver.recv().await, Some(Command::StartFit(3))));
    assert!(refresh.admit(3));
    assert!(!refresh.admit(3));
    assert!(refresh.complete(3));
    assert!(!refresh.refreshing());
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn admitted_work_keeps_new_demand_and_delays_catchup_from_completion() -> Result<()> {
    let runtime = tokio::runtime::Handle::current();
    let (sender, mut receiver) = mpsc::channel(8);
    let mut refresh = Refresh::default();
    refresh.request(&runtime, &sender.downgrade());
    assert!(refresh.admit(1));
    refresh.request(&runtime, &sender.downgrade());
    refresh.request(&runtime, &sender.downgrade());
    tokio::time::advance(std::time::Duration::from_millis(200)).await;
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert!(!refresh.admit(3));
    assert!(!refresh.complete(0));
    assert_eq!(refresh.in_flight, Some(1));
    assert!(refresh.complete(1));
    assert_eq!(refresh.attempted_through, 1);
    assert!(refresh.refreshing());

    refresh.schedule(&runtime, &sender.downgrade());
    tokio::task::yield_now().await;
    tokio::time::advance(std::time::Duration::from_millis(199)).await;
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    assert!(matches!(receiver.recv().await, Some(Command::StartFit(3))));
    assert!(refresh.admit(3));
    assert!(!refresh.complete(1));
    assert_eq!(refresh.in_flight, Some(3));
    assert!(refresh.complete(3));
    assert!(!refresh.refreshing());
    // A completed attempt, successful or failed, cannot retry itself.
    refresh.schedule(&runtime, &sender.downgrade());
    tokio::time::advance(std::time::Duration::from_millis(200)).await;
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert!(!refresh.admit(3));
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn cancelling_pending_debounce_prevents_unadmitted_work() -> Result<()> {
    let runtime = tokio::runtime::Handle::current();
    let (sender, mut receiver) = mpsc::channel(8);
    let mut refresh = Refresh::default();
    refresh.request(&runtime, &sender.downgrade());
    tokio::task::yield_now().await;
    refresh.cancel_debounce();
    tokio::time::advance(std::time::Duration::from_millis(200)).await;
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert!(refresh.in_flight.is_none());
    Ok(())
}

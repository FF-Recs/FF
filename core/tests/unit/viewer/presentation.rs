use super::support::*;
use super::*;
#[test]
fn view_retry_reuses_uuid_but_revisit_and_stale_replies_do_not() {
    let mut state = state();
    state.media_ready = true;
    let (visit, event, card) = state.start_view().unwrap();
    let attempt = state.view_attempt;
    assert_eq!(card.session_id, "original");
    state.view(
        visit,
        attempt,
        &event,
        Err(anyhow::anyhow!("temporary")),
        Instant::now(),
    );
    let (_, retry, _) = state.start_view().unwrap();
    let retry_attempt = state.view_attempt;
    assert_ne!(attempt, retry_attempt);
    assert_eq!(event, retry);
    state.view(
        visit.wrapping_sub(1),
        retry_attempt,
        &event,
        Ok(()),
        Instant::now(),
    );
    assert!(!state.viewed());
    state.view(visit, retry_attempt, &event, Ok(()), Instant::now());
    assert!(state.viewed());
    assert!(state.navigate(1));
    assert!(state.navigate(-1));
    state.media_ready = true;
    let (next_visit, next_event, _) = state.start_view().unwrap();
    assert_ne!(visit, next_visit);
    assert_ne!(event, next_event);
    state.view(visit, retry_attempt, &event, Ok(()), Instant::now());
    assert!(!state.viewed());
}
#[test]
fn presentation_alone_promotes_lookahead_to_durable_entry() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    let mut empty = context("missing");
    empty.entries.clear();
    empty.cursor = None;
    assert!(state.restore(generation, empty));
    assert!(state.selection(
        generation,
        Ok(page("new", vec![card(1, "new"), card(2, "new")], false))
    ));
    assert!(state.current_entry_id().is_none());
    state.media_ready = true;
    let (visit, event, _) = state.start_view().unwrap();
    let attempt = state.view_attempt;
    state.view(
        visit,
        attempt,
        &event,
        Err(anyhow::anyhow!("disk")),
        Instant::now(),
    );
    assert!(state.current_entry_id().is_none());
    let (_, retry, _) = state.start_view().unwrap();
    assert_eq!(event, retry);
    let retry_attempt = state.view_attempt;
    state.view(visit, retry_attempt, &event, Ok(()), Instant::now());
    assert_eq!(state.current_entry_id(), Some(event.as_str()));
    assert_eq!(state.items[1].entry_id, None);
}
#[test]
fn failed_unseen_media_does_not_become_back_history() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    let mut empty = context("");
    empty.entries.clear();
    empty.cursor = None;
    state.restore(generation, empty);
    state.selection(
        generation,
        Ok(page("new", vec![card(1, "new"), card(2, "new")], true)),
    );
    assert!(state.navigate(1));
    assert_eq!(state.current().unwrap().post_id, 2);
    assert_eq!(state.items.len(), 1);
    assert!(!state.navigate(-1));
}
#[test]
fn stale_same_event_presentation_reply_cannot_acknowledge_or_fail_a_retry() {
    let mut state = state();
    assert!(state.media_ready(state.media_generation, Instant::now()));
    let (visit, event, _) = state.start_view().unwrap();
    let first_attempt = state.view_attempt;
    assert!(state.start_view().is_none());
    assert_eq!(state.view_attempt, first_attempt);
    assert_eq!(
        state.request_navigation(NavigationIntent::Next),
        Admission::Deferred
    );
    state.view(
        visit,
        first_attempt,
        &event,
        Err(anyhow::anyhow!("first presentation failed")),
        Instant::now(),
    );
    let (retry_visit, retry_event, _) = state.start_view().unwrap();
    let retry_attempt = state.view_attempt;
    assert_eq!(retry_visit, visit);
    assert_eq!(retry_event, event, "the durable event remains retry-stable");
    assert_ne!(retry_attempt, first_attempt);
    state.view(visit, first_attempt, &event, Ok(()), Instant::now());
    assert!(state.viewing());
    assert!(!state.viewed());
    assert!(state.current_entry_id().is_none());
    assert!(state.take_navigation().is_none());
    state.view(
        visit,
        first_attempt,
        &event,
        Err(anyhow::anyhow!("duplicate old failure")),
        Instant::now(),
    );
    assert!(state.viewing());
    assert!(state.view_error().is_empty());
    state.view(visit, retry_attempt, &event, Ok(()), Instant::now());
    assert!(state.viewed());
    assert!(!state.viewing());
    assert_eq!(state.current_entry_id(), Some(event.as_str()));
    state.view(
        visit,
        retry_attempt,
        &event,
        Err(anyhow::anyhow!("duplicate failure after success")),
        Instant::now(),
    );
    assert!(state.viewed());
    assert!(state.view_error().is_empty());
    assert_eq!(state.take_navigation(), Some(NavigationIntent::Next));
    assert!(state.navigate(1));
    assert!(state.take_navigation().is_none());
}

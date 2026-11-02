use super::support::*;
use super::*;
#[test]
fn context_restore_keeps_cursor_forward_history_and_offer_identity() {
    let mut state = ViewerState::default();
    state.advance = true;
    let generation = state.begin_context();
    assert!(state.restore(generation, context("entry-70")));
    assert_eq!(state.current_entry_id(), Some("entry-70"));
    assert_eq!(state.current().unwrap().session_id, "original-offer");
    assert_eq!(
        (state.index, state.furthest, state.items.len()),
        (70, 149, 150)
    );
    assert!(!state.advance);
    assert!(state.selection(
        generation,
        Ok(page("new-model-cohort", vec![card(150, "new")], false))
    ));
    assert_eq!(state.current_entry_id(), Some("entry-70"));
    assert_eq!(state.items[150].entry_id, None);
    present(&mut state);
    let offered = state.current().unwrap().clone();
    assert!(state.start_vote(offered.clone(), Grade::Love));
    let attempt = state.vote_attempt;
    state.vote(generation, state.visit, attempt, &offered, persisted(70));
    assert!(state.take_navigation().is_none());
    assert_eq!(state.items.len(), 151);
    assert_eq!(state.items[70].card.current_grade, Some(Grade::Love));
    assert!(state.navigate(1));
    assert_eq!(state.current_entry_id(), Some("entry-71"));
    assert_eq!(state.items[150].card.session_id, "new");
}
#[test]
fn policy_and_context_boundaries_reject_old_reads_and_frames_without_erasing_history() {
    let mut state = ViewerState::default();
    let old = state.begin_context();
    assert!(state.restore(old, context("entry-2")));
    let frame = state.media_generation;
    state.policy_pending = true;
    state.suspend();
    assert_eq!(state.items.len(), 150);
    assert!(!state.navigate(1));
    assert!(!state.media_ready(frame, Instant::now()));
    assert!(!state.media_ready(state.media_generation, Instant::now()));
    assert!(!state.selection(old, Ok(page("stale", vec![], true))));
    let generation = state.begin_context();
    assert!(!state.restore(old, context("entry-100")));
    assert_eq!(state.current().unwrap().post_id, 2);
    state.policy_pending = false;
    let mut accessible = context("entry-3");
    accessible.entries.retain(|entry| entry.card.post_id != 2);
    assert!(state.restore(generation, accessible));
    assert_eq!(state.current_entry_id(), Some("entry-3"));
    let generation = state.begin_context();
    assert!(state.restore(generation, context("entry-2")));
    assert_eq!(state.current_entry_id(), Some("entry-2"));
}
#[test]
fn old_post_may_receive_a_new_offer_without_reusing_navigation_identity() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    state.restore(generation, context("entry-149"));
    assert_eq!(state.selection_exclusions(), vec![149]);
    assert!(state.selection(
        generation,
        Ok(page(
            "new",
            vec![
                card(20, "new"),
                card(149, "duplicate-current"),
                card(20, "duplicate-lookahead"),
            ],
            false
        ))
    ));
    assert_eq!(state.items.len(), 151);
    assert_eq!(state.items[20].card.session_id, "original-offer");
    assert_eq!(state.items[150].card.session_id, "new");
    assert_eq!(state.items[150].entry_id, None);
    assert_eq!(state.selection_exclusions(), vec![149, 20]);
    assert!(state.navigate(1));
    state.media_ready = true;
    let (visit, event, _) = state.start_view().unwrap();
    let attempt = state.view_attempt;
    state.view(visit, attempt, &event, Ok(()), Instant::now());
    assert_ne!(state.current_entry_id(), Some("entry-20"));
    assert_eq!(state.current_entry_id(), Some(event.as_str()));
    assert_eq!(
        state.items[20].card.post_id,
        state.current().unwrap().post_id
    );
}
#[test]
fn failed_policy_projection_stays_closed_until_a_successful_accessible_retry() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    assert!(state.restore(generation, context("entry-70")));
    let original = cards_value(&state.items);
    state.policy_pending = true;
    let generation = state.begin_context();
    assert!(state.restore_failed(generation, anyhow::anyhow!("projection unavailable")));
    assert!(!state.context_pending);
    assert!(state.policy_pending);
    assert!(!state.navigate(1));
    assert!(!state.navigate(-1));
    assert!(!state.media_ready(state.media_generation, Instant::now()));
    assert!(!state.start_vote(state.current().unwrap().clone(), Grade::Love));
    state.toggle_advance(Instant::now());
    assert!(!state.advance);
    assert!(!state.selection_owned(
        generation,
        state.lookahead_generation,
        Ok(page("unfiltered", vec![card(999, "unfiltered")], false))
    ));
    assert_eq!(cards_value(&state.items), original);
    let retry = state.begin_context();
    assert!(state.policy_pending);
    assert!(!state.restore(generation, context("entry-70")));
    assert!(
        state.policy_pending,
        "a stale successful restore cannot open the gate"
    );
    let mut accessible = context("entry-71");
    accessible.entries.retain(|entry| entry.card.post_id != 70);
    assert!(state.restore(retry, accessible));
    assert!(!state.blocked());
    assert_eq!(state.current_entry_id(), Some("entry-71"));
    assert!(state.items.iter().all(|item| item.card.post_id != 70));
    assert!(state.navigate(1));
}

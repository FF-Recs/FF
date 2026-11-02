use super::support::*;
use super::*;
#[test]
fn media_window_caps_both_directions_in_the_middle_of_history() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    assert!(state.restore(generation, context("entry-70")));
    assert_eq!(
        state
            .media_window()
            .map(|(card, distance)| (card.post_id, distance))
            .collect::<Vec<_>>(),
        vec![
            (70, 0),
            (71, 1),
            (72, 2),
            (73, 3),
            (74, 4),
            (75, 5),
            (69, 1),
            (68, 2),
            (67, 3),
            (66, 4),
            (65, 5)
        ]
    );
    assert!(
        state
            .media_window()
            .all(|(card, _)| card.session_id == "original-offer")
    );
}

#[test]
fn media_window_frontier_uses_only_available_offers_without_expanding_lookahead() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    assert!(state.restore(generation, context("entry-149")));
    assert!(state.selection(
        generation,
        Ok(page(
            "new",
            (150..154).map(|id| card(id, "new")).collect(),
            false,
        ))
    ));
    assert_eq!(
        state
            .media_window()
            .map(|(card, distance)| (card.post_id, distance))
            .collect::<Vec<_>>(),
        vec![
            (149, 0),
            (150, 1),
            (151, 2),
            (152, 3),
            (153, 4),
            (148, 1),
            (147, 2),
            (146, 3),
            (145, 4),
            (144, 5)
        ]
    );
    assert_eq!(state.items.len(), 154);
    assert!(!state.needs_refill());
}

#[test]
fn media_window_forward_history_precedes_unseen_offers_under_one_cap() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    assert!(state.restore(generation, context("entry-147")));
    assert!(state.selection(
        generation,
        Ok(page(
            "new",
            (150..154).map(|id| card(id, "new")).collect(),
            false,
        ))
    ));
    let window: Vec<_> = state.media_window().map(|(card, _)| card).collect();
    assert_eq!(
        window.iter().map(|card| card.post_id).collect::<Vec<_>>(),
        vec![147, 148, 149, 150, 151, 152, 146, 145, 144, 143, 142]
    );
    assert_eq!(window[1].session_id, "original-offer");
    assert_eq!(window[2].session_id, "original-offer");
    assert_eq!(window[3].session_id, "new");
}

#[test]
fn media_window_tracks_current_visit_while_grade_navigation_is_deferred() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    assert!(state.restore(generation, context("entry-147")));
    assert!(state.selection(generation, Ok(page("new", vec![card(150, "new")], false))));
    present(&mut state);
    let offered = state.current().unwrap().clone();
    assert!(state.start_vote(offered, Grade::Love));
    assert_eq!(
        state.request_navigation(NavigationIntent::Next),
        Admission::Deferred
    );
    assert_eq!(
        state
            .media_window()
            .map(|(card, distance)| (card.post_id, distance))
            .collect::<Vec<_>>(),
        vec![
            (147, 0),
            (148, 1),
            (149, 2),
            (150, 3),
            (146, 1),
            (145, 2),
            (144, 3),
            (143, 4),
            (142, 5)
        ]
    );
}

#[test]
fn media_window_is_empty_when_blocked_loading_empty_or_current_is_invalid() {
    let mut state = ViewerState::default();
    assert_eq!(state.media_window().count(), 0);
    state.loading = false;
    assert_eq!(state.media_window().count(), 0);
    let generation = state.begin_context();
    assert!(state.restore(generation, context("entry-0")));
    assert_eq!(
        state
            .media_window()
            .map(|(card, _)| card.post_id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5]
    );
    state.loading = true;
    assert_eq!(state.media_window().count(), 0);
    state.loading = false;
    state.policy_pending = true;
    assert_eq!(state.media_window().count(), 0);
    state.policy_pending = false;
    let generation = state.begin_context();
    assert_eq!(state.media_window().count(), 0);
    state.policy_pending = true;
    assert!(state.restore_failed(generation, anyhow::anyhow!("projection")));
    assert_eq!(state.media_window().count(), 0);
    let generation = state.begin_context();
    assert!(state.restore(generation, context("entry-149")));
    assert_eq!(
        state
            .media_window()
            .map(|(card, _)| card.post_id)
            .collect::<Vec<_>>(),
        vec![149, 148, 147, 146, 145, 144]
    );
    state.index = usize::MAX;
    assert_eq!(state.media_window().count(), 0);
}

#[test]
fn media_window_drops_obsolete_offers_and_accepts_only_current_replacements() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    assert!(state.restore(generation, context("entry-148")));
    assert!(state.selection(generation, Ok(page("old", vec![card(150, "old")], false))));
    let old_read = state.lookahead_generation;
    assert!(state.media_window().any(|(card, _)| card.post_id == 150));
    assert!(state.invalidate_lookahead());
    assert_eq!(
        state
            .media_window()
            .map(|(card, _)| card.post_id)
            .collect::<Vec<_>>(),
        vec![148, 149, 147, 146, 145, 144, 143]
    );
    assert!(!state.selection_owned(
        generation,
        old_read,
        Ok(page("stale", vec![card(999, "stale")], false))
    ));
    assert!(!state.media_window().any(|(card, _)| card.post_id == 999));
    assert!(state.selection_owned(
        generation,
        state.lookahead_generation,
        Ok(page("new", vec![card(151, "new")], false))
    ));
    assert_eq!(
        state
            .media_window()
            .map(|(card, _)| card.post_id)
            .collect::<Vec<_>>(),
        vec![148, 149, 151, 147, 146, 145, 144, 143]
    );
    let generation = state.begin_context();
    let mut replacement = context("entry-149");
    replacement
        .entries
        .retain(|entry| entry.card.post_id != 148);
    assert!(state.restore(generation, replacement));
    assert!(
        !state
            .media_window()
            .any(|(card, _)| matches!(card.post_id, 148 | 151))
    );
    assert_eq!(state.media_window().next().unwrap().0.post_id, 149);
}

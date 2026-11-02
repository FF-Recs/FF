use super::support::*;
use super::*;
#[test]
fn navigation_preserves_all_exact_ids_and_original_card_provenance() {
    let mut state = state();
    let base = 9_007_199_254_740_993_i64;
    state.items = (0..150)
        .map(|i| item(base + i, "original-session"))
        .collect();
    state.index = 139;
    state.furthest = 145;
    assert!(state.navigate(1));
    assert_eq!(
        (state.index, state.furthest, state.items.len()),
        (140, 145, 150)
    );
    let current = state.current().unwrap();
    assert_eq!(current.post_id, base + 140);
    assert_eq!(current.session_id, "original-session");
    assert_eq!(current.model_id, "original-model");
    assert_eq!(
        (current.epoch, current.position, current.universe_size),
        (7, 123, 654321)
    );
}
#[test]
fn boundaries_back_pause_filter_retry_and_media_attempts_are_independent() {
    let mut state = state();
    state.advance = true;
    assert!(!state.navigate(-1));
    assert!(!state.advance);
    assert!(state.navigate(isize::MAX));
    assert!(!state.navigate(1));
    let selection = state.generation;
    let visit = state.visit;
    let old_media = state.media_generation;
    state.retry_media();
    assert_eq!((state.generation, state.visit), (selection, visit));
    assert!(!state.media_ready(old_media, Instant::now()));
    assert!(!state.media_ready);
    state.select("balanced".into(), "cat -dog".into());
    let mode = state.mode.clone();
    let filter = state.filter.clone();
    state.select(mode, filter);
    assert_eq!(state.filter, "cat -dog");
    assert!(!state.selection(selection, Ok(page("old", vec![], true))));
}
#[test]
fn timed_advance_requires_persisted_view_and_never_applies_to_video_or_review() {
    let now = Instant::now();
    let mut state = state();
    state.media_ready = true;
    state.advance = true;
    state.ready_since = Some(now);
    assert!(state.advance_deadline(2).is_none());
    acknowledge_view(&mut state, now);
    assert!(!state.timed_advance(now + Duration::from_secs(1), 2));
    assert!(state.timed_advance(now + Duration::from_secs(2), 2));
    assert_eq!(state.index, 1);
    state.index = 0;
    state.media_ready = true;
    acknowledge_view(&mut state, now);
    state.ready_since = Some(now);
    state.items[0].card.file_extension = "mp4".into();
    assert!(state.advance_deadline(2).is_none());
    state.items[0].card.file_extension = "jpg".into();
    state.mode = "rating_review".into();
    assert!(state.advance_deadline(2).is_none());
}
#[test]
fn selection_errors_stop_next_but_context_and_back_can_escape() {
    let mut state = state();
    present(&mut state);
    assert!(state.navigate(1));
    state.session = "frontier".into();
    state.more = true;
    assert_eq!(
        state.request_navigation(NavigationIntent::Next),
        Admission::Deferred
    );
    assert!(state.start_refill());
    state.refill(
        state.generation,
        "frontier",
        Err(anyhow::anyhow!("offline")),
    );
    assert!(state.take_navigation().is_none());
    assert!(!state.invalidate_lookahead());
    assert!(state.take_navigation().is_none());
    assert_eq!(
        state.request_navigation(NavigationIntent::Back),
        Admission::Ready
    );
    assert_eq!(state.take_navigation(), Some(NavigationIntent::Back));
    assert!(state.navigate(-1));
    state.selection_error = "selection failed".into();
    let context = NavigationIntent::Context {
        mode: "fresh".into(),
        filter: "cat".into(),
        fresh: true,
    };
    assert_eq!(state.request_navigation(context.clone()), Admission::Ready);
    assert_eq!(state.pending_context(), Some(("fresh", "cat", true)));
    assert_eq!(state.take_navigation(), Some(context));
}

#[test]
fn initial_context_and_restored_frontier_navigation_have_distinct_admission() {
    let mut state = ViewerState::default();
    assert_eq!(state.grade_admission(), Admission::Unavailable);
    assert_eq!(
        state.navigation_admission(&NavigationIntent::Next),
        Admission::Unavailable
    );
    assert_eq!(
        state.navigation_admission(&NavigationIntent::Back),
        Admission::Unavailable
    );
    let initial = NavigationIntent::Context {
        mode: "fresh".into(),
        filter: String::new(),
        fresh: true,
    };
    assert_eq!(
        state.navigation_admission(&initial),
        state.context_admission()
    );
    assert_eq!(state.request_navigation(initial.clone()), Admission::Ready);
    assert_eq!(state.take_navigation(), Some(initial));
    let generation = state.begin_context();
    assert_eq!(state.context_admission(), Admission::Unavailable);
    assert_eq!(
        state.navigation_admission(&NavigationIntent::Next),
        Admission::Unavailable
    );
    assert!(state.restore(generation, context("entry-149")));
    assert_eq!(
        state.request_navigation(NavigationIntent::Next),
        Admission::Deferred
    );
    assert!(
        state.take_navigation().is_none(),
        "restore needs a backend continuation first"
    );
    assert!(state.selection(
        generation,
        Ok(page("continued", vec![card(150, "continued")], false)),
    ));
    assert_eq!(state.take_navigation(), Some(NavigationIntent::Next));
    assert!(state.navigate(1));
    assert_eq!(state.current().unwrap().post_id, 150);
}

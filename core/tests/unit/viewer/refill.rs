use super::support::*;
use super::*;
fn history_campaign(mode: &str) -> ViewerState {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    assert!(state.restore(
        generation,
        BrowseContext {
            context_id: format!("long-{mode}-history"),
            mode: mode.into(),
            filter: String::new(),
            entries: Vec::new(),
            cursor: None,
        }
    ));
    assert!(state.selection(
        generation,
        Ok(page(
            "initial",
            (0..5).map(|id| card(id, "initial")).collect(),
            false,
        ))
    ));
    state
}
#[test]
fn stale_selection_and_refill_cannot_replace_current_generation() {
    let mut state = state();
    state.generation = 5;
    state.session = "current".into();
    state.refilling = true;
    assert!(!state.selection(4, Ok(page("stale", vec![], true))));
    state.refill(5, "stale", Ok(page("stale", vec![], true)));
    assert!(state.refilling);
    assert_eq!(state.current().unwrap().post_id, i64::MAX);
    state.refill(
        5,
        "current",
        Ok(page(
            "next",
            vec![card(i64::MAX, "duplicate"), card(5, "next")],
            false,
        )),
    );
    assert_eq!(state.session, "next");
    assert_eq!(state.items.len(), 3);
    assert_eq!(state.items[0].card.session_id, "original");
    assert_eq!(state.items[2].card.session_id, "next");
    state.refill(
        5,
        "next",
        Ok(page("empty", vec![card(5, "duplicate")], false)),
    );
    assert!(state.more);
    assert_eq!(state.session, "empty");
}
#[test]
fn all_modes_share_initial_lookahead_low_water_and_deficit_policy() {
    let mut state = state();
    for mode in [
        "balanced",
        "favorites",
        "fresh",
        "discover",
        "develop",
        "train_broadly",
        "q3_probe",
        "rating_review",
    ] {
        state.mode = mode.into();
        assert_eq!(state.selection_page_size(), 5);
        state.items = (0..5).map(|id| item(id, "initial")).collect();
        for ahead in 0..=4 {
            state.furthest = 4 - ahead;
            state.index = state.furthest;
            assert_eq!(state.unvisited(), ahead);
            let expected = (ahead <= 2).then_some(4 - ahead);
            assert_eq!(state.needs_refill(), expected.is_some());
            if state.is_auto() {
                assert_eq!(
                    state.auto_next_request().map(|request| request.page_size),
                    expected,
                );
                assert!(state.gallery_next_request().is_none());
            } else {
                assert_eq!(
                    state
                        .gallery_next_request()
                        .map(|request| request.page_size),
                    expected,
                );
                assert!(state.auto_next_request().is_none());
            }
        }
    }
}
#[test]
fn auto_refill_uses_frontier_deficit_and_excludes_only_current_and_lookahead() {
    let mut state = state();
    state.select("balanced".into(), String::new());
    let base = 9_007_199_254_740_993_i64;
    assert!(state.selection(
        state.generation,
        Ok(page(
            "initial",
            (0..5).map(|id| card(base + id, "initial")).collect(),
            false,
        )),
    ));
    assert!(!state.needs_refill());
    assert!(!state.start_refill());
    assert!(state.navigate(1));
    assert!(!state.needs_refill());
    assert!(state.navigate(1));
    assert!(state.needs_refill());
    let request = state.auto_next_request().unwrap();
    assert_eq!(request.session_id, "initial");
    assert_eq!(request.page_size, 2);
    assert_eq!(
        request.exclude_post_ids,
        (base + 2..base + 5).collect::<Vec<_>>()
    );
    assert!(state.navigate(-1));
    assert!(!state.needs_refill());
    assert!(state.auto_next_request().is_none());
    assert!(!state.start_refill());
    assert!(state.navigate(1));
    assert!(state.start_refill());
    assert!(!state.start_refill());
    // A valid reply remains valid if the user backtracks while it is in flight.
    assert!(state.navigate(-1));
    state.media_ready = true;
    let (_, event, _) = state.start_view().unwrap();
    let visit = state.visit;
    let media_generation = state.media_generation;
    let retained = cards_value(&state.items);
    let current = state.current().unwrap().post_id;
    state.refill(
        state.generation,
        &request.session_id,
        Ok(page(
            "continued",
            vec![card(base + 5, "continued"), card(base + 6, "continued")],
            false,
        )),
    );
    assert_eq!(state.session, "continued");
    assert_eq!((state.index, state.furthest), (1, 2));
    assert_eq!(state.current().unwrap().post_id, current);
    assert_eq!(
        (state.visit, state.media_generation),
        (visit, media_generation)
    );
    assert_eq!(state.view_event(), Some(event.as_str()));
    assert!(state.viewing());
    assert!(state.media_ready);
    assert_eq!(cards_value(&state.items[..5]), retained);
    assert_eq!(state.items[5].card.session_id, "continued");
    assert_eq!(state.unvisited(), 4);
    assert!(!state.needs_refill());
}
#[test]
fn auto_lookahead_stays_bounded_without_truncating_history() {
    let mut state = history_campaign("fresh");
    let mut next_id = 5;
    let mut visited = Vec::new();
    let offered = state.current().unwrap().clone();
    present(&mut state);
    visited.push((state.current_entry_id().unwrap().to_owned(), offered));
    for _ in 0..160 {
        assert!(state.navigate(1));
        assert!(state.current_entry_id().is_none());
        let offered = state.current().unwrap().clone();
        present(&mut state);
        visited.push((state.current_entry_id().unwrap().to_owned(), offered));
        assert!(state.unvisited() <= 4);
        if let Some(request) = state.auto_next_request() {
            assert!(state.unvisited() <= 2);
            assert_eq!(request.page_size, 4 - state.unvisited());
            assert_eq!(request.exclude_post_ids.len(), state.unvisited() + 1);
            assert_eq!(
                request.exclude_post_ids[0],
                state.current().unwrap().post_id
            );
            assert!(
                request
                    .exclude_post_ids
                    .contains(&state.current().unwrap().post_id)
            );
            assert!(state.start_refill());
            let items = (0..request.page_size)
                .map(|_| {
                    let item = card(next_id, "continued");
                    next_id += 1;
                    item
                })
                .collect();
            state.refill(
                state.generation,
                &request.session_id,
                Ok(page("continued", items, false)),
            );
            assert_eq!(state.unvisited(), 4);
        }
        assert!(state.items.len() <= state.index + 5);
    }
    assert_eq!(state.index, 160);
    assert_eq!(state.items[0].card.post_id, 0);
    assert_eq!(visited.len(), 161);
    let mut entry_ids = std::collections::HashSet::new();
    for (entry, (entry_id, offered)) in state.items.iter().zip(&visited) {
        assert!(entry_ids.insert(entry_id));
        assert_eq!(entry.entry_id.as_ref(), Some(entry_id));
        assert_eq!(
            serde_json::to_value(&entry.card).unwrap(),
            serde_json::to_value(offered).unwrap()
        );
    }
    assert!(
        state.items[visited.len()..]
            .iter()
            .all(|entry| entry.entry_id.is_none())
    );
}
#[test]
fn auto_zero_append_adopts_token_and_exhaustion_without_replacing_visit_or_history() {
    let mut state = state();
    state.mode = "favorites".into();
    state.session = "original".into();
    state.more = true;
    state.media_ready = true;
    let (_, event, _) = state.start_view().unwrap();
    let visit = state.visit;
    let retained = cards_value(&state.items);
    assert!(state.start_refill());
    state.refill(
        state.generation,
        "original",
        Ok(page("duplicate-only", vec![card(i64::MAX, "other")], false)),
    );
    assert_eq!(state.session, "duplicate-only");
    assert!(
        state.more,
        "deduplication cannot declare backend exhaustion"
    );
    assert!(!state.refilling);
    assert!(!state.needs_refill(), "do not poll a no-progress frontier");
    assert!(!state.start_refill());
    assert_eq!(state.visit, visit);
    assert_eq!((state.index, state.furthest), (0, 0));
    assert_eq!(state.view_event(), Some(event.as_str()));
    assert!(state.viewing());
    assert_eq!(cards_value(&state.items), retained);
    assert!(state.navigate(1));
    assert!(state.needs_refill());
    assert_eq!(state.auto_next_request().unwrap().page_size, 4);
    assert!(state.start_refill());
    let visit = state.visit;
    state.refill(
        state.generation,
        "duplicate-only",
        Ok(page("empty-exhausted", vec![], true)),
    );
    assert_eq!(state.session, "empty-exhausted");
    assert!(!state.more);
    assert!(!state.start_refill());
    assert_eq!(state.visit, visit);
    assert_eq!(state.current().unwrap().post_id, 4);
    assert_eq!(cards_value(&state.items), retained);
}
#[test]
fn auto_explicit_resume_uses_new_token_even_without_buffered_forward_progress() {
    let mut state = state();
    state.mode = "balanced".into();
    state.session = "original".into();
    state.more = true;
    assert!(state.navigate(1));
    assert!(state.start_refill());
    let visit = state.visit;
    state.refill(
        state.generation,
        "original",
        Ok(page("continued-empty", vec![], false)),
    );
    assert_eq!(state.session, "continued-empty");
    assert!(state.more);
    assert!(!state.needs_refill());
    assert!(!state.start_refill());
    assert!(!state.navigate(1));
    state.resume_refill();
    let request = state.auto_next_request().unwrap();
    assert_eq!(request.session_id, "continued-empty");
    assert_eq!(request.page_size, 4);
    assert_eq!(request.exclude_post_ids, vec![4]);
    assert!(state.start_refill());
    assert_eq!(state.visit, visit);
    assert_eq!(state.current().unwrap().post_id, 4);
}
#[test]
fn empty_initial_pages_preserve_explicit_exhaustion_without_polling_in_both_modes() {
    for mode in ["balanced", "discover"] {
        for exhausted in [false, true] {
            let mut state = state();
            state.select(mode.into(), String::new());
            assert!(state.selection(state.generation, Ok(page("empty", vec![], exhausted)),));
            assert_eq!(state.session, "empty");
            assert_eq!(state.more, !exhausted);
            assert!(!state.needs_refill());
            assert!(state.auto_next_request().is_none());
            assert!(state.gallery_next_request().is_none());
            assert!(!state.start_refill());
        }
    }
}
#[test]
fn gallery_short_chunks_and_cohort_boundaries_preserve_the_active_visit_and_offers() {
    let mut state = state();
    state.select("discover".into(), "cat -dog".into());
    let base = 9_007_199_254_740_993_i64;
    assert!(state.selection(
        state.generation,
        Ok(page(
            "cohort-a",
            (0..5).map(|id| card(base + id, "cohort-a")).collect(),
            false,
        )),
    ));
    assert!(!state.start_refill());
    assert!(state.navigate(2));
    let mut next_id = base + 5;
    for (token, count) in [("cohort-a", 1), ("cohort-b", 1), ("cohort-b", 2)] {
        let request = state.gallery_next_request().unwrap();
        assert_eq!(request.session_id, state.session);
        assert_eq!(request.page_size, 2);
        assert_eq!(
            request.exclude_post_ids,
            state.items[state.index..]
                .iter()
                .map(|item| item.card.post_id)
                .collect::<Vec<_>>(),
        );
        assert!(state.start_refill());
        assert!(state.navigate(-1));
        assert!(!state.needs_refill());
        assert!(state.gallery_next_request().is_none());
        state.media_ready = true;
        let (_, event, current) = state.start_view().unwrap();
        let visit = state.visit;
        let media_generation = state.media_generation;
        let indices = (state.index, state.furthest);
        let retained = cards_value(&state.items);
        let before = state.items.len();
        let items = (0..count)
            .map(|_| {
                let mut item = card(next_id, token);
                item.position = (next_id - base) as usize;
                item.model_id = format!("model-{next_id}");
                next_id += 1;
                item
            })
            .collect();
        state.refill(
            state.generation,
            &request.session_id,
            Ok(page(token, items, false)),
        );
        assert_eq!(state.session, token);
        assert!(state.more, "a short cohort chunk is not global exhaustion");
        assert!(!state.refill_stalled);
        assert_eq!(cards_value(&state.items[..before]), retained);
        assert_eq!(state.current().unwrap().post_id, current.post_id);
        assert_eq!((state.index, state.furthest), indices);
        assert_eq!(
            (state.visit, state.media_generation),
            (visit, media_generation)
        );
        assert_eq!(state.view_event(), Some(event.as_str()));
        assert!(state.viewing() && state.media_ready);
        assert_eq!(state.items.last().unwrap().card.session_id, token);
        assert!(!state.start_refill(), "backtracking never requests a page");
        assert!(state.navigate(1));
        assert!(!state.needs_refill());
        if count == 1 {
            assert!(state.navigate(1));
            assert!(state.needs_refill());
        }
    }
    assert_eq!(state.unvisited(), 4);
    assert_eq!(state.items[0].card.session_id, "cohort-a");
    assert_eq!(state.items[6].card.session_id, "cohort-b");
    assert_eq!(state.filter, "cat -dog");
}
#[test]
fn gallery_zero_append_and_failed_continuations_recover_only_on_explicit_resume() {
    for mode in [
        "discover",
        "develop",
        "train_broadly",
        "q3_probe",
        "rating_review",
    ] {
        for duplicate in [false, true] {
            let mut state = state();
            state.mode = mode.into();
            state.session = "original".into();
            state.more = true;
            assert!(state.navigate(1));
            state.media_ready = true;
            let (_, event, current) = state.start_view().unwrap();
            let visit = state.visit;
            let media_generation = state.media_generation;
            let retained = cards_value(&state.items);
            assert!(state.start_refill());
            let items = if duplicate {
                vec![card(4, "duplicate")]
            } else {
                vec![]
            };
            state.refill(
                state.generation,
                "original",
                Ok(page("continued-empty", items, false)),
            );
            assert_eq!(state.session, "continued-empty");
            assert!(state.more);
            assert!(!state.needs_refill());
            assert!(!state.start_refill());
            assert!(state.gallery_next_request().is_none());
            assert!(!state.navigate(1));
            state.resume_refill();
            let request = state.gallery_next_request().unwrap();
            assert_eq!(request.session_id, "continued-empty");
            assert_eq!(request.page_size, 4);
            assert_eq!(request.exclude_post_ids, vec![4]);
            assert!(state.start_refill());
            state.refill(
                state.generation,
                &request.session_id,
                Err(anyhow::anyhow!("offline")),
            );
            assert_eq!(state.session, "continued-empty");
            assert!(!state.refill_error.is_empty());
            assert!(!state.start_refill());
            state.refill_error.clear();
            state.resume_refill();
            let request = state.gallery_next_request().unwrap();
            assert_eq!(request.session_id, "continued-empty");
            assert!(state.start_refill());
            state.refill(
                state.generation,
                &request.session_id,
                Ok(page("recovered", vec![card(5, "recovered")], false)),
            );
            assert!(!state.refill_stalled);
            assert_eq!(state.session, "recovered");
            assert_eq!(cards_value(&state.items[..2]), retained);
            assert_eq!(state.current().unwrap().post_id, current.post_id);
            assert_eq!(
                (state.visit, state.media_generation),
                (visit, media_generation)
            );
            assert_eq!(state.view_event(), Some(event.as_str()));
            assert!(state.media_ready && state.viewing());
            assert!(state.start_refill());
            state.refill(
                state.generation,
                "recovered",
                Ok(page("exhausted", vec![], true)),
            );
            assert_eq!(state.session, "exhausted");
            assert!(!state.more);
            assert!(!state.start_refill());
            assert!(state.navigate(1));
        }
    }
}
#[test]
fn gallery_lookahead_stays_bounded_without_truncating_history() {
    let mut state = history_campaign("discover");
    let mut next_id = 5;
    let mut visited = Vec::new();
    let offered = state.current().unwrap().clone();
    present(&mut state);
    visited.push((state.current_entry_id().unwrap().to_owned(), offered));
    for step in 0..160 {
        assert!(state.navigate(1));
        assert!(state.current_entry_id().is_none());
        let offered = state.current().unwrap().clone();
        present(&mut state);
        visited.push((state.current_entry_id().unwrap().to_owned(), offered));
        assert!(state.unvisited() <= 4);
        if let Some(request) = state.gallery_next_request() {
            assert_eq!(state.unvisited(), 2);
            assert_eq!(request.page_size, 2);
            assert_eq!(request.exclude_post_ids.len(), state.unvisited() + 1);
            assert_eq!(
                request.exclude_post_ids[0],
                state.current().unwrap().post_id
            );
            assert!(
                request
                    .exclude_post_ids
                    .contains(&state.current().unwrap().post_id)
            );
            assert!(state.navigate(-1));
            present(&mut state);
            assert!(state.gallery_next_request().is_none());
            assert!(!state.start_refill());
            assert!(state.navigate(1));
            present(&mut state);
            assert!(state.start_refill());
            let token = format!("cohort-{}", step / 10);
            let items = (0..request.page_size)
                .map(|_| {
                    let item = card(next_id, &token);
                    next_id += 1;
                    item
                })
                .collect();
            state.refill(
                state.generation,
                &request.session_id,
                Ok(page(&token, items, false)),
            );
            assert_eq!(state.unvisited(), 4);
        }
        assert!(state.items.len() <= state.index + 5);
    }
    assert_eq!(state.index, 160);
    assert_eq!(state.items[0].card.post_id, 0);
    assert_eq!(visited.len(), 161);
    let mut entry_ids = std::collections::HashSet::new();
    for (entry, (entry_id, offered)) in state.items.iter().zip(&visited) {
        assert!(entry_ids.insert(entry_id));
        assert_eq!(entry.entry_id.as_ref(), Some(entry_id));
        assert_eq!(
            serde_json::to_value(&entry.card).unwrap(),
            serde_json::to_value(offered).unwrap()
        );
    }
    assert!(
        state.items[visited.len()..]
            .iter()
            .all(|entry| entry.entry_id.is_none())
    );
}
#[test]
fn exclusion_tail_is_bounded_even_with_large_visited_forward_history() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    state.restore(generation, context("entry-20"));
    state.selection(
        generation,
        Ok(page("new", vec![card(150, "new"), card(151, "new")], false)),
    );
    assert_eq!(state.selection_exclusions(), vec![20, 150, 151]);
    assert!(state.navigate(129));
    assert_eq!(state.selection_exclusions(), vec![149, 150, 151]);
}

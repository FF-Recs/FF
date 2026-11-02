use super::support::*;
use super::*;
#[test]
fn publication_replaces_only_unseen_work_and_keeps_pending_presentation_authority() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    state.restore(generation, context("entry-149"));
    state.selection(
        generation,
        Ok(page("old-model", vec![card(150, "old-model")], false)),
    );
    state.media_ready = true;
    let (visit, event, _) = state.start_view().unwrap();
    let attempt = state.view_attempt;
    let media = state.media_generation;
    let old_read = state.lookahead_generation;
    assert!(state.invalidate_lookahead());
    assert!(!state.selection_owned(
        generation,
        old_read,
        Ok(page("obsolete", vec![card(999, "obsolete")], false))
    ));
    state.refill_owned(
        generation,
        old_read,
        "",
        Ok(page("obsolete", vec![card(998, "obsolete")], false)),
    );
    assert_eq!(
        (state.generation, state.visit, state.media_generation),
        (generation, visit, media)
    );
    assert_eq!(state.items.len(), 150);
    assert!(state.viewing());
    state.view(visit, attempt, &event, Ok(()), Instant::now());
    assert!(state.viewed());
    assert_eq!(state.current_entry_id(), Some("entry-149"));
    assert_eq!(state.current().unwrap().model_id, "original-model");
    assert_eq!(state.session, "old-model");
    assert!(state.start_refill());
    let mut new_offer = card(151, "old-model");
    new_offer.model_id = "published-model".into();
    state.refill_owned(
        generation,
        state.lookahead_generation,
        "old-model",
        Ok(page("old-model", vec![new_offer], false)),
    );
    assert_eq!(state.items[150].card.session_id, "old-model");
    assert_eq!(state.items[150].card.model_id, "published-model");
}
#[test]
fn publication_preserves_exhausted_and_no_progress_stops_until_explicit_retry() {
    for exhausted in [false, true] {
        for empty in [false, true] {
            let mut state = ViewerState::default();
            let generation = state.begin_context();
            let mut restored = context("entry-149");
            if empty {
                restored.entries.clear();
                restored.cursor = None;
            }
            state.restore(generation, restored);
            state.selection(generation, Ok(page("stopped", vec![], exhausted)));
            if exhausted {
                // A fully visited final page can be exhausted without stalling.
                state.refill_stalled = false;
            }
            let stopped = state.refill_stalled;
            let retained = cards_value(&state.items);
            let stale_read = state.lookahead_generation;
            assert!(!state.invalidate_lookahead());
            assert_eq!(state.refill_stalled, stopped);
            assert_eq!(state.more, !exhausted);
            assert_eq!(cards_value(&state.items), retained);
            assert!(!state.selection_owned(
                generation,
                stale_read,
                Ok(page("stale", vec![card(500, "stale")], false))
            ));
            assert!(
                !state.invalidate_lookahead(),
                "later publication cannot undo the stop"
            );
            state.resume_refill();
            assert!(state.selection_owned(
                generation,
                state.lookahead_generation,
                Ok(page(
                    "explicit-retry",
                    vec![card(500, "explicit-retry")],
                    true
                ))
            ));
            assert_eq!(
                state.items.last().unwrap().card.session_id,
                "explicit-retry"
            );
        }
    }
}
#[test]
fn publication_discards_queued_offers_without_undoing_a_no_progress_guard() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    state.restore(generation, context("entry-149"));
    state.selection(
        generation,
        Ok(page("queued", vec![card(150, "queued")], false)),
    );
    state.refill_stalled = true;
    assert!(!state.invalidate_lookahead());
    assert_eq!(state.items.len(), 150);
    assert_eq!(state.current_entry_id(), Some("entry-149"));
    assert!(state.refill_stalled);
    assert!(state.more);
    assert!(!state.start_refill());
    state.resume_refill();
    assert!(state.selection_owned(
        generation,
        state.lookahead_generation,
        Ok(page(
            "explicit-next",
            vec![card(150, "explicit-next")],
            false
        ))
    ));
    assert_eq!(state.items.last().unwrap().card.session_id, "explicit-next");
}
#[test]
fn publication_replaces_live_pending_work_even_without_buffered_future_cards() {
    for pending_read in [false, true] {
        let mut state = ViewerState::default();
        let generation = state.begin_context();
        state.restore(generation, context("entry-149"));
        state.refilling = pending_read;
        state.more = !pending_read;
        assert!(state.invalidate_lookahead());
        assert_eq!(state.items.len(), 150);
        assert!(!state.refilling);
        assert_eq!(state.current_entry_id(), Some("entry-149"));
    }
}
#[test]
fn publication_keeps_gallery_cohort_and_auto_epoch_continuation_identity() {
    for mode in ["discover", "balanced"] {
        let mut state = ViewerState::default();
        let generation = state.begin_context();
        state.restore(generation, context("entry-149"));
        state.mode = mode.into();
        let original = serde_json::to_value(state.current().unwrap()).unwrap();
        let mut queued = card(150, "live-continuation");
        queued.epoch = 12;
        queued.position = 90;
        queued.model_id = "previous-model".into();
        state.selection(
            generation,
            Ok(page("live-continuation", vec![queued], false)),
        );
        let stale_read = state.lookahead_generation;
        assert!(state.invalidate_lookahead());
        assert_eq!(state.session, "live-continuation");
        assert_eq!(state.items.len(), 150);
        let token = if state.is_auto() {
            state.auto_next_request().unwrap().session_id
        } else {
            state.gallery_next_request().unwrap().session_id
        };
        assert_eq!(token, "live-continuation");
        state.refill_owned(
            generation,
            stale_read,
            &token,
            Ok(page(&token, vec![card(999, "stale")], false)),
        );
        assert_eq!(state.items.len(), 150);
        assert!(state.start_refill());
        let mut replanned = card(151, &token);
        replanned.epoch = 12;
        replanned.position = 91;
        replanned.model_id = "published-model".into();
        state.refill_owned(
            generation,
            state.lookahead_generation,
            &token,
            Ok(page(&token, vec![replanned], false)),
        );
        assert_eq!(state.session, token);
        assert_eq!(state.items[150].card.epoch, 12);
        assert_eq!(state.items[150].card.position, 91);
        assert_eq!(state.items[150].card.model_id, "published-model");
        assert_eq!(
            serde_json::to_value(state.current().unwrap()).unwrap(),
            original
        );
    }
}
#[test]
fn publication_does_not_reopen_a_confirmed_exhausted_token_with_queued_offers() {
    let mut state = ViewerState::default();
    let generation = state.begin_context();
    state.restore(generation, context("entry-149"));
    state.selection(
        generation,
        Ok(page(
            "exhausted-continuation",
            vec![card(150, "exhausted-continuation")],
            true,
        )),
    );
    assert!(!state.invalidate_lookahead());
    assert_eq!(state.session, "exhausted-continuation");
    assert_eq!(state.items.len(), 150);
    assert!(!state.more);
    assert!(!state.start_refill());
}

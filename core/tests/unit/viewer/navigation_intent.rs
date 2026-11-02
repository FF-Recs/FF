use super::support::*;
use super::*;
#[test]
fn latest_explicit_navigation_wins_over_grade_default_during_both_writes() {
    let context = NavigationIntent::Context {
        mode: "balanced".into(),
        filter: "cat -dog".into(),
        fresh: true,
    };
    for latest in [
        NavigationIntent::Next,
        NavigationIntent::Back,
        context.clone(),
    ] {
        for vote_reply_first in [false, true] {
            let mut state = state();
            state.items.push(item(9, "lookahead"));
            present(&mut state);
            assert!(state.navigate(1));
            assert!(state.media_ready(state.media_generation, Instant::now()));
            let (visit, event, offered) = state.start_view().unwrap();
            let view_attempt = state.view_attempt;
            state.advance = true;
            assert!(state.start_vote(offered.clone(), Grade::Love));
            let vote_attempt = state.vote_attempt;
            for intent in [
                NavigationIntent::Next,
                NavigationIntent::Next,
                context.clone(),
                NavigationIntent::Back,
                latest.clone(),
            ] {
                assert_eq!(state.request_navigation(intent), Admission::Deferred);
                assert!(state.take_navigation().is_none());
            }
            assert!(
                !state.advance,
                "an accepted Back pauses timed advance immediately"
            );
            if vote_reply_first {
                state.vote(
                    state.generation,
                    visit,
                    vote_attempt,
                    &offered,
                    persisted(offered.post_id),
                );
                assert!(state.take_navigation().is_none());
                state.view(visit, view_attempt, &event, Ok(()), Instant::now());
            } else {
                state.view(visit, view_attempt, &event, Ok(()), Instant::now());
                assert!(state.take_navigation().is_none());
                state.vote(
                    state.generation,
                    visit,
                    vote_attempt,
                    &offered,
                    persisted(offered.post_id),
                );
            }
            assert_eq!(state.current().unwrap().post_id, 4);
            assert_eq!(state.items[1].card.current_grade, Some(Grade::Love));
            assert_eq!(state.take_navigation(), Some(latest.clone()));
            assert!(state.take_navigation().is_none());
            match &latest {
                NavigationIntent::Next => {
                    assert!(state.navigate(1));
                    assert_eq!(state.current().unwrap().post_id, 9);
                }
                NavigationIntent::Back => {
                    assert!(state.navigate(-1));
                    assert_eq!(state.current().unwrap().post_id, i64::MAX);
                }
                NavigationIntent::Context { mode, filter, .. } => {
                    state.select(mode.clone(), filter.clone());
                    assert!(state.current().is_none());
                }
            }
            state.vote(0, visit, vote_attempt, &offered, persisted(offered.post_id));
            assert!(
                state.take_navigation().is_none(),
                "late acknowledgement cannot move again"
            );
        }
    }
}

#[test]
fn pending_default_survives_publication_and_waits_for_replacement_lookahead() {
    for vote_reply_first in [false, true] {
        let mut state = state();
        state.items.truncate(1);
        state.more = true;
        state.session = "frontier".into();
        assert!(state.start_refill());
        assert!(state.media_ready(state.media_generation, Instant::now()));
        let (visit, event, offered) = state.start_view().unwrap();
        let view_attempt = state.view_attempt;
        assert!(state.start_vote(offered.clone(), Grade::Love));
        let vote_attempt = state.vote_attempt;
        let stale_read = state.lookahead_generation;
        assert!(state.invalidate_lookahead());
        assert_eq!(state.visit, visit);
        if vote_reply_first {
            state.vote(
                state.generation,
                visit,
                vote_attempt,
                &offered,
                persisted(offered.post_id),
            );
            assert!(state.take_navigation().is_none());
            state.view(visit, view_attempt, &event, Ok(()), Instant::now());
        } else {
            state.view(visit, view_attempt, &event, Ok(()), Instant::now());
            state.vote(
                state.generation,
                visit,
                vote_attempt,
                &offered,
                persisted(offered.post_id),
            );
        }
        assert!(state.take_navigation().is_none(), "no next card exists yet");
        state.refill_owned(
            state.generation,
            stale_read,
            "frontier",
            Ok(page("stale", vec![card(3, "stale")], false)),
        );
        assert_eq!(state.items.len(), 1);
        assert!(state.start_refill());
        state.refill_owned(
            state.generation,
            state.lookahead_generation,
            "frontier",
            Ok(page("replacement", vec![card(5, "replacement")], false)),
        );
        assert_eq!(state.index, 0);
        assert_eq!(state.take_navigation(), Some(NavigationIntent::Next));
        assert!(state.navigate(1));
        assert_eq!(state.current().unwrap().post_id, 5);
        assert!(state.take_navigation().is_none());
        state.vote(
            state.generation,
            visit,
            vote_attempt,
            &offered,
            persisted(offered.post_id),
        );
        assert_eq!(state.index, 1);
        assert!(state.take_navigation().is_none());
    }
}

#[test]
fn exhausted_and_no_progress_frontiers_drop_intent_without_publication_movement() {
    for exhausted in [false, true] {
        let mut state = state();
        state.items.truncate(1);
        state.session = "frontier".into();
        state.more = true;
        present(&mut state);
        let offered = state.current().unwrap().clone();
        assert!(state.start_vote(offered.clone(), Grade::Love));
        let attempt = state.vote_attempt;
        state.vote(
            state.generation,
            state.visit,
            attempt,
            &offered,
            persisted(offered.post_id),
        );
        assert!(state.start_refill());
        state.refill(
            state.generation,
            "frontier",
            Ok(page("empty", vec![], exhausted)),
        );
        assert!(state.take_navigation().is_none());
        assert!(!state.start_refill());
        assert!(!state.invalidate_lookahead());
        assert!(state.take_navigation().is_none());
        if exhausted {
            assert_eq!(
                state.request_navigation(NavigationIntent::Next),
                Admission::Unavailable
            );
        } else {
            assert_eq!(
                state.request_navigation(NavigationIntent::Next),
                Admission::Deferred
            );
            assert!(
                state.start_refill(),
                "only explicit Next resumes a no-progress frontier"
            );
            state.refill(
                state.generation,
                "empty",
                Ok(page("resumed", vec![card(8, "resumed")], false)),
            );
            assert_eq!(state.take_navigation(), Some(NavigationIntent::Next));
            assert!(state.navigate(1));
            assert_eq!(state.current().unwrap().post_id, 8);
        }
        assert!(state.take_navigation().is_none());
    }
}

#[test]
fn pending_navigation_cannot_race_timed_advance_and_dies_at_visit_boundaries() {
    let now = Instant::now();
    for automatic in [false, true] {
        let mut state = state();
        present(&mut state);
        state.advance = true;
        if automatic {
            let offered = state.current().unwrap().clone();
            assert!(state.start_vote(offered.clone(), Grade::Love));
            let attempt = state.vote_attempt;
            state.vote(
                state.generation,
                state.visit,
                attempt,
                &offered,
                persisted(offered.post_id),
            );
        } else {
            assert_eq!(
                state.request_navigation(NavigationIntent::Next),
                Admission::Ready
            );
        }
        assert!(state.advance_deadline(0).is_none());
        assert!(!state.timed_advance(now + Duration::from_secs(100), 0));
        assert_eq!(state.index, 0);
        assert_eq!(state.take_navigation(), Some(NavigationIntent::Next));
        assert!(state.navigate(1));
        assert!(state.take_navigation().is_none());
        assert!(!state.timed_advance(now + Duration::from_secs(100), 0));
    }
    for boundary in ["visit", "suspend", "select"] {
        let mut state = state();
        let context = NavigationIntent::Context {
            mode: "fresh".into(),
            filter: "cat".into(),
            fresh: false,
        };
        assert_eq!(state.request_navigation(context), Admission::Ready);
        assert_eq!(state.pending_context(), Some(("fresh", "cat", false)));
        match boundary {
            "visit" => state.begin_visit(),
            "suspend" => state.suspend(),
            _ => state.select("balanced".into(), String::new()),
        }
        assert!(state.pending_context().is_none());
        assert!(state.take_navigation().is_none());
    }
}

#[test]
fn grading_preserves_preexisting_explicit_context_in_frontier_review_history_and_correction() {
    for kind in ["frontier", "review", "history", "correction"] {
        let mut state = state();
        match kind {
            "review" => state.mode = "rating_review".into(),
            "history" => state.furthest = 1,
            "correction" => state.items[0].card.current_grade = Some(Grade::Meh),
            _ => {}
        }
        assert!(state.media_ready(state.media_generation, Instant::now()));
        let (visit, event, offered) = state.start_view().unwrap();
        let applied = NavigationIntent::Context {
            mode: state.mode.clone(),
            filter: "cat".into(),
            fresh: false,
        };
        assert_eq!(state.request_navigation(applied), Admission::Deferred);
        let (mode, filter, _) = state.pending_context().unwrap();
        let latest = NavigationIntent::Context {
            mode: mode.to_owned(),
            filter: filter.to_owned(),
            fresh: true,
        };
        assert_eq!(
            state.request_navigation(latest.clone()),
            Admission::Deferred
        );
        assert_eq!(
            state.pending_context(),
            Some((state.mode.as_str(), "cat", true))
        );
        assert!(state.start_vote(offered.clone(), Grade::Love));
        assert_eq!(
            state.pending_context(),
            Some((state.mode.as_str(), "cat", true))
        );
        let view_attempt = state.view_attempt;
        let vote_attempt = state.vote_attempt;
        state.view(visit, view_attempt, &event, Ok(()), Instant::now());
        state.vote(
            state.generation,
            visit,
            vote_attempt,
            &offered,
            persisted(offered.post_id),
        );
        assert_eq!(state.current().unwrap().post_id, offered.post_id);
        assert_eq!(state.take_navigation(), Some(latest));
        assert!(state.take_navigation().is_none());
        assert!(state.pending_context().is_none());
    }
}

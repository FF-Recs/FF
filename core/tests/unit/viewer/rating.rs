use super::support::*;
use super::*;
#[test]
fn first_vote_advances_only_after_persistence_and_retry_preserves_offer() {
    let mut state = state();
    present(&mut state);
    let offered = state.current().unwrap().clone();
    assert!(state.start_vote(offered.clone(), Grade::Love));
    assert_eq!(state.index, 0);
    assert!(!state.navigate(1));
    let event = state.vote_event().map(str::to_owned);
    let attempt = state.vote_attempt;
    state.vote(0, 0, attempt, &offered, Err(anyhow::anyhow!("offline")));
    assert!(state.take_navigation().is_none());
    let (retry, grade) = state
        .failed_vote()
        .map(|(card, grade)| (card.clone(), grade))
        .unwrap();
    assert_eq!(retry.session_id, "original");
    assert_eq!(retry.post_id, i64::MAX);
    assert!(state.start_vote(retry.clone(), grade));
    assert_eq!(state.vote_event().map(str::to_owned), event);
    let retry_attempt = state.vote_attempt;
    assert_ne!(attempt, retry_attempt);
    state.vote(0, 0, retry_attempt, &retry, persisted(i64::MAX));
    assert_eq!(state.index, 0);
    assert_eq!(state.take_navigation(), Some(NavigationIntent::Next));
    assert!(state.navigate(1));
    assert!(state.take_navigation().is_none());
    assert_eq!(state.index, 1);
    assert_eq!(state.items[0].card.current_grade, Some(Grade::Love));
}
#[test]
fn rejected_submission_preserves_failed_retry_and_pending_write_authority() {
    let mut state = state();
    present(&mut state);
    let offered = state.current().unwrap().clone();
    state.reject_vote("write capacity unavailable".into());
    assert!(state.vote_request().is_none());
    assert!(state.failed_vote().is_none());
    assert!(state.vote_event().is_none());
    assert_eq!(state.vote_attempt, 0);
    assert!(!state.vote_error().is_empty());
    assert_eq!(state.context_admission(), Admission::Deferred);

    assert!(state.start_vote(offered.clone(), Grade::Love));
    let attempt = state.vote_attempt;
    let event = state.vote_event().unwrap().to_owned();
    state.reject_vote("late rejected submission".into());
    assert!(state.voting());
    assert!(state.vote_error().is_empty());
    assert_eq!(state.vote_attempt, attempt);
    assert_eq!(state.vote_request().unwrap().0.session_id, "original");
    assert!(state.failed_vote().is_none());
    state.vote(
        state.generation,
        state.visit,
        attempt,
        &offered,
        Err(anyhow::anyhow!("rating offline")),
    );

    state.reject_vote("presentation acknowledgement unavailable".into());
    assert_eq!(state.vote_event(), Some(event.as_str()));
    assert!(!state.vote_error().is_empty());
    let (retry, grade) = state.failed_vote().unwrap();
    let retry = retry.clone();
    assert_eq!(retry.session_id, "original");
    assert_eq!(grade, Grade::Love);
    state.retry_media();
    assert_eq!(state.grade_admission(), Admission::Unavailable);
    assert!(state.start_vote(retry.clone(), grade));
    assert_eq!(state.vote_event(), Some(event.as_str()));
    assert!(state.vote_error().is_empty());
    let retry_attempt = state.vote_attempt;
    assert_ne!(retry_attempt, attempt);
    state.vote(
        state.generation,
        state.visit,
        retry_attempt,
        &retry,
        persisted(retry.post_id),
    );
    assert_eq!(state.current().unwrap().current_grade, Some(Grade::Love));
    assert_eq!(state.take_navigation(), Some(NavigationIntent::Next));
}
#[test]
fn corrections_review_and_history_votes_stay() {
    for mode in ["discover", "rating_review"] {
        let mut state = state();
        state.mode = mode.into();
        state.advance = true;
        if mode == "discover" {
            state.items[0].card.current_grade = Some(Grade::Meh);
        }
        present(&mut state);
        let offered = state.current().unwrap().clone();
        assert!(state.start_vote(offered.clone(), Grade::Love));
        let attempt = state.vote_attempt;
        state.vote(0, 0, attempt, &offered, persisted(i64::MAX));
        assert!(state.take_navigation().is_none());
        assert_eq!(state.index, 0);
        assert_eq!(state.items[0].card.session_id, "original");
        assert_eq!(state.items[0].card.model_id, "original-model");
        if mode == "discover" {
            assert!(!state.advance);
        }
    }
    let mut state = state();
    state.furthest = 1;
    present(&mut state);
    let offered = state.current().unwrap().clone();
    assert!(state.start_vote(offered.clone(), Grade::Love));
    let attempt = state.vote_attempt;
    state.vote(0, 0, attempt, &offered, persisted(i64::MAX));
    assert!(state.take_navigation().is_none());
    assert_eq!(state.index, 0);
}
#[test]
fn stale_vote_has_no_authority_over_a_new_selection_or_visit() {
    let mut state = state();
    present(&mut state);
    let offered = state.current().unwrap().clone();
    let generation = state.generation;
    let visit = state.visit;
    assert!(state.start_vote(offered.clone(), Grade::Love));
    let stale_attempt = state.vote_attempt;
    state.select("fresh".into(), String::new());
    assert!(state.selection(
        state.generation,
        Ok(page(
            "new-selection",
            vec![card(7, "new-offer"), card(8, "new-offer")],
            false
        )),
    ));
    present(&mut state);
    let current = state.current().unwrap().clone();
    assert!(state.start_vote(current.clone(), Grade::Love));
    let current_attempt = state.vote_attempt;
    let active_event = state.vote_event().map(str::to_owned);
    state.vote(
        generation,
        visit,
        stale_attempt,
        &offered,
        persisted(offered.post_id),
    );
    assert!(state.voting());
    assert_eq!(state.vote_event().map(str::to_owned), active_event);
    assert_eq!(state.items[0].card.current_grade, None);
    state.vote(
        state.generation,
        state.visit,
        current_attempt,
        &current,
        persisted(7),
    );
    assert_eq!(state.take_navigation(), Some(NavigationIntent::Next));
    assert!(state.navigate(1));
    present(&mut state);
    let next = state.current().unwrap().clone();
    assert!(state.start_vote(next.clone(), Grade::Love));
    let active_event = state.vote_event().map(str::to_owned);
    state.vote(
        state.generation,
        state.visit.wrapping_sub(1),
        current_attempt,
        &current,
        persisted(7),
    );
    assert!(state.voting());
    assert_eq!(state.vote_event().map(str::to_owned), active_event);
    assert_eq!(state.index, 1);
    assert_eq!(state.items[0].card.current_grade, Some(Grade::Love));
    assert_eq!(state.items[0].card.session_id, "new-offer");
    assert_eq!(state.items[1].card.current_grade, None);
}
#[test]
fn delayed_initial_and_continuation_offers_project_the_acknowledged_grade() {
    for continuation in [false, true] {
        for stale_grade in [None, Some(Grade::Meh)] {
            let mut state = ViewerState::default();
            let generation = state.begin_context();
            state.restore(generation, context("entry-20"));
            state.session = "pending-cohort".into();
            state.refilling = true;
            let mut delayed = card(20, "offered-before-vote");
            delayed.current_grade = stale_grade;
            present(&mut state);
            let offered = state.current().unwrap().clone();
            assert!(state.start_vote(offered.clone(), Grade::Love));
            let attempt = state.vote_attempt;
            state.vote(generation, state.visit, attempt, &offered, persisted(20));
            assert!(state.take_navigation().is_none());
            assert!(state.navigate(1));
            let read = state.lookahead_generation;
            if continuation {
                state.refill_owned(
                    generation,
                    read,
                    "pending-cohort",
                    Ok(page("next-cohort", vec![delayed.clone()], false)),
                );
            } else {
                assert!(state.selection_owned(
                    generation,
                    read,
                    Ok(page("next-cohort", vec![delayed.clone()], false))
                ));
            }
            assert_eq!(state.items.len(), 151);
            delayed.current_grade = Some(Grade::Love);
            assert_eq!(
                serde_json::to_value(&state.items[150].card).unwrap(),
                serde_json::to_value(&delayed).unwrap()
            );
            assert_eq!(state.items[20].card.current_grade, Some(Grade::Love));
            assert_eq!(state.current_entry_id(), Some("entry-21"));
            assert_eq!(state.items[150].entry_id, None);
        }
    }
}
#[test]
fn grade_admission_requires_this_display_and_accepts_one_grade_during_presentation() {
    let mut state = state();
    assert_eq!(state.grade_admission(), Admission::Unavailable);
    assert!(!state.start_vote(state.items[0].card.clone(), Grade::Love));
    assert!(state.media_ready(state.media_generation, Instant::now()));
    assert_eq!(state.grade_admission(), Admission::Deferred);
    let (visit, event, offered) = state.start_view().unwrap();
    let view_attempt = state.view_attempt;
    assert_eq!(state.grade_admission(), Admission::Deferred);
    assert!(!state.start_vote(state.items[1].card.clone(), Grade::Love));
    let mut other_offer = offered.clone();
    other_offer.session_id = "different-offer-of-the-same-post".into();
    assert!(!state.start_vote(other_offer, Grade::Love));
    assert!(state.start_vote(offered.clone(), Grade::Love));
    let vote_attempt = state.vote_attempt;
    assert_eq!(state.grade_admission(), Admission::Unavailable);
    assert!(!state.start_vote(offered.clone(), Grade::Meh));
    state.vote(
        state.generation,
        visit,
        vote_attempt,
        &offered,
        persisted(offered.post_id),
    );
    assert!(
        state.take_navigation().is_none(),
        "presentation acknowledgement still pending"
    );
    state.view(visit, view_attempt, &event, Ok(()), Instant::now());
    assert_eq!(state.grade_admission(), Admission::Ready);
    assert_eq!(state.take_navigation(), Some(NavigationIntent::Next));
    assert!(state.navigate(1));
    assert_eq!(state.grade_admission(), Admission::Unavailable);
    assert!(!state.start_vote(offered, Grade::Love));
    assert!(state.take_navigation().is_none());
}

#[test]
fn failed_presentation_and_rating_hold_intent_and_retry_ids_without_cross_post_grades() {
    let mut state = state();
    assert!(state.media_ready(state.media_generation, Instant::now()));
    let (visit, view_event, offered) = state.start_view().unwrap();
    let view_attempt = state.view_attempt;
    assert!(state.start_vote(offered.clone(), Grade::Love));
    let vote_attempt = state.vote_attempt;
    let vote_event = state.vote_event().map(str::to_owned);
    assert_eq!(
        state.request_navigation(NavigationIntent::Next),
        Admission::Deferred
    );
    state.view(
        visit,
        view_attempt,
        &view_event,
        Err(anyhow::anyhow!("presentation unavailable")),
        Instant::now(),
    );
    state.vote(
        state.generation,
        visit,
        vote_attempt,
        &offered,
        Err(anyhow::anyhow!("presentation unavailable")),
    );
    assert_eq!(state.grade_admission(), Admission::Unavailable);
    assert!(!state.start_vote(offered.clone(), Grade::Love));
    assert!(state.take_navigation().is_none());
    let (retry_visit, retry_view_event, retry_offer) = state.start_view().unwrap();
    let retry_view_attempt = state.view_attempt;
    assert_eq!(
        (retry_visit, retry_view_event.as_str()),
        (visit, view_event.as_str())
    );
    assert_eq!(retry_offer.session_id, offered.session_id);
    assert!(state.start_vote(offered.clone(), Grade::Love));
    let retry_vote_attempt = state.vote_attempt;
    assert_eq!(state.vote_event().map(str::to_owned), vote_event);
    state.view(
        visit,
        retry_view_attempt,
        &view_event,
        Ok(()),
        Instant::now(),
    );
    state.vote(
        state.generation,
        visit,
        retry_vote_attempt,
        &offered,
        Err(anyhow::anyhow!("rating offline")),
    );
    assert!(state.take_navigation().is_none());
    state.retry_media();
    assert_eq!(state.grade_admission(), Admission::Unavailable);
    assert!(
        state.start_vote(offered.clone(), Grade::Love),
        "same accepted rating can retry without reloading media"
    );
    assert_eq!(state.vote_event().map(str::to_owned), vote_event);
    let final_vote_attempt = state.vote_attempt;
    state.vote(
        state.generation,
        visit,
        final_vote_attempt,
        &offered,
        persisted(offered.post_id),
    );
    assert_eq!(state.take_navigation(), Some(NavigationIntent::Next));
    assert!(state.navigate(1));
    assert!(!state.start_vote(offered, Grade::Love));
    assert_eq!(state.current().unwrap().current_grade, None);
    assert!(state.take_navigation().is_none());
}

#[test]
fn correction_removes_only_automatic_navigation_and_replaces_failed_grade_identity() {
    for explicit in [false, true] {
        let mut state = state();
        present(&mut state);
        let offered = state.current().unwrap().clone();
        assert!(state.start_vote(offered.clone(), Grade::Love));
        let first_attempt = state.vote_attempt;
        state.vote(
            state.generation,
            state.visit,
            first_attempt,
            &offered,
            persisted(offered.post_id),
        );
        if explicit {
            assert_eq!(
                state.request_navigation(NavigationIntent::Next),
                Admission::Ready
            );
        }
        let corrected = state.current().unwrap().clone();
        assert!(state.start_vote(corrected.clone(), Grade::Meh));
        let failed_event = state.vote_event().map(str::to_owned);
        let failed_attempt = state.vote_attempt;
        state.vote(
            state.generation,
            state.visit,
            failed_attempt,
            &corrected,
            Err(anyhow::anyhow!("offline")),
        );
        assert!(state.take_navigation().is_none());
        assert!(state.start_vote(corrected.clone(), Grade::Dislike));
        assert_ne!(state.vote_event().map(str::to_owned), failed_event);
        let retry_attempt = state.vote_attempt;
        state.vote(
            state.generation,
            state.visit,
            retry_attempt,
            &corrected,
            Ok(RatingResult {
                post_id: corrected.post_id,
                grade: Grade::Dislike,
                revision: 3,
                changed: true,
            }),
        );
        assert_eq!(
            state.take_navigation(),
            explicit.then_some(NavigationIntent::Next)
        );
        assert_eq!(state.index, 0);
        assert_eq!(state.current().unwrap().current_grade, Some(Grade::Dislike));
    }
}

#[test]
fn stale_vote_reply_cannot_finish_same_grade_or_changed_grade_retry() {
    for changed_grade in [false, true] {
        let mut state = state();
        present(&mut state);
        let generation = state.generation;
        let visit = state.visit;
        let offered = state.current().unwrap().clone();
        assert!(state.start_vote(offered.clone(), Grade::Love));
        let first_attempt = state.vote_attempt;
        let first_event = state.vote_event().map(str::to_owned);
        assert!(!state.start_vote(offered.clone(), Grade::Meh));
        assert_eq!(state.vote_attempt, first_attempt);
        state.vote(
            generation,
            visit,
            first_attempt,
            &offered,
            Err(anyhow::anyhow!("first rating failed")),
        );
        let grade = if changed_grade {
            Grade::Dislike
        } else {
            Grade::Love
        };
        assert!(state.start_vote(offered.clone(), grade));
        let retry_attempt = state.vote_attempt;
        let retry_event = state.vote_event().map(str::to_owned);
        assert_ne!(retry_attempt, first_attempt);
        assert_eq!(retry_event == first_event, !changed_grade);
        assert_eq!(
            state.request_navigation(NavigationIntent::Next),
            Admission::Deferred
        );
        state.vote(
            generation,
            visit,
            first_attempt,
            &offered,
            Err(anyhow::anyhow!("duplicate old failure")),
        );
        assert!(state.voting());
        assert!(state.vote_error().is_empty());
        assert_eq!(state.vote_event().map(str::to_owned), retry_event);
        assert!(state.take_navigation().is_none());
        state.vote(
            generation,
            visit,
            first_attempt,
            &offered,
            Ok(RatingResult {
                post_id: offered.post_id,
                grade: Grade::Love,
                revision: 1,
                changed: true,
            }),
        );
        assert!(
            state.voting(),
            "a useful global revision has no command authority"
        );
        assert_eq!(state.vote_event().map(str::to_owned), retry_event);
        assert_eq!(
            state.acknowledged_grades[&offered.post_id],
            (1, Grade::Love)
        );
        assert_eq!(state.current().unwrap().current_grade, Some(Grade::Love));
        assert!(state.take_navigation().is_none());
        state.vote(
            generation,
            visit,
            retry_attempt,
            &offered,
            Ok(RatingResult {
                post_id: offered.post_id,
                grade,
                revision: 2,
                changed: true,
            }),
        );
        assert!(!state.voting());
        assert!(state.vote_event().map(str::to_owned).is_none());
        assert!(state.failed_vote().is_none());
        assert_eq!(state.current().unwrap().current_grade, Some(grade));
        state.vote(
            generation,
            visit,
            first_attempt,
            &offered,
            Err(anyhow::anyhow!("late failure after retry success")),
        );
        assert!(state.vote_error().is_empty());
        assert_eq!(state.take_navigation(), Some(NavigationIntent::Next));
        assert!(state.navigate(1));
        assert_eq!(state.current().unwrap().current_grade, None);
        assert!(state.take_navigation().is_none());
    }
}

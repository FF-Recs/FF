use super::*;
pub(super) fn card(post_id: i64, session: &str) -> Card {
    Card {
        post_id,
        md5: "0123456789abcdef0123456789abcdef".into(),
        file_extension: "jpg".into(),
        content_rating: "s".into(),
        selection_score: 0.1234567,
        current_grade: None,
        facet: "discover".into(),
        session_id: session.into(),
        epoch: 7,
        position: 123,
        universe_size: 654321,
        model_id: "original-model".into(),
    }
}
pub(super) fn item(post_id: i64, session: &str) -> ViewerItem {
    ViewerItem {
        card: card(post_id, session),
        entry_id: None,
    }
}
pub(super) fn cards_value(items: &[ViewerItem]) -> serde_json::Value {
    serde_json::to_value(items.iter().map(|item| &item.card).collect::<Vec<_>>()).unwrap()
}
pub(super) fn page(session: &str, items: Vec<Card>, exhausted: bool) -> Page {
    Page {
        session_id: session.into(),
        items,
        exhausted,
    }
}
pub(super) fn state() -> ViewerState {
    ViewerState {
        loading: false,
        items: vec![item(i64::MAX, "original"), item(4, "later")],
        ..ViewerState::default()
    }
}
pub(super) fn persisted(post_id: i64) -> anyhow::Result<RatingResult> {
    Ok(RatingResult {
        post_id,
        grade: Grade::Love,
        revision: 2,
        changed: true,
    })
}
pub(super) fn present(state: &mut ViewerState) {
    let now = Instant::now();
    assert!(state.media_ready(state.media_generation, now));
    acknowledge_view(state, now);
}
pub(super) fn acknowledge_view(state: &mut ViewerState, now: Instant) {
    let (visit, event, _) = state.start_view().unwrap();
    let attempt = state.view_attempt;
    state.view(visit, attempt, &event, Ok(()), now);
    assert!(state.viewed());
}
pub(super) fn context(cursor: &str) -> BrowseContext {
    BrowseContext {
        context_id: "durable-context".into(),
        mode: "discover".into(),
        filter: "cat -dog".into(),
        entries: (0..150)
            .map(|id| crate::types::BrowseEntry {
                entry_id: format!("entry-{id}"),
                card: card(id, "original-offer"),
            })
            .collect(),
        cursor: Some(cursor.into()),
    }
}

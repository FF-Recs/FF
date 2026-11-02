use super::support::*;
use super::*;

#[test]
fn contexts_retain_forward_history_and_project_blacklist_without_rewriting_offers() -> Result<()> {
    let (_root, mut backend) = fixture()?;
    let context = backend.activate_context("activate-a", "discover", "", false)?;
    let page = gallery_select(&mut backend, None, "discover", 3)?;
    assert_eq!(page.items.len(), 3);
    let a = &page.items[0];
    let b = &page.items[1];
    backend.presented("visit-a", &context.context_id, None, a)?;
    backend.presented("visit-b", &context.context_id, None, b)?;
    backend.presented("back", &context.context_id, Some("visit-a"), a)?;
    assert!(
        backend
            .presented("branch", &context.context_id, None, &page.items[2])
            .is_err()
    );
    backend.rate("grade-a".into(), a, Grade::Love)?;
    let resumed = backend.activate_context("resume", "discover", "", false)?;
    assert_eq!(resumed.context_id, context.context_id);
    assert_eq!(resumed.cursor.as_deref(), Some("visit-a"));
    assert_eq!(resumed.entries[0].card.current_grade, Some(Grade::Love));
    assert_eq!(resumed.entries[0].card.model_id, a.model_id);
    assert_eq!(resumed.entries[1].card.post_id, b.post_id);
    let row = backend.corpus.find_post(a.post_id).unwrap();
    let blocked = backend
        .catalog
        .tag(backend.corpus.post_tags(row)[0] as usize)
        .name
        .clone();
    backend.save_settings(
        uuid::Uuid::new_v4().to_string(),
        UserSettings {
            blacklist: vec![blocked],
            ..UserSettings::default()
        },
    )?;
    let visible = backend.accessible_context(backend.store.active_context()?.unwrap());
    assert_eq!(
        visible
            .entries
            .iter()
            .map(|entry| entry.card.post_id)
            .collect::<Vec<_>>(),
        vec![b.post_id]
    );
    assert_eq!(visible.cursor.as_deref(), Some("visit-b"));
    assert_eq!(backend.store.active_context()?.unwrap().entries.len(), 2);
    assert!(
        backend
            .presented("blocked-new", &context.context_id, Some("visit-a"), a)
            .is_err()
    );
    backend.presented("visit-a", &context.context_id, None, a)?;
    backend.save_settings(uuid::Uuid::new_v4().to_string(), UserSettings::default())?;
    let restored = backend.accessible_context(backend.store.active_context()?.unwrap());
    assert_eq!(restored.entries.len(), 2);
    assert_eq!(restored.cursor.as_deref(), Some("visit-a"));
    let other = backend.activate_context("other", "balanced", "", false)?;
    assert_ne!(other.context_id, context.context_id);
    assert!(
        backend
            .presented("stale", &context.context_id, Some("visit-b"), b)
            .is_err()
    );
    backend.presented("back", &context.context_id, Some("visit-a"), a)?;
    assert_eq!(
        backend.store.active_context()?.unwrap().context_id,
        other.context_id
    );
    let fresh = backend.activate_context("fresh", "discover", "", true)?;
    assert_ne!(fresh.context_id, context.context_id);
    assert!(fresh.entries.is_empty());
    backend.store.verify_history()?;
    let replayed = backend.store.active_context()?.unwrap();
    assert_eq!(replayed.context_id, fresh.context_id);
    assert_eq!(backend.store.ratings()?[0].grade, Grade::Love);
    Ok(())
}

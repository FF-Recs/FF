use super::support::*;
use super::*;

fn advance_review_counter(backend: &mut Backend, count: usize) -> Result<()> {
    let post_id = backend.corpus.post_ids().iter().copied().max().unwrap() + 1;
    for _ in 0..count {
        let session = uuid::Uuid::new_v4().to_string();
        let mut card = backend.card(
            &backend.generation,
            "rating_review",
            &session,
            0,
            0,
            1,
            0,
            1.0,
        );
        card.post_id = post_id;
        backend.store.create_gallery(
            &session,
            "rating_review",
            "",
            "test-review",
            1,
            &[card],
            false,
        )?;
        backend.store.rate(
            &uuid::Uuid::new_v4().to_string(),
            post_id,
            Grade::Love,
            &session,
            None,
        )?;
    }
    Ok(())
}

#[test]
fn gallery_revalidates_off_page_views_and_empty_eligibility_claims() -> Result<()> {
    let (_root, mut backend) = gallery_fixture(40)?;
    let offered = gallery_select(&mut backend, None, "discover", 5)?;
    let (mut job, _receive) = gallery_job(
        &mut backend,
        Some(gallery_next(&offered, 2, vec![])),
        "discover",
        2,
    )?;
    job.plan(&backend.corpus, &backend.store.path)?;
    assert!(backend.revalidate(&job)?);
    let deferred_row = match &job.task {
        PlanTask::Gallery(selection) => {
            selection.lifecycle.original().plan.as_ref().unwrap().rows[4].0
        }
        _ => unreachable!(),
    };
    let card = backend.card(
        &backend.generation,
        "discover",
        "other",
        0,
        0,
        1,
        deferred_row,
        1.0,
    );
    backend
        .store
        .create_gallery("other", "discover", "", "M0", 1, &[card], false)?;
    backend.store.viewed(
        "other-view",
        backend.corpus.post_ids()[deferred_row],
        "other",
    )?;
    assert!(!backend.revalidate(&job)?);
    job.task.fail(&mut backend, anyhow!("Stale plan"));
    backend.planner = Some(job.state);
    let held: Vec<_> = (1..=40).collect();
    let (mut empty_job, _empty_receive) = gallery_job(
        &mut backend,
        Some(gallery_next(&offered, 2, held)),
        "discover",
        2,
    )?;
    empty_job.plan(&backend.corpus, &backend.store.path)?;
    backend.rate(
        uuid::Uuid::new_v4().to_string(),
        &offered.items[0],
        Grade::Like,
    )?;
    assert!(!backend.revalidate(&empty_job)?);
    empty_job.task.fail(&mut backend, anyhow!("Stale plan"));
    backend.planner = Some(empty_job.state);
    Ok(())
}

#[test]
fn gallery_review_checkpoint_and_cooldown_changes_invalidate_without_grade_deltas() -> Result<()> {
    let (_root, mut backend) = gallery_fixture(6)?;
    seed_review_history(&mut backend)?;
    let first = gallery_select(&mut backend, None, "rating_review", 1)?;
    assert_eq!(first.items.len(), 1);
    let (mut job, _receive) = gallery_job(
        &mut backend,
        Some(gallery_next(&first, 1, vec![first.items[0].post_id])),
        "rating_review",
        1,
    )?;
    job.plan(&backend.corpus, &backend.store.path)?;
    assert!(backend.revalidate(&job)?);
    let deferred = match &job.task {
        PlanTask::Gallery(selection) => {
            selection.lifecycle.original().plan.as_ref().unwrap().rows[2].0
        }
        _ => unreachable!(),
    };
    change_checkpoint(&mut backend, deferred, Some(3))?;
    assert!(backend.pending_grades.is_empty());
    assert!(!backend.revalidate(&job)?);
    job.task.fail(&mut backend, anyhow!("Stale plan"));
    backend.planner = Some(job.state);
    for row in 0..backend.corpus.post_count {
        change_checkpoint(&mut backend, row, Some(0))?;
    }
    let (mut empty, _empty_receive) = gallery_job(
        &mut backend,
        Some(gallery_next(&first, 1, vec![first.items[0].post_id])),
        "rating_review",
        1,
    )?;
    empty.plan(&backend.corpus, &backend.store.path)?;
    assert!(
        matches!(&empty.task, PlanTask::Gallery(selection) if selection.rows.as_ref().unwrap().is_empty())
    );
    assert!(backend.revalidate(&empty)?);
    assert_eq!(backend.store.review_sequence()?, 500);
    let mut counter_card = backend.card(
        &backend.generation,
        "rating_review",
        "boundary-seed",
        0,
        0,
        1,
        0,
        1.0,
    );
    counter_card.post_id = backend.corpus.post_ids().iter().copied().max().unwrap() + 1;
    crate::store::fixtures::review_counter_seed(&mut backend.store, 499, counter_card)?;
    assert_eq!(backend.store.review_sequence()?, 999);
    assert!(backend.revalidate(&empty)?);
    advance_review_counter(&mut backend, 1)?;
    assert_eq!(backend.store.review_sequence()?, 1000);
    assert!(backend.pending_grades.is_empty());
    assert!(!backend.revalidate(&empty)?);
    empty.task.fail(&mut backend, anyhow!("Stale plan"));
    backend.planner = Some(empty.state);
    Ok(())
}

use super::support::*;
use super::*;

#[test]
fn blacklist_alias_validation_is_atomic_and_independent_of_labels() -> Result<()> {
    let (_root, mut backend) = fixture()?;
    backend.catalog = TagCatalog::fixture(
        (0..3)
            .map(|i| Tag {
                name: format!("artist{i}"),
                category: 1,
                post_count: 1,
            })
            .collect(),
        vec![("old_name".into(), 0)],
    )?;
    let cards = refresh_offers(&mut backend)?;
    backend.rate("rating".into(), &cards[0], Grade::Like)?;
    let original = backend.store.ratings()?;
    let generation = backend.generation.clone();
    let settings = backend.save_settings(
        uuid::Uuid::new_v4().to_string(),
        UserSettings {
            blacklist: vec![" OLD_NAME ".into(), "artist0".into()],
            ..UserSettings::default()
        },
    )?;
    assert_eq!(settings.blacklist, ["artist0"]);
    assert!(backend.blacklist.blocks(0) && backend.blacklist.blocks(1));
    assert!(!backend.blacklist.blocks(2));
    for invalid in ["not_a_tag", "+artist1", "artist1 artist2"] {
        assert!(
            backend
                .save_settings(
                    uuid::Uuid::new_v4().to_string(),
                    UserSettings {
                        blacklist: vec![invalid.into()],
                        ..settings.clone()
                    }
                )
                .is_err()
        );
        assert_eq!(backend.store.preferences()?, Some(settings.clone()));
        assert_eq!(backend.settings, settings);
        assert!(backend.blacklist.blocks(0));
    }
    let after = backend.store.ratings()?;
    assert_eq!(after.len(), original.len());
    assert_eq!(after[0].grade, original[0].grade);
    assert_eq!(after[0].revision, original[0].revision);
    assert_eq!(after[0].first_rated_at, original[0].first_rated_at);
    assert_eq!(after[0].context_json, original[0].context_json);
    assert!(Arc::ptr_eq(&generation, &backend.generation));
    assert_eq!(labels(after)[0].post_id, cards[0].post_id);
    Ok(())
}

#[test]
fn blacklist_applies_to_every_mode_and_all_blocked_results_are_exhausted() -> Result<()> {
    for mode in [
        "discover",
        "develop",
        "train_broadly",
        "q3_probe",
        "rating_review",
        "favorites",
        "balanced",
        "fresh",
    ] {
        let (_root, mut backend) = fixture()?;
        let cards = refresh_offers(&mut backend)?;
        backend.rate("seed".into(), &cards[0], Grade::Like)?;
        // A changed rating without a forecast clears its current checkpoint.
        if mode == "rating_review" {
            change_checkpoint(&mut backend, 0, None)?;
        }
        let auto = matches!(mode, "favorites" | "balanced" | "fresh");
        let (before, before_exhausted) = if auto {
            let (reply, mut receive) = oneshot::channel();
            let task = backend
                .prepare(PlanRequest::StartAuto(
                    AutoStartRequest {
                        preset: mode.into(),
                        page_size: 5,
                        seed: 3,
                        filter: String::new(),
                        exclude_post_ids: vec![],
                    },
                    reply,
                ))
                .context("Auto admission")?;
            let mut eligibility = backend.eligibility(&task)?;
            eligibility.changes = backend.pending_grades.drain().collect();
            let mut job = PlanJob {
                task,
                state: backend.planner.take().unwrap(),
                generation: backend.generation.clone(),
                eligibility,
                cancelled: Arc::new(AtomicBool::new(false)),
                _permit: Arc::new(Semaphore::new(1)).try_acquire_owned()?,
            };
            job.plan(&backend.corpus, &backend.store.path)?;
            backend.planner = Some(job.state);
            backend.commit(job.task, &job.generation);
            let page = receive.try_recv()??;
            (page.items, page.exhausted)
        } else {
            let page = gallery_select(&mut backend, None, mode, 5)?;
            (page.items, page.exhausted)
        };
        let expected: &[i64] = match mode {
            "discover" | "fresh" => &[3, 4],
            "develop" => &[2],
            "train_broadly" | "q3_probe" => &[2, 3, 4],
            "rating_review" => &[1],
            _ => &[1, 2, 3, 4],
        };
        let actual: std::collections::BTreeSet<_> =
            before.iter().map(|card| card.post_id).collect();
        assert_eq!(
            actual,
            expected.iter().copied().collect(),
            "Eligible {mode} offers; exhausted={before_exhausted}"
        );
        backend.save_settings(
            uuid::Uuid::new_v4().to_string(),
            UserSettings {
                blacklist: vec!["artist0".into(), "artist1".into(), "artist2".into()],
                ..UserSettings::default()
            },
        )?;
        let (mut job, _receive) = if auto {
            let (mut job, receive) = auto_job(&mut backend, None, 5)?;
            if let PlanTask::Auto(selection) = &mut job.task {
                selection.lifecycle.original_mut().preset = mode.into();
            }
            job.plan(&backend.corpus, &backend.store.path)?;
            assert!(matches!(&job.task, PlanTask::Auto(selection)
                if selection.rows.as_ref().is_some_and(|plan| plan.rows.is_empty() && plan.exhausted)));
            (job, Some(receive))
        } else {
            let (mut job, _receive) = gallery_job(&mut backend, None, mode, 5)?;
            job.plan(&backend.corpus, &backend.store.path)?;
            assert!(matches!(&job.task, PlanTask::Gallery(selection)
                if selection.rows.as_ref().is_some_and(Vec::is_empty)));
            (job, None)
        };
        assert!(backend.revalidate(&job)?);
        backend.save_settings(uuid::Uuid::new_v4().to_string(), UserSettings::default())?;
        assert!(
            !backend.revalidate(&job)?,
            "An old empty/exhausted {mode} claim must not survive unblocking"
        );
        backend.reclaim_changes(&mut job);
        backend.planner = Some(job.state);
        job.task.fail(&mut backend, anyhow!("Test finished"));
        let grades = backend.store.ratings()?;
        assert_eq!(grades.len(), 1);
        assert_eq!(grades[0].grade, Grade::Like);
    }
    Ok(())
}

#[test]
fn preference_retries_do_not_republish_an_obsolete_blacklist() -> Result<()> {
    let (_root, mut backend) = fixture()?;
    let first = backend.save_settings(
        "preferences-a".into(),
        UserSettings {
            blacklist: vec!["artist0".into()],
            ..UserSettings::default()
        },
    )?;
    let latest = backend.save_settings(
        "preferences-b".into(),
        UserSettings {
            blacklist: vec!["artist1".into()],
            ..UserSettings::default()
        },
    )?;
    assert_eq!(
        backend.save_settings("preferences-a".into(), first)?,
        latest
    );
    assert_eq!(backend.store.preferences()?, Some(latest.clone()));
    assert_eq!(backend.settings, latest);
    assert!(!backend.blacklist.blocks(0));
    assert!(backend.blacklist.blocks(2));
    Ok(())
}

#[test]
fn policy_changes_reject_pending_offers_and_invalidate_cached_admission() -> Result<()> {
    fn recheck<R>(backend: &mut Backend, mut job: PlanJob, _reply: R) -> Result<()> {
        job.plan(&backend.corpus, &backend.store.path)?;
        assert!(backend.revalidate(&job)?);
        let has_offers = match &job.task {
            PlanTask::Gallery(selection) => !selection.rows.as_ref().unwrap().is_empty(),
            PlanTask::Auto(selection) => !selection.rows.as_ref().unwrap().rows.is_empty(),
        };
        assert!(has_offers, "Exercise a real pending offered page");
        backend.save_settings(
            "policy".into(),
            UserSettings {
                blacklist: vec!["artist0".into(), "artist1".into(), "artist2".into()],
                ..UserSettings::default()
            },
        )?;
        assert!(
            !backend.revalidate(&job)?,
            "Old offers cannot commit after a policy change"
        );
        job.eligibility = backend.eligibility(&job.task)?;
        job.plan(&backend.corpus, &backend.store.path)?;
        match &job.task {
            PlanTask::Gallery(selection) => {
                assert!(selection.rows.as_ref().unwrap().is_empty())
            }
            PlanTask::Auto(selection) => {
                assert!(selection.rows.as_ref().unwrap().rows.is_empty())
            }
        }
        assert!(backend.revalidate(&job)?);
        backend.planner = Some(job.state);
        job.task.fail(backend, anyhow!("Test finished"));
        Ok(())
    }
    for mode in ["discover", "rating_review", "balanced"] {
        let (_root, mut backend) = fixture()?;
        if mode == "rating_review" {
            let cards = refresh_offers(&mut backend)?;
            backend.rate("grade".into(), &cards[0], Grade::Like)?;
            change_checkpoint(&mut backend, 0, None)?;
        }
        if mode == "balanced" {
            let (job, reply) = auto_job(&mut backend, None, 5)?;
            recheck(&mut backend, job, reply)?;
        } else {
            let (job, reply) = gallery_job(&mut backend, None, mode, 5)?;
            recheck(&mut backend, job, reply)?;
        }
    }
    Ok(())
}

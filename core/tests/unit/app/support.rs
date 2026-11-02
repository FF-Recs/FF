use super::*;

pub(super) fn fixture() -> Result<(tempfile::TempDir, Backend)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.workdir/ff-rs/app");
    std::fs::create_dir_all(&root)?;
    let temporary = tempfile::tempdir_in(root)?;
    let config = AppConfig {
        corpus_path: temporary.path().join("corpus"),
        interaction_path: temporary.path().join("interactions.sqlite3"),
        settings_path: temporary.path().join("config.toml"),
    };
    let corpus = Arc::new(Corpus::fixture(
        &[1, 2, 3, 4],
        &[vec![0], vec![0], vec![1], vec![2]],
        3,
        &[0, 1, 2],
        &[],
    ));
    let catalog = TagCatalog::fixture(
        (0..3)
            .map(|i| Tag {
                name: format!("artist{i}"),
                category: 1,
                post_count: 1,
            })
            .collect(),
        vec![],
    )?;
    let store = Store::open(&config.interaction_path)?;
    let familiarity = Familiarity::build(&corpus, &[]);
    let backend = Backend {
        corpus,
        catalog,
        store,
        settings: UserSettings::default(),
        blacklist: Arc::new(Blacklist::default()),
        generation: Arc::new(FitResult {
            model_id: "test-generation".into(),
            status: model::ModelStatus {
                route: model::ModelRoute::Ensemble,
                fallback_reason: None,
            },
            scores: Some(ScoreBuffers {
                not_dislike: vec![0.99; 4],
                like_or_love: vec![0.95; 4],
                love: vec![0.8; 4],
                expected_grade: vec![2.74; 4],
            }),
        }),
        planner: Some(Box::new(PlannerState {
            familiarity,
            grades: vec![None; 4],
            admission_revision: 0,
            rebuild: false,
        })),
        live_artist_counts: vec![0; 3],
        grades: vec![None; 4],
        rated_rows: vec![],
        pending_grades: HashMap::new(),
        rating_count: 0,
        auto: None,
        gallery: None,
        planning: None,
        pool: Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build()?),
        refresh: Refresh::default(),
    };
    Ok((temporary, backend))
}
pub(super) fn gallery_fixture(count: usize) -> Result<(tempfile::TempDir, Backend)> {
    let (temporary, mut backend) = fixture()?;
    let ids: Vec<_> = (1..=count as i64).collect();
    let tags: Vec<_> = (0..count as i32).map(|tag| vec![tag]).collect();
    let artists: Vec<_> = (0..count as i32).collect();
    backend.corpus = Arc::new(Corpus::fixture(&ids, &tags, count, &artists, &[]));
    backend.catalog = TagCatalog::fixture(
        (0..count)
            .map(|i| Tag {
                name: format!("artist{i}"),
                category: 1,
                post_count: 1,
            })
            .collect(),
        vec![],
    )?;
    backend.generation = Arc::new(FitResult {
        model_id: "M0".into(),
        status: backend.generation.status.clone(),
        scores: Some(ScoreBuffers {
            not_dislike: vec![0.99; count],
            like_or_love: vec![0.95; count],
            love: vec![0.8; count],
            expected_grade: vec![2.74; count],
        }),
    });
    backend.planner = Some(Box::new(PlannerState {
        familiarity: Familiarity::build(&backend.corpus, &[]),
        grades: vec![None; count],
        admission_revision: 0,
        rebuild: false,
    }));
    backend.grades = vec![None; count];
    backend.live_artist_counts = vec![0; count];
    Ok((temporary, backend))
}

pub(super) fn seed_review_history(backend: &mut Backend) -> Result<()> {
    let path = backend.store.path.with_file_name("review-history.sqlite3");
    backend.store = crate::store::fixtures::review_history(&path, backend.corpus.post_ids(), 500)?;
    let ratings = backend.store.ratings()?;
    backend.rating_count = ratings.len();
    backend.grades.fill(None);
    backend.rated_rows.clear();
    for rating in &ratings {
        if let Some(row) = backend.corpus.find_post(rating.post_id) {
            backend.grades[row] = Some(rating.grade);
            backend.rated_rows.push(row);
        }
    }
    let familiarity = Familiarity::build(&backend.corpus, &ratings);
    backend.live_artist_counts = familiarity.rated_count.clone();
    let planner = backend.planner.as_mut().context("Seed test planner")?;
    planner.familiarity = familiarity;
    planner.grades.clone_from(&backend.grades);
    backend.pending_grades.clear();
    Ok(())
}

pub(super) fn change_checkpoint(
    backend: &mut Backend,
    row: usize,
    predicted_grade: Option<u8>,
) -> Result<()> {
    let session = uuid::Uuid::new_v4().to_string();
    let card = backend.card(&backend.generation, "discover", &session, 0, 0, 1, row, 1.0);
    backend.store.create_gallery(
        &session,
        "discover",
        "",
        "test-checkpoint",
        1,
        &[card],
        false,
    )?;
    backend.store.rate(
        &uuid::Uuid::new_v4().to_string(),
        backend.corpus.post_ids()[row],
        Grade::Like,
        &session,
        predicted_grade.map(|predicted_grade| RatingPrediction {
            predicted_grade,
            model_id: "test-checkpoint".into(),
        }),
    )?;
    Ok(())
}

pub(super) fn gallery_job(
    backend: &mut Backend,
    next: Option<GalleryNextRequest>,
    facet: &str,
    size: usize,
) -> Result<(PlanJob, oneshot::Receiver<Result<Gallery>>)> {
    let (reply, receive) = oneshot::channel();
    let request = match next {
        Some(request) => PlanRequest::NextGallery(request, reply),
        None => PlanRequest::Gallery(
            GalleryStartRequest {
                facet: facet.into(),
                page_size: size,
                filter: String::new(),
                exclude_post_ids: vec![],
            },
            reply,
        ),
    };
    let task = backend
        .prepare(request)
        .context("Test selection admission failed")?;
    let mut eligibility = backend.eligibility(&task)?;
    eligibility.changes = backend.pending_grades.drain().collect();
    Ok((
        PlanJob {
            task,
            state: backend
                .planner
                .take()
                .context("Missing Gallery test planner")?,
            generation: backend.generation.clone(),
            eligibility,
            cancelled: Arc::new(AtomicBool::new(false)),
            _permit: Arc::new(Semaphore::new(1)).try_acquire_owned()?,
        },
        receive,
    ))
}

pub(super) fn gallery_next(page: &Gallery, size: usize, retained: Vec<i64>) -> GalleryNextRequest {
    GalleryNextRequest {
        session_id: page.session_id.clone(),
        page_size: size,
        exclude_post_ids: retained,
    }
}

pub(super) fn gallery_select(
    backend: &mut Backend,
    next: Option<GalleryNextRequest>,
    facet: &str,
    size: usize,
) -> Result<Gallery> {
    let (mut job, mut receive) = gallery_job(backend, next, facet, size)?;
    let result = job.plan(&backend.corpus, &backend.store.path);
    backend.reclaim_changes(&mut job);
    let result = result.and_then(|_| {
        ensure!(backend.revalidate(&job)?, "Stale Gallery test plan");
        Ok(())
    });
    backend.planner = Some(job.state);
    match result {
        Ok(()) => backend.commit(job.task, &job.generation),
        Err(error) => job.task.fail(backend, error),
    }
    receive.try_recv().context("Gallery test reply")?
}

pub(super) fn auto_job(
    backend: &mut Backend,
    next: Option<AutoNextRequest>,
    size: usize,
) -> Result<(PlanJob, oneshot::Receiver<Result<AutoPage>>)> {
    let (reply, receive) = oneshot::channel();
    let request = match next {
        Some(request) => PlanRequest::NextAuto(request, reply),
        None => PlanRequest::StartAuto(
            AutoStartRequest {
                preset: "balanced".into(),
                page_size: size,
                seed: 7,
                filter: String::new(),
                exclude_post_ids: vec![],
            },
            reply,
        ),
    };
    let task = backend
        .prepare(request)
        .context("Test selection admission failed")?;
    let mut eligibility = backend.eligibility(&task)?;
    eligibility.changes = backend.pending_grades.drain().collect();
    Ok((
        PlanJob {
            task,
            state: backend.planner.take().context("Missing test planner")?,
            generation: backend.generation.clone(),
            eligibility,
            cancelled: Arc::new(AtomicBool::new(false)),
            _permit: Arc::new(Semaphore::new(1)).try_acquire_owned()?,
        },
        receive,
    ))
}

pub(super) fn launch(backend: Backend) -> Result<AppHandle> {
    let (sender, receiver) = mpsc::channel(64);
    let weak = sender.downgrade();
    let planning = Arc::new(Semaphore::new(1));
    let gate = planning.clone();
    let runtime = tokio::runtime::Handle::current();
    std::thread::spawn(move || backend.run(receiver, weak, runtime, gate));
    Ok(AppHandle { sender, planning })
}
// Capture real worker/timer messages at the actor boundary. Tests decide
// when to deliver them, without adding synchronization to production fits.
pub(super) struct ObservedWorkers {
    pub(super) app: AppHandle,
    pub(super) receiver: mpsc::Receiver<Command>,
    _sender: mpsc::Sender<Command>,
}
impl ObservedWorkers {
    pub(super) async fn next(&mut self) -> Result<Command> {
        tokio::time::timeout(std::time::Duration::from_secs(5), self.receiver.recv())
            .await?
            .context("Worker channel closed")
    }
}
pub(super) fn launch_observed(backend: Backend) -> Result<ObservedWorkers> {
    let (sender, receiver) = mpsc::channel(64);
    let (worker_sender, workers) = mpsc::channel(64);
    let weak = worker_sender.downgrade();
    let planning = Arc::new(Semaphore::new(1));
    let gate = planning.clone();
    let runtime = tokio::runtime::Handle::current();
    std::thread::spawn(move || backend.run(receiver, weak, runtime, gate));
    Ok(ObservedWorkers {
        app: AppHandle { sender, planning },
        receiver: workers,
        _sender: worker_sender,
    })
}
pub(super) fn refresh_offers(backend: &mut Backend) -> Result<Vec<Card>> {
    let offered: Vec<_> = (0..4)
        .map(|row| {
            backend.card(
                &backend.generation,
                "discover",
                "earlier",
                0,
                row,
                4,
                row,
                2.74,
            )
        })
        .collect();
    backend.store.create_gallery(
        "earlier",
        "discover",
        "",
        "test-generation",
        4,
        &offered,
        false,
    )?;
    Ok(offered)
}
pub(super) async fn block_compute(
    pool: &rayon::ThreadPool,
) -> Result<std::sync::mpsc::SyncSender<()>> {
    let (release, wait) = std::sync::mpsc::sync_channel(1);
    let (started, ready) = oneshot::channel();
    pool.spawn(move || {
        let _ = started.send(());
        let _ = wait.recv();
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), ready).await??;
    Ok(release)
}

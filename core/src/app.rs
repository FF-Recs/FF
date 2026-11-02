use crate::{
    catalog::TagCatalog,
    corpus::Corpus,
    model::{self, FitResult, TrainingLabel},
    selection::{self, Blacklist, Familiarity, TagFilter},
    settings,
    store::Store,
    types::*,
};
use anyhow::{Context, Result, anyhow, ensure};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub corpus_path: PathBuf,
    pub interaction_path: PathBuf,
    pub settings_path: PathBuf,
}
#[derive(Clone)]
pub struct AppHandle {
    sender: mpsc::Sender<Command>,
    planning: Arc<Semaphore>,
}
type Reply<T> = oneshot::Sender<Result<T>>;
enum Command {
    Health(Reply<Health>),
    Settings(Reply<UserSettings>),
    SaveSettings(String, UserSettings, Reply<UserSettings>),
    RestoreContext(Reply<Option<BrowseContext>>),
    ActivateContext(String, String, String, bool, Reply<BrowseContext>),
    Presented(String, String, Option<String>, Card, Reply<()>),
    Facets(Reply<Vec<Facet>>),
    Search(String, usize, Reply<Vec<TagSuggestion>>),
    Plan(PlanRequest, OwnedSemaphorePermit),
    Planned(Box<PlanCompletion>),
    Rate(String, Card, Grade, Reply<RatingResult>),
    Viewed(String, Card, Reply<()>),
    StartFit(u64),
    Fitted(u64, Result<Option<Arc<FitResult>>>),
    Shutdown(Reply<()>),
}
enum PlanRequest {
    Gallery(GalleryStartRequest, Reply<Gallery>),
    NextGallery(GalleryNextRequest, Reply<Gallery>),
    StartAuto(AutoStartRequest, Reply<AutoPage>),
    NextAuto(AutoNextRequest, Reply<AutoPage>),
}
impl PlanRequest {
    fn is_closed(&self) -> bool {
        match self {
            Self::Gallery(_, reply) | Self::NextGallery(_, reply) => reply.is_closed(),
            Self::StartAuto(_, reply) | Self::NextAuto(_, reply) => reply.is_closed(),
        }
    }
    fn fail(self, error: anyhow::Error) {
        match self {
            Self::Gallery(_, reply) | Self::NextGallery(_, reply) => {
                let _ = reply.send(Err(error));
            }
            Self::StartAuto(_, reply) | Self::NextAuto(_, reply) => {
                let _ = reply.send(Err(error));
            }
        }
    }
}
impl AppHandle {
    pub async fn open(config: AppConfig) -> Result<Self> {
        let runtime = tokio::runtime::Handle::current();
        let backend = tokio::task::spawn_blocking(move || Backend::open(config))
            .await
            .context("Initialize native backend task")??;
        let (sender, receiver) = mpsc::channel(64);
        let weak = sender.downgrade();
        let planning = Arc::new(Semaphore::new(1));
        let gate = planning.clone();
        std::thread::Builder::new()
            .name("ff-application".into())
            .spawn(move || backend.run(receiver, weak, runtime, gate))?;
        Ok(Self { sender, planning })
    }
    async fn request<T>(&self, make: impl FnOnce(Reply<T>) -> Command) -> Result<T> {
        let (send, receive) = oneshot::channel();
        self.sender
            .send(make(send))
            .await
            .map_err(|_| anyhow!("Application has shut down"))?;
        receive
            .await
            .context("Application worker stopped before replying")?
    }
    async fn plan<T>(&self, make: impl FnOnce(Reply<T>) -> PlanRequest) -> Result<T> {
        // The permit travels with the job, not the caller's future. Dropping an
        // obsolete request cannot admit another dense working set prematurely.
        let permit = self
            .planning
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| anyhow!("Application is shutting down"))?;
        self.request(|reply| Command::Plan(make(reply), permit))
            .await
    }
    pub async fn health(&self) -> Result<Health> {
        self.request(Command::Health).await
    }
    pub async fn settings(&self) -> Result<UserSettings> {
        self.request(Command::Settings).await
    }
    pub async fn save_settings(
        &self,
        event_id: String,
        value: UserSettings,
    ) -> Result<UserSettings> {
        self.request(|r| Command::SaveSettings(event_id, value, r))
            .await
    }
    pub async fn restore_context(&self) -> Result<Option<BrowseContext>> {
        self.request(Command::RestoreContext).await
    }
    pub async fn activate_context(
        &self,
        event_id: String,
        mode: String,
        filter: String,
        fresh: bool,
    ) -> Result<BrowseContext> {
        self.request(|r| Command::ActivateContext(event_id, mode, filter, fresh, r))
            .await
    }
    pub async fn presented(
        &self,
        event_id: String,
        context_id: String,
        entry_id: Option<String>,
        card: Card,
    ) -> Result<()> {
        self.request(|r| Command::Presented(event_id, context_id, entry_id, card, r))
            .await
    }
    pub async fn facets(&self) -> Result<Vec<Facet>> {
        self.request(Command::Facets).await
    }
    pub async fn search_tags(&self, query: String, limit: usize) -> Result<Vec<TagSuggestion>> {
        self.request(|r| Command::Search(query, limit, r)).await
    }
    pub async fn start_gallery(&self, value: GalleryStartRequest) -> Result<Gallery> {
        self.plan(|r| PlanRequest::Gallery(value, r)).await
    }
    pub async fn next_gallery(&self, value: GalleryNextRequest) -> Result<Gallery> {
        self.plan(|r| PlanRequest::NextGallery(value, r)).await
    }
    pub async fn start_auto(&self, value: AutoStartRequest) -> Result<AutoPage> {
        self.plan(|r| PlanRequest::StartAuto(value, r)).await
    }
    pub async fn next_auto(&self, value: AutoNextRequest) -> Result<AutoPage> {
        self.plan(|r| PlanRequest::NextAuto(value, r)).await
    }
    pub async fn rate(&self, event_id: String, card: Card, grade: Grade) -> Result<RatingResult> {
        self.request(|r| Command::Rate(event_id, card, grade, r))
            .await
    }
    pub async fn viewed(&self, event_id: String, card: Card) -> Result<()> {
        self.request(|r| Command::Viewed(event_id, card, r)).await
    }
    pub async fn shutdown(&self) -> Result<()> {
        self.request(Command::Shutdown).await
    }
}

struct GalleryPlan {
    generation: Weak<FitResult>,
    revision: u64,
    eligibility: [u8; 32],
    blacklist: Weak<Blacklist>,
    rows: std::collections::VecDeque<(usize, f32)>,
}
struct GalleryState {
    session: String,
    facet: String,
    filter: TagFilter,
    capacity: usize,
    issued: Vec<usize>,
    complete: bool,
    plan: Option<GalleryPlan>,
}
impl GalleryState {
    fn new(facet: String, filter: TagFilter, capacity: usize) -> Self {
        Self {
            session: format!("gallery-{}", uuid::Uuid::new_v4().simple()),
            facet,
            filter,
            capacity,
            issued: vec![],
            complete: false,
            plan: None,
        }
    }
    fn owner(&self) -> crate::store::GalleryOwner<'_> {
        crate::store::GalleryOwner {
            session: &self.session,
            facet: &self.facet,
            filter: &self.filter.canonical,
            capacity: self.capacity,
            issued: self.issued.len(),
        }
    }
}
struct AutoState {
    session: String,
    preset: String,
    filter: TagFilter,
    seed: i32,
    epoch: u32,
    candidates: Vec<usize>,
    candidate_generation: Option<Weak<FitResult>>,
    candidate_revision: u64,
    candidate_blacklist: Weak<Blacklist>,
    emitted: HashSet<usize>,
}
impl AutoState {
    fn new(preset: String, filter: TagFilter, seed: i32, epoch: u32) -> Self {
        Self {
            session: format!("auto-{}", uuid::Uuid::new_v4().simple()),
            preset,
            filter,
            seed,
            epoch,
            candidates: vec![],
            candidate_generation: None,
            candidate_revision: 0,
            candidate_blacklist: Weak::new(),
            emitted: HashSet::new(),
        }
    }
}
struct PlannerState {
    familiarity: Familiarity,
    grades: Vec<Option<Grade>>,
    // Shared synchronization may happen in a Gallery job before Auto resumes.
    admission_revision: u64,
    rebuild: bool,
}
impl PlannerState {
    fn synchronize(
        &mut self,
        corpus: &Corpus,
        changes: &mut Vec<(usize, Grade)>,
        path: &std::path::Path,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<()> {
        if self.rebuild {
            // Only recovery from a worker panic rereads full private history.
            let ratings = Store::snapshot_ratings_at(path)?;
            self.admission_revision = self
                .admission_revision
                .checked_add(1)
                .context("Selection admission revision overflow")?;
            self.familiarity = Familiarity::build(corpus, &ratings);
            self.grades.fill(None);
            for rating in ratings {
                if let Some(row) = corpus.find_post(rating.post_id) {
                    self.grades[row] = Some(rating.grade);
                }
            }
            changes.clear();
            self.rebuild = false;
        }
        while let Some(&(row, grade)) = changes.last() {
            ensure!(!cancelled(), "Selection cancelled");
            let previous = self.grades[row];
            if previous != Some(grade) {
                self.admission_revision = self
                    .admission_revision
                    .checked_add(1)
                    .context("Selection admission revision overflow")?;
                self.familiarity.update(corpus, row, previous, grade);
                self.grades[row] = Some(grade);
            }
            changes.pop();
        }
        Ok(())
    }
}
struct Eligibility {
    changes: Vec<(usize, Grade)>,
    review_rows: Vec<usize>,
    unavailable: HashSet<i64>,
    recent: Vec<usize>,
    checkpoints: HashMap<i64, Checkpoint>,
    review_exclusions: HashSet<i64>,
    review_sequence: i64,
    blacklist: Arc<Blacklist>,
}
/// The original continuation is always retained until a durable commit succeeds.
enum SelectionState<S> {
    Start(S),
    Continue { original: S, proposed: Option<S> },
}
impl<S> SelectionState<S> {
    fn original(&self) -> &S {
        match self {
            Self::Start(state)
            | Self::Continue {
                original: state, ..
            } => state,
        }
    }
    fn original_mut(&mut self) -> &mut S {
        match self {
            Self::Start(state)
            | Self::Continue {
                original: state, ..
            } => state,
        }
    }
    fn is_continuation(&self) -> bool {
        matches!(self, Self::Continue { .. })
    }
    fn proposed(&self) -> Option<&S> {
        match self {
            Self::Start(_) => None,
            Self::Continue { proposed, .. } => proposed.as_ref(),
        }
    }
    fn selected(&self) -> &S {
        self.proposed().unwrap_or_else(|| self.original())
    }
    fn selected_mut(&mut self) -> &mut S {
        match self {
            Self::Start(state) => state,
            Self::Continue { original, proposed } => proposed.as_mut().unwrap_or(original),
        }
    }
    fn take_proposal(&mut self) -> Option<S> {
        match self {
            Self::Start(_) => None,
            Self::Continue { proposed, .. } => proposed.take(),
        }
    }
    fn propose(&mut self, state: S) {
        if let Self::Continue { proposed, .. } = self {
            *proposed = Some(state);
        }
    }
    fn into_selected(self, use_proposal: bool) -> S {
        match self {
            Self::Start(state) => state,
            Self::Continue { original, proposed } => {
                if use_proposal {
                    proposed.unwrap_or(original)
                } else {
                    original
                }
            }
        }
    }
}
struct Selection<S, R, P> {
    lifecycle: SelectionState<S>,
    size: usize,
    retained: HashSet<i64>,
    rows: Option<R>,
    reply: Reply<P>,
}
impl<S, R, P> Selection<S, R, P> {
    fn admit(
        input: Result<(SelectionState<S>, usize, HashSet<i64>)>,
        reply: Reply<P>,
    ) -> Option<Self> {
        match input {
            Ok((lifecycle, size, retained)) => Some(Self {
                lifecycle,
                size,
                retained,
                rows: None,
                reply,
            }),
            Err(error) => {
                let _ = reply.send(Err(error));
                None
            }
        }
    }
}
enum PlanTask {
    Gallery(Selection<GalleryState, Vec<(usize, f32)>, Gallery>),
    Auto(Selection<AutoState, selection::AutoPlan, AutoPage>),
}
impl PlanTask {
    fn is_closed(&self) -> bool {
        match self {
            Self::Gallery(selection) => selection.reply.is_closed(),
            Self::Auto(selection) => selection.reply.is_closed(),
        }
    }
    fn fail(self, backend: &mut Backend, error: anyhow::Error) {
        match self {
            Self::Gallery(selection) => {
                if selection.lifecycle.is_continuation() {
                    backend.gallery = Some(selection.lifecycle.into_selected(false));
                }
                let _ = selection.reply.send(Err(error));
            }
            Self::Auto(selection) => {
                if selection.lifecycle.is_continuation() {
                    backend.auto = Some(selection.lifecycle.into_selected(false));
                }
                let _ = selection.reply.send(Err(error));
            }
        }
    }
}
struct PlanJob {
    task: PlanTask,
    state: Box<PlannerState>,
    generation: Arc<FitResult>,
    eligibility: Eligibility,
    cancelled: Arc<AtomicBool>,
    _permit: OwnedSemaphorePermit,
}
struct PlanCompletion {
    job: PlanJob,
    result: Result<()>,
}
struct Backend {
    corpus: Arc<Corpus>,
    catalog: TagCatalog,
    store: Store,
    settings: UserSettings,
    blacklist: Arc<Blacklist>,
    generation: Arc<FitResult>,
    planner: Option<Box<PlannerState>>,
    live_artist_counts: Vec<i32>,
    grades: Vec<Option<Grade>>,
    rated_rows: Vec<usize>,
    pending_grades: HashMap<usize, Grade>,
    rating_count: usize,
    auto: Option<AutoState>,
    gallery: Option<GalleryState>,
    planning: Option<Arc<AtomicBool>>,
    pool: Arc<rayon::ThreadPool>,
    refresh: Refresh,
}
#[derive(Default)]
struct Refresh {
    requested: u64,
    // Completed attempts, including failures, not successful publication or a
    // ratings snapshot watermark. The serving generation owns published state.
    attempted_through: u64,
    in_flight: Option<u64>,
    debounce: Option<tokio::task::JoinHandle<()>>,
    error: Option<String>,
}
impl Refresh {
    fn refreshing(&self) -> bool {
        self.in_flight.is_some() || self.attempted_through < self.requested
    }
    fn request(&mut self, runtime: &tokio::runtime::Handle, sender: &mpsc::WeakSender<Command>) {
        self.requested += 1;
        self.schedule(runtime, sender);
    }
    fn schedule(&mut self, runtime: &tokio::runtime::Handle, sender: &mpsc::WeakSender<Command>) {
        if self.in_flight.is_some() || self.attempted_through >= self.requested {
            return;
        }
        self.cancel_debounce();
        let request = self.requested;
        let sender = sender.clone();
        self.debounce = Some(runtime.spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            if let Some(sender) = sender.upgrade() {
                let _ = sender.send(Command::StartFit(request)).await;
            }
        }));
    }
    fn admit(&mut self, request: u64) -> bool {
        if self.in_flight.is_some()
            || request != self.requested
            || request <= self.attempted_through
        {
            return false;
        }
        self.cancel_debounce();
        self.in_flight = Some(request);
        true
    }
    fn complete(&mut self, request: u64) -> bool {
        if self.in_flight != Some(request) {
            return false;
        }
        self.in_flight = None;
        self.attempted_through = request;
        true
    }
    fn cancel_debounce(&mut self) {
        if let Some(task) = self.debounce.take() {
            task.abort();
        }
    }
}
fn session(json: &str) -> String {
    serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|v| {
            ["auto_session_id", "gallery_session_id", "session_id"]
                .iter()
                .find_map(|key| {
                    v.get(key)
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.trim().is_empty())
                        .map(str::to_owned)
                })
        })
        .unwrap_or_default()
}
fn labels(ratings: Vec<StoredRating>) -> Vec<TrainingLabel> {
    ratings
        .into_iter()
        .map(|r| TrainingLabel {
            post_id: r.post_id,
            grade: r.grade as u8,
            session_id: session(&r.context_json),
            updated_at: r.updated_at,
        })
        .collect()
}
fn artist_count(corpus: &Corpus, counts: &[i32], row: usize) -> i32 {
    corpus
        .post_artists(row)
        .iter()
        .map(|&artist| counts[artist as usize])
        .max()
        .unwrap_or(0)
}
fn review_eligible(
    corpus: &Corpus,
    generation: &FitResult,
    eligibility: &Eligibility,
    row: usize,
) -> bool {
    let Some(scores) = generation.scores.as_ref() else {
        return false;
    };
    let id = corpus.post_ids()[row];
    !eligibility.blacklist.blocks(row)
        && !eligibility.review_exclusions.contains(&id)
        && eligibility.checkpoints.get(&id).is_none_or(|c| {
            eligibility
                .review_sequence
                .saturating_sub(c.review_sequence)
                >= 500
                && c.predicted_grade != selection::median_grade(scores, row)
        })
}
fn filtered_rows(
    corpus: &Corpus,
    filter: &TagFilter,
    accept: impl Fn(usize) -> bool,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<usize>> {
    let phase = Instant::now();
    let mut rows = Vec::new();
    let mut visit = |offset: usize, row: usize| -> Result<()> {
        if offset % 4096 == 0 {
            ensure!(!cancelled(), "Selection cancelled");
        }
        if accept(row) && filter.matches(corpus, row) {
            rows.push(row);
        }
        Ok(())
    };
    if let Some(&tag) = filter
        .required
        .iter()
        .min_by_key(|&&tag| corpus.tag_posts(tag).len())
    {
        for (offset, &row) in corpus.tag_posts(tag).iter().enumerate() {
            visit(offset, row as usize)?;
        }
    } else {
        for row in 0..corpus.post_count {
            visit(row, row)?;
        }
    }
    tracing::info!(
        phase = "candidate_filter",
        candidates = rows.len(),
        elapsed_seconds = phase.elapsed().as_secs_f64(),
        "Selection phase"
    );
    Ok(rows)
}
fn gallery_eligibility_delta(digest: &mut [u8; 32], id: i64, remove: bool) {
    // Add/subtract SHA-256 identities modulo 2^256. Unlike XOR, this does
    // not admit cheap linear-dependency collisions among exclusion sets.
    let mut carry = 0i16;
    for (output, byte) in digest.iter_mut().zip(Sha256::digest(id.to_le_bytes())) {
        let byte = i16::from(byte);
        let value = i16::from(*output) + if remove { -byte } else { byte } + carry;
        *output = value as u8;
        carry = value.div_euclid(256);
    }
}

fn gallery_eligibility(
    corpus: &Corpus,
    generation: &FitResult,
    eligibility: &Eligibility,
    state: &GalleryState,
    retained: &HashSet<i64>,
) -> [u8; 32] {
    // Order-independent metadata digests avoid copying/sorting the corpus-sized
    // exclusion set. Issued rows are already accounted for by this cohort.
    let mut digest = [0u8; 32];
    let issued: HashSet<_> = state.issued.iter().copied().collect();
    let mut include = |id: i64| gallery_eligibility_delta(&mut digest, id, false);
    if state.facet == "rating_review" {
        for &row in &eligibility.review_rows {
            let id = corpus.post_ids()[row];
            if !issued.contains(&row)
                && !retained.contains(&id)
                && state.filter.matches(corpus, row)
                && review_eligible(corpus, generation, eligibility, row)
            {
                include(id);
            }
        }
    } else {
        for &id in eligibility.unavailable.iter().chain(
            retained
                .iter()
                .filter(|id| !eligibility.unavailable.contains(*id)),
        ) {
            if let Some(row) = corpus.find_post(id)
                && !issued.contains(&row)
                && state.filter.matches(corpus, row)
            {
                include(id);
            }
        }
    }
    digest
}

impl PlanJob {
    fn plan(&mut self, corpus: &Corpus, path: &std::path::Path) -> Result<()> {
        let cancelled = || self.cancelled.load(Ordering::Relaxed) || self.task.is_closed();
        ensure!(!cancelled(), "Selection cancelled");
        self.state
            .synchronize(corpus, &mut self.eligibility.changes, path, &cancelled)?;
        let phase = Instant::now();
        match &mut self.task {
            PlanTask::Gallery(selection) => {
                let Selection {
                    lifecycle,
                    size,
                    retained,
                    rows: output,
                    reply,
                } = selection;
                let cancelled = || self.cancelled.load(Ordering::Relaxed) || reply.is_closed();
                *output = None;
                let next = lifecycle.is_continuation();
                // Replanning never consumes a speculative prefix.
                lifecycle.take_proposal();
                let state = lifecycle.original_mut();
                let rank = |state: &mut GalleryState| -> Result<Vec<(usize, f32)>> {
                    let eligibility = gallery_eligibility(
                        corpus,
                        &self.generation,
                        &self.eligibility,
                        state,
                        retained,
                    );
                    let reusable = state.plan.as_ref().is_some_and(|plan| {
                        plan.generation.as_ptr() == Arc::as_ptr(&self.generation)
                            && plan.revision == self.state.admission_revision
                            && plan.eligibility == eligibility
                            && plan.blacklist.as_ptr() == Arc::as_ptr(&self.eligibility.blacklist)
                    });
                    if !reusable {
                        let issued: HashSet<_> = state.issued.iter().copied().collect();
                        let candidates = if state.facet == "rating_review" {
                            self.eligibility
                                .review_rows
                                .iter()
                                .copied()
                                .filter(|&row| {
                                    !issued.contains(&row)
                                        && !retained.contains(&corpus.post_ids()[row])
                                        && state.filter.matches(corpus, row)
                                        && review_eligible(
                                            corpus,
                                            &self.generation,
                                            &self.eligibility,
                                            row,
                                        )
                                })
                                .collect()
                        } else {
                            filtered_rows(
                                corpus,
                                &state.filter,
                                |row| {
                                    let id = corpus.post_ids()[row];
                                    !issued.contains(&row)
                                        && !self.eligibility.blacklist.blocks(row)
                                        && !self.eligibility.unavailable.contains(&id)
                                        && !retained.contains(&id)
                                },
                                &cancelled,
                            )?
                        };
                        let rows = selection::gallery(
                            corpus,
                            self.generation.scores.as_ref(),
                            &self.state.familiarity,
                            &self.state.grades,
                            &state.facet,
                            state.capacity,
                            &state.issued,
                            &candidates,
                            if self.generation.scores.is_some() {
                                &self.generation.model_id
                            } else {
                                &state.session
                            },
                            &cancelled,
                        )?;
                        ensure!(
                            rows.len() <= state.capacity.saturating_sub(state.issued.len()),
                            "Gallery plan exceeds remaining cohort capacity"
                        );
                        state.plan = Some(GalleryPlan {
                            generation: Arc::downgrade(&self.generation),
                            revision: self.state.admission_revision,
                            eligibility,
                            blacklist: Arc::downgrade(&self.eligibility.blacklist),
                            rows: rows.into(),
                        });
                    }
                    Ok(state
                        .plan
                        .as_ref()
                        .context("Missing Gallery plan")?
                        .rows
                        .iter()
                        .take(*size)
                        .copied()
                        .collect())
                };
                let mut rows = if state.complete { vec![] } else { rank(state)? };
                if next && rows.is_empty() {
                    let mut fresh = GalleryState::new(
                        state.facet.clone(),
                        state.filter.clone(),
                        state.capacity,
                    );
                    rows = rank(&mut fresh)?;
                    // Empty replacements are revalidated but never persisted.
                    lifecycle.propose(fresh);
                }
                ensure!(!cancelled(), "Selection cancelled");
                tracing::info!(
                    phase = "gallery_ranking",
                    elapsed_seconds = phase.elapsed().as_secs_f64(),
                    "Selection phase"
                );
                *output = Some(rows);
                Ok(())
            }
            PlanTask::Auto(selection) => {
                let Selection {
                    lifecycle,
                    size,
                    retained,
                    rows: output,
                    reply,
                } = selection;
                let cancelled = || self.cancelled.load(Ordering::Relaxed) || reply.is_closed();
                *output = None;
                let next = lifecycle.is_continuation();
                // Reclaim admission, never the speculative epoch's emissions.
                let previous = lifecycle.take_proposal();
                let state = lifecycle.original_mut();
                if let Some(mut previous) = previous {
                    state.candidates = std::mem::take(&mut previous.candidates);
                    state.candidate_generation = previous.candidate_generation.take();
                    state.candidate_revision = previous.candidate_revision;
                    state.candidate_blacklist = std::mem::take(&mut previous.candidate_blacklist);
                }
                let recent: HashSet<_> = self.eligibility.recent.iter().copied().collect();
                let current_pool = state
                    .candidate_generation
                    .as_ref()
                    .is_some_and(|generation| generation.as_ptr() == Arc::as_ptr(&self.generation));
                if !current_pool
                    || state.candidate_revision != self.state.admission_revision
                    || state.candidate_blacklist.as_ptr()
                        != Arc::as_ptr(&self.eligibility.blacklist)
                {
                    let phase = Instant::now();
                    // Recent views and retained cards are volatile predicates,
                    // not permanent exclusions from this generation's admission.
                    state.candidate_generation = None;
                    state.candidates = filtered_rows(
                        corpus,
                        &state.filter,
                        |row| {
                            !self.eligibility.blacklist.blocks(row)
                                && selection::auto_eligible(
                                    &state.preset,
                                    self.state.grades[row],
                                    self.generation
                                        .scores
                                        .as_ref()
                                        .map(|scores| scores.like_or_love[row]),
                                )
                                && (state.preset != "fresh"
                                    || self.state.grades[row].is_none()
                                        && !corpus.post_artists(row).is_empty()
                                        && self.state.familiarity.post_rated_count[row] == 0)
                        },
                        &cancelled,
                    )?;
                    state.candidate_generation = Some(Arc::downgrade(&self.generation));
                    state.candidate_revision = self.state.admission_revision;
                    state.candidate_blacklist = Arc::downgrade(&self.eligibility.blacklist);
                    tracing::info!(phase="auto_candidates", preset=%state.preset, candidates=state.candidates.len(), elapsed_seconds=phase.elapsed().as_secs_f64(), "Selection phase");
                }
                let rank = |state: &AutoState| -> Result<selection::AutoPlan> {
                    let phase = Instant::now();
                    let plan = selection::auto_page(
                        corpus,
                        self.generation.scores.as_ref(),
                        &self.state.familiarity,
                        &self.state.grades,
                        &state.candidates,
                        &state.emitted,
                        &self.eligibility.recent,
                        &state.preset,
                        *size,
                        state.seed,
                        state.epoch,
                        &|row| {
                            !recent.contains(&row)
                                && !retained.contains(&corpus.post_ids()[row])
                                && state.filter.matches(corpus, row)
                                && !self.eligibility.blacklist.blocks(row)
                        },
                        &cancelled,
                    )?;
                    tracing::info!(phase="auto_ranking", preset=%state.preset, candidates=state.candidates.len(), elapsed_seconds=phase.elapsed().as_secs_f64(), "Selection phase");
                    Ok(plan)
                };
                let mut plan = rank(state)?;
                if next && plan.rows.is_empty() && !plan.exhausted {
                    let epoch = state.epoch.checked_add(1).context("Auto epoch overflow")?;
                    let seed = (state.seed as u32)
                        .wrapping_mul(1664525)
                        .wrapping_add(1013904223)
                        .wrapping_add(epoch.wrapping_mul(2246822519))
                        as i32;
                    let mut new =
                        AutoState::new(state.preset.clone(), state.filter.clone(), seed, epoch);
                    // The latest admission pool is already available. A genuine
                    // rollover resets emission, not admission or volatile filters.
                    new.candidates = std::mem::take(&mut state.candidates);
                    new.candidate_generation = state.candidate_generation.take();
                    new.candidate_revision = state.candidate_revision;
                    new.candidate_blacklist = std::mem::take(&mut state.candidate_blacklist);
                    plan = rank(&new)?;
                    lifecycle.propose(new);
                }
                ensure!(!cancelled(), "Selection cancelled");
                *output = Some(plan);
                Ok(())
            }
        }
    }
}
impl Backend {
    fn save_settings(&mut self, event_id: String, value: UserSettings) -> Result<UserSettings> {
        let mut value = settings::validate(value)?;
        let policy = if value.blacklist == self.blacklist.names {
            self.blacklist.clone()
        } else {
            let compiled = Blacklist::compile(&self.catalog, &self.corpus, &value.blacklist)?;
            if compiled.names == self.blacklist.names {
                self.blacklist.clone()
            } else {
                Arc::new(compiled)
            }
        };
        if value.blacklist != policy.names {
            value.blacklist = policy.names.clone();
        }
        let retried = self.store.contains_event(&event_id)?;
        self.store.save_preferences(&event_id, &value)?;
        if retried {
            return Ok(self.settings.clone());
        }
        self.blacklist = policy;
        self.settings = value;
        Ok(self.settings.clone())
    }
    fn accessible_context(&self, mut context: BrowseContext) -> BrowseContext {
        let accessible = |entry: &BrowseEntry| {
            self.corpus
                .find_post(entry.card.post_id)
                .is_some_and(|row| !self.blacklist.blocks(row))
        };
        let cursor = context
            .cursor
            .as_ref()
            .and_then(|id| {
                context
                    .entries
                    .iter()
                    .position(|entry| &entry.entry_id == id)
            })
            .unwrap_or(0);
        context.cursor = context
            .entries
            .iter()
            .skip(cursor)
            .find(|entry| accessible(entry))
            .or_else(|| {
                context
                    .entries
                    .iter()
                    .take(cursor)
                    .rev()
                    .find(|entry| accessible(entry))
            })
            .map(|entry| entry.entry_id.clone());
        context.entries.retain(accessible);
        for entry in &mut context.entries {
            entry.card.current_grade = self
                .corpus
                .find_post(entry.card.post_id)
                .and_then(|row| self.grades[row]);
        }
        context
    }
    fn activate_context(
        &mut self,
        event_id: &str,
        mode: &str,
        filter: &str,
        fresh: bool,
    ) -> Result<BrowseContext> {
        ensure!(
            matches!(mode, "favorites" | "balanced" | "fresh")
                || selection::facets().iter().any(|facet| facet.slug == mode),
            "Unknown browsing mode '{mode}'"
        );
        let filter = TagFilter::parse(&self.catalog, filter)?;
        let context = self
            .store
            .activate_context(event_id, mode, &filter.canonical, fresh)?;
        Ok(self.accessible_context(context))
    }
    fn presented(
        &mut self,
        event_id: &str,
        context_id: &str,
        entry_id: Option<&str>,
        card: &Card,
    ) -> Result<()> {
        if !self.store.contains_event(event_id)? {
            let row = self
                .corpus
                .find_post(card.post_id)
                .context("Post is absent from corpus")?;
            ensure!(
                !self.blacklist.blocks(row),
                "Post is blocked by the blacklist"
            );
            if entry_id.is_none() {
                let context = self
                    .store
                    .active_context()?
                    .context("No active browsing context")?;
                ensure!(
                    context.context_id == context_id,
                    "Browsing context is no longer active"
                );
                let visible = self.accessible_context(context);
                ensure!(
                    visible.entries.last().map(|entry| entry.entry_id.as_str())
                        == visible.cursor.as_deref(),
                    "New recommendations require the history frontier"
                );
            }
        }
        self.store.presented(event_id, context_id, entry_id, card)
    }
    fn open(config: AppConfig) -> Result<Self> {
        let start = Instant::now();
        tracing::info!(phase = "corpus", "Opening native application");
        let corpus = Arc::new(Corpus::open(&config.corpus_path)?);
        let catalog = TagCatalog::load(&config.corpus_path)?;
        let mut store = Store::open(&config.interaction_path)?;
        let preferences = store.preferences()?;
        let imported = preferences.is_none();
        let mut settings = match preferences {
            Some(settings) => settings,
            None => settings::load(&config.settings_path)?,
        };
        settings = settings::validate(settings)?;
        let blacklist = Arc::new(Blacklist::compile(&catalog, &corpus, &settings.blacklist)?);
        settings.blacklist = blacklist.names.clone();
        if imported {
            store.save_preferences(
                &format!("preferences-import-{}", uuid::Uuid::new_v4()),
                &settings,
            )?;
        }
        let ratings = store.ratings()?;
        let rating_count = ratings.len();
        let mut grades = vec![None; corpus.post_count];
        let mut rated_rows = Vec::with_capacity(rating_count);
        for r in &ratings {
            if let Some(row) = corpus.find_post(r.post_id) {
                grades[row] = Some(r.grade);
                rated_rows.push(row);
            }
        }
        let familiarity = Familiarity::build(&corpus, &ratings);
        let live_artist_counts = familiarity.rated_count.clone();
        // One startup copy, never a per-request corpus-sized clone.
        let planner = Some(Box::new(PlannerState {
            familiarity,
            grades: grades.clone(),
            admission_revision: 0,
            rebuild: false,
        }));
        let threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .saturating_sub(1)
            .max(1);
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|n| format!("ff-compute-{n}"))
                .build()?,
        );
        tracing::info!(
            phase = "fit",
            rating_count,
            threads,
            elapsed_seconds = start.elapsed().as_secs_f64(),
            "Corpus and private state opened"
        );
        let training = labels(ratings);
        let generation = Arc::new(pool.install(|| model::fit(&corpus, &training))?);
        tracing::info!(phase="ready", model_id=%generation.model_id, elapsed_seconds=start.elapsed().as_secs_f64(), "Native application ready");
        Ok(Self {
            corpus,
            catalog,
            store,
            settings,
            blacklist,
            generation,
            planner,
            live_artist_counts,
            grades,
            rated_rows,
            pending_grades: HashMap::new(),
            rating_count,
            auto: None,
            gallery: None,
            planning: None,
            pool,
            refresh: Refresh::default(),
        })
    }
    fn run(
        mut self,
        mut receiver: mpsc::Receiver<Command>,
        sender: mpsc::WeakSender<Command>,
        runtime: tokio::runtime::Handle,
        gate: Arc<Semaphore>,
    ) {
        let mut shutdown = false;
        let mut shutdown_waiters = Vec::new();
        while let Some(command) = receiver.blocking_recv() {
            match command {
                Command::Shutdown(reply) => {
                    shutdown = true;
                    gate.close();
                    shutdown_waiters.push(reply);
                    if let Some(cancel) = &self.planning {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    self.refresh.cancel_debounce();
                }
                Command::Planned(completion) => {
                    self.planning = None;
                    let PlanCompletion { mut job, result } = *completion;
                    self.reclaim_changes(&mut job);
                    let result = result.and_then(|_| {
                        ensure!(!shutdown && !job.task.is_closed(), "Selection cancelled");
                        self.revalidate(&job)
                    });
                    match result {
                        Ok(false) => match self.eligibility(&job.task) {
                            Ok(mut eligibility) => {
                                eligibility.changes = self.pending_grades.drain().collect();
                                job.eligibility = eligibility;
                                job.generation = self.generation.clone();
                                self.dispatch(job, &sender);
                            }
                            Err(error) => {
                                self.planner = Some(job.state);
                                job.task.fail(&mut self, error);
                            }
                        },
                        Ok(true) => {
                            self.planner = Some(job.state);
                            self.commit(job.task, &job.generation);
                        }
                        Err(error) => {
                            self.planner = Some(job.state);
                            job.task.fail(&mut self, error);
                        }
                    }
                }
                Command::Fitted(request, result) => {
                    if !self.refresh.complete(request) {
                        continue;
                    }
                    if !shutdown {
                        match result {
                            Ok(Some(generation)) => {
                                tracing::info!(model_id=%generation.model_id, "Publishing complete serving generation");
                                self.generation = generation;
                                self.refresh.error = None;
                            }
                            Ok(None) => {
                                tracing::info!(model_id=%self.generation.model_id, "Reusing unchanged serving generation; skipped fitting and scoring");
                                self.refresh.error = None;
                            }
                            Err(error) => {
                                tracing::error!(error=%error, "Refresh failed; retaining previous complete generation");
                                self.refresh.error = Some(format!("{error:#}"));
                            }
                        }
                        // Demand arriving during a fit still requires a trailing
                        // catch-up, even if its writes preceded that fit's read.
                        self.refresh.schedule(&runtime, &sender);
                    }
                }
                Command::StartFit(request) => {
                    if !shutdown
                        && let Some(sender) = sender.upgrade()
                        && self.refresh.admit(request)
                    {
                        let path = self.store.path.clone();
                        let corpus = self.corpus.clone();
                        let serving = self.generation.clone();
                        self.pool.spawn(move || {
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    let ratings = Store::snapshot_ratings_at(&path)?;
                                    let training = labels(ratings);
                                    model::fit_if_changed(
                                        &corpus,
                                        &training,
                                        &serving.model_id,
                                        serving.status.route,
                                    )
                                    .map(|result| result.map(Arc::new))
                                }))
                                .unwrap_or_else(|_| Err(anyhow!("Model worker panicked")));
                            drop(corpus);
                            drop(serving);
                            let _ = sender.blocking_send(Command::Fitted(request, result));
                        });
                    }
                }
                command if shutdown => Self::reject(command),
                Command::Health(reply) => {
                    let _ = reply.send(Ok(Health {
                        corpus_id: self.corpus.corpus_id.clone(),
                        model_id: self.generation.model_id.clone(),
                        model_status: self.generation.status.clone(),
                        rating_count: self.rating_count,
                        refreshing: self.refresh.refreshing(),
                        refresh_error: self.refresh.error.clone(),
                    }));
                }
                Command::Settings(reply) => {
                    let _ = reply.send(Ok(self.settings.clone()));
                }
                Command::SaveSettings(event, value, reply) => {
                    let _ = reply.send(self.save_settings(event, value));
                }
                Command::RestoreContext(reply) => {
                    let result = self
                        .store
                        .active_context()
                        .map(|context| context.map(|context| self.accessible_context(context)));
                    let _ = reply.send(result);
                }
                Command::ActivateContext(event, mode, filter, fresh, reply) => {
                    let result = self.activate_context(&event, &mode, &filter, fresh);
                    let _ = reply.send(result);
                }
                Command::Presented(event, context, entry, card, reply) => {
                    let result = self.presented(&event, &context, entry.as_deref(), &card);
                    let _ = reply.send(result);
                }
                Command::Facets(reply) => {
                    let _ = reply.send(Ok(selection::facets()));
                }
                Command::Search(query, limit, reply) => {
                    let _ = reply.send(if (1..=50).contains(&limit) {
                        Ok(self.catalog.search(&query, limit))
                    } else {
                        Err(anyhow!("Tag limit must be in [1,50]"))
                    });
                }
                Command::Plan(request, permit) => {
                    if request.is_closed() {
                        continue;
                    }
                    if let Some(task) = self.prepare(request) {
                        match self.eligibility(&task) {
                            Ok(mut eligibility) => {
                                eligibility.changes = self.pending_grades.drain().collect();
                                let state = self
                                    .planner
                                    .take()
                                    .expect("selection permit owns planner state");
                                let job = PlanJob {
                                    task,
                                    state,
                                    generation: self.generation.clone(),
                                    eligibility,
                                    cancelled: Arc::new(AtomicBool::new(false)),
                                    _permit: permit,
                                };
                                self.dispatch(job, &sender);
                            }
                            Err(error) => task.fail(&mut self, error),
                        }
                    }
                }
                Command::Viewed(event, card, reply) => {
                    let _ = reply.send(self.store.viewed(&event, card.post_id, &card.session_id));
                }
                Command::Rate(event, card, grade, reply) => {
                    let result = self.rate(event, &card, grade);
                    if result.as_ref().is_ok_and(|r| r.changed) {
                        self.refresh.request(&runtime, &sender);
                    }
                    let _ = reply.send(result);
                }
            }
            if shutdown && self.refresh.in_flight.is_none() && self.planning.is_none() {
                break;
            }
        }
        gate.close();
        self.refresh.cancel_debounce();
        // In-flight workers retain a sender until their completion is consumed.
        // Releasing resources precedes acknowledging explicit shutdown.
        drop(receiver);
        drop(self);
        for reply in shutdown_waiters {
            let _ = reply.send(Ok(()));
        }
        tracing::info!("Native application worker stopped");
    }
    fn dispatch(&mut self, mut job: PlanJob, sender: &mpsc::WeakSender<Command>) {
        if let Some(sender) = sender.upgrade() {
            self.planning = Some(job.cancelled.clone());
            let corpus = self.corpus.clone();
            let path = self.store.path.clone();
            self.pool.spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    job.plan(&corpus, &path)
                }))
                .unwrap_or_else(|_| {
                    job.state.rebuild = true;
                    Err(anyhow!("Selection worker panicked"))
                });
                drop(corpus);
                let _ = sender
                    .blocking_send(Command::Planned(Box::new(PlanCompletion { job, result })));
            });
        } else {
            self.reclaim_changes(&mut job);
            self.planner = Some(job.state);
            job.task.fail(self, anyhow!("Application has shut down"));
        }
    }
    fn reclaim_changes(&mut self, job: &mut PlanJob) {
        // A cancelled synchronization returns its unapplied deltas. Newer live
        // grades already queued by the actor take precedence over older work.
        for (row, grade) in job.eligibility.changes.drain(..) {
            self.pending_grades.entry(row).or_insert(grade);
        }
    }
    fn reject(command: Command) {
        macro_rules! reject {
            ($r:expr) => {{
                let _ = $r.send(Err(anyhow!("Application is shutting down")));
            }};
        }
        match command {
            Command::Health(r) => reject!(r),
            Command::Settings(r) => reject!(r),
            Command::SaveSettings(_, _, r) => reject!(r),
            Command::RestoreContext(r) => reject!(r),
            Command::ActivateContext(_, _, _, _, r) => reject!(r),
            Command::Presented(_, _, _, _, r) => reject!(r),
            Command::Facets(r) => reject!(r),
            Command::Search(_, _, r) => reject!(r),
            Command::Plan(request, _) => request.fail(anyhow!("Application is shutting down")),
            Command::Rate(_, _, _, r) => reject!(r),
            Command::Viewed(_, _, r) => reject!(r),
            Command::Shutdown(r) => reject!(r),
            Command::StartFit(_) | Command::Fitted(_, _) | Command::Planned(_) => {}
        }
    }
    fn prepare(&mut self, request: PlanRequest) -> Option<PlanTask> {
        match request {
            PlanRequest::Gallery(request, reply) => {
                let input = (|| {
                    ensure!(
                        (1..=100).contains(&request.page_size),
                        "Gallery size must be in [1,100]"
                    );
                    ensure!(
                        request.exclude_post_ids.len() <= 1000,
                        "At most 1000 buffered posts are allowed"
                    );
                    let capacity = selection::facets()
                        .into_iter()
                        .find(|f| f.slug == request.facet)
                        .with_context(|| format!("Unknown facet '{}'", request.facet))?
                        .cohort_size;
                    let filter = TagFilter::parse(&self.catalog, &request.filter)?;
                    Ok((
                        SelectionState::Start(GalleryState::new(request.facet, filter, capacity)),
                        request.page_size,
                        request.exclude_post_ids.into_iter().collect(),
                    ))
                })();
                Selection::admit(input, reply).map(PlanTask::Gallery)
            }
            PlanRequest::NextGallery(request, reply) => {
                let input = (|| {
                    ensure!(
                        (1..=100).contains(&request.page_size),
                        "Gallery size must be in [1,100]"
                    );
                    ensure!(
                        request.exclude_post_ids.len() <= 1000,
                        "At most 1000 buffered posts are allowed"
                    );
                    ensure!(
                        self.gallery
                            .as_ref()
                            .is_some_and(|state| state.session == request.session_id),
                        "Gallery session is no longer active"
                    );
                    let original = self.gallery.take().expect("validated Gallery continuation");
                    Ok((
                        SelectionState::Continue {
                            original,
                            proposed: None,
                        },
                        request.page_size,
                        request.exclude_post_ids.into_iter().collect(),
                    ))
                })();
                Selection::admit(input, reply).map(PlanTask::Gallery)
            }
            PlanRequest::StartAuto(request, reply) => {
                let input = (|| {
                    ensure!(
                        (1..=200).contains(&request.page_size),
                        "Auto page size must be in [1,200]"
                    );
                    ensure!(
                        matches!(request.preset.as_str(), "favorites" | "balanced" | "fresh"),
                        "Unknown Auto preset '{}'",
                        request.preset
                    );
                    let filter = TagFilter::parse(&self.catalog, &request.filter)?;
                    Ok((
                        SelectionState::Start(AutoState::new(
                            request.preset,
                            filter,
                            request.seed,
                            0,
                        )),
                        request.page_size,
                        request.exclude_post_ids.into_iter().collect(),
                    ))
                })();
                Selection::admit(input, reply).map(PlanTask::Auto)
            }
            PlanRequest::NextAuto(request, reply) => {
                let input = (|| {
                    ensure!(
                        (1..=200).contains(&request.page_size),
                        "Auto page size must be in [1,200]"
                    );
                    ensure!(
                        self.auto
                            .as_ref()
                            .is_some_and(|state| state.session == request.session_id),
                        "Auto session is no longer active"
                    );
                    let original = self.auto.take().expect("validated Auto continuation");
                    Ok((
                        SelectionState::Continue {
                            original,
                            proposed: None,
                        },
                        request.page_size,
                        request.exclude_post_ids.into_iter().collect(),
                    ))
                })();
                Selection::admit(input, reply).map(PlanTask::Auto)
            }
        }
    }
    fn eligibility(&self, task: &PlanTask) -> Result<Eligibility> {
        let review = matches!(task, PlanTask::Gallery(selection) if selection.lifecycle.original().facet == "rating_review");
        Ok(Eligibility {
            blacklist: self.blacklist.clone(),
            changes: vec![],
            review_rows: if review {
                self.rated_rows.clone()
            } else {
                vec![]
            },
            unavailable: if matches!(task, PlanTask::Gallery(_)) && !review {
                self.store.unavailable()?
            } else {
                HashSet::new()
            },
            recent: if matches!(task, PlanTask::Auto(_)) {
                self.store
                    .recent_views(500)?
                    .into_iter()
                    .filter_map(|id| self.corpus.find_post(id))
                    .collect()
            } else {
                vec![]
            },
            checkpoints: if review {
                self.store.checkpoints()?
            } else {
                HashMap::new()
            },
            review_exclusions: if review {
                self.store.review_exclusions()?
            } else {
                HashSet::new()
            },
            review_sequence: if review {
                self.store.review_sequence()?
            } else {
                0
            },
        })
    }
    fn revalidate(&self, job: &PlanJob) -> Result<bool> {
        // Publication invalidates the complete ranking, not just its label.
        // Rerank with one current generation before any offer is persisted.
        if !Arc::ptr_eq(&job.generation, &self.generation)
            || !Arc::ptr_eq(&job.eligibility.blacklist, &self.blacklist)
        {
            return Ok(false);
        }
        let live = self.eligibility(&job.task)?;
        let check = |row: usize, filter: &TagFilter| {
            !self.blacklist.blocks(row) && filter.matches(&self.corpus, row)
        };
        Ok(match &job.task {
            PlanTask::Gallery(selection) => {
                let rows = selection.rows.as_ref().context("Missing Gallery plan")?;
                let retained = &selection.retained;
                let state = selection.lifecycle.selected();
                // Empty pages and exhaustion claims require complete input validation.
                if !self.pending_grades.is_empty()
                    || gallery_eligibility(&self.corpus, &job.generation, &live, state, retained)
                        != gallery_eligibility(
                            &self.corpus,
                            &job.generation,
                            &job.eligibility,
                            state,
                            retained,
                        )
                {
                    return Ok(false);
                }
                rows.iter().all(|&(row, _)| {
                    let id = self.corpus.post_ids()[row];
                    check(row, &state.filter)
                        && !state.issued.contains(&row)
                        && !retained.contains(&id)
                        && if state.facet == "rating_review" {
                            self.grades[row].is_some()
                                && review_eligible(&self.corpus, &job.generation, &live, row)
                        } else {
                            !live.unavailable.contains(&id)
                        }
                })
            }
            PlanTask::Auto(selection) => {
                let plan = selection.rows.as_ref().context("Missing Auto plan")?;
                let retained = &selection.retained;
                // Empty pages and exhaustion are eligibility claims too.
                if !self.pending_grades.is_empty() || live.recent != job.eligibility.recent {
                    return Ok(false);
                }
                let state = selection.lifecycle.selected();
                plan.rows.iter().all(|&row| {
                    check(row, &state.filter)
                        && !live.recent.contains(&row)
                        && !retained.contains(&self.corpus.post_ids()[row])
                        && selection::auto_eligible(
                            &state.preset,
                            self.grades[row],
                            self.generation
                                .scores
                                .as_ref()
                                .map(|scores| scores.like_or_love[row]),
                        )
                        && (state.preset != "fresh"
                            || self.grades[row].is_none()
                                && !self.corpus.post_artists(row).is_empty()
                                && artist_count(&self.corpus, &self.live_artist_counts, row) == 0)
                })
            }
        })
    }
    fn card(
        &self,
        generation: &FitResult,
        facet: &str,
        session: &str,
        epoch: u32,
        position: usize,
        universe_size: usize,
        row: usize,
        score: f32,
    ) -> Card {
        Card {
            post_id: self.corpus.post_ids()[row],
            md5: self.corpus.md5(row),
            file_extension: self.corpus.extension(row).into(),
            content_rating: match self.corpus.rating_code(row) {
                1 => "s",
                2 => "q",
                3 => "e",
                _ => unreachable!("validated corpus rating"),
            }
            .into(),
            selection_score: score,
            current_grade: self.grades[row],
            facet: facet.into(),
            session_id: session.into(),
            epoch,
            position,
            universe_size,
            model_id: generation.model_id.clone(),
        }
    }
    fn commit(&mut self, task: PlanTask, generation: &FitResult) {
        match task {
            PlanTask::Gallery(mut selection) => {
                let result = (|| {
                    ensure!(!selection.reply.is_closed(), "Selection cancelled");
                    let rows = selection.rows.take().context("Missing Gallery plan")?;
                    self.commit_gallery(&mut selection.lifecycle, generation, &rows)
                })();
                match result {
                    Ok(page) => {
                        self.gallery =
                            if !page.items.is_empty() || selection.lifecycle.is_continuation() {
                                Some(selection.lifecycle.into_selected(!page.items.is_empty()))
                            } else {
                                None
                            };
                        let _ = selection.reply.send(Ok(page));
                    }
                    Err(error) => PlanTask::Gallery(selection).fail(self, error),
                }
            }
            PlanTask::Auto(mut selection) => {
                let result = (|| {
                    ensure!(!selection.reply.is_closed(), "Selection cancelled");
                    let plan = selection.rows.take().context("Missing Auto plan")?;
                    self.commit_auto(&mut selection.lifecycle, generation, plan)
                })();
                match result {
                    Ok(page) => {
                        self.auto = Some(selection.lifecycle.into_selected(!page.items.is_empty()));
                        let _ = selection.reply.send(Ok(page));
                    }
                    Err(error) => PlanTask::Auto(selection).fail(self, error),
                }
            }
        }
    }
    fn commit_gallery(
        &mut self,
        lifecycle: &mut SelectionState<GalleryState>,
        generation: &FitResult,
        rows: &[(usize, f32)],
    ) -> Result<Gallery> {
        let phase = Instant::now();
        let next = lifecycle.is_continuation();
        let original = lifecycle.original();
        let serving = if rows.is_empty() {
            original
        } else {
            lifecycle.selected()
        };
        let end = serving.issued.len() + rows.len();
        let exhausted = rows.is_empty();
        let completed = exhausted
            || end == serving.capacity
            || serving
                .plan
                .as_ref()
                .is_some_and(|plan| plan.rows.len() == rows.len());
        let items: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(index, &(row, score))| {
                self.card(
                    generation,
                    &serving.facet,
                    &serving.session,
                    0,
                    serving.issued.len() + index,
                    end,
                    row,
                    score,
                )
            })
            .collect();
        if !serving.issued.is_empty() || !items.is_empty() {
            let previous = if !items.is_empty() && lifecycle.proposed().is_some() {
                Some((original.owner(), true))
            } else if !next && !items.is_empty() {
                self.gallery
                    .as_ref()
                    .map(|previous| (previous.owner(), false))
            } else {
                None
            };
            self.store.commit_gallery(
                serving.owner(),
                &generation.model_id,
                &items,
                completed,
                exhausted,
                previous,
            )?;
        } else if !next && let Some(previous) = &self.gallery {
            self.store.retire_gallery(previous.owner())?;
        }
        let facet = serving.facet.clone();
        let session_id = if serving.issued.is_empty() && items.is_empty() {
            String::new()
        } else {
            serving.session.clone()
        };
        // No cursor or issued-history mutation precedes the transaction.
        if !items.is_empty() || next {
            if rows.is_empty() {
                lifecycle.take_proposal();
            }
            let committed = lifecycle.selected_mut();
            if let Some(plan) = &mut committed.plan {
                plan.rows.drain(..rows.len());
                if committed.facet == "rating_review" {
                    for &(row, _) in rows {
                        gallery_eligibility_delta(
                            &mut plan.eligibility,
                            self.corpus.post_ids()[row],
                            true,
                        );
                    }
                }
            }
            committed.issued.extend(rows.iter().map(|&(row, _)| row));
            committed.complete = completed;
        }
        tracing::info!(phase="gallery_commit", facet=%facet, elapsed_seconds=phase.elapsed().as_secs_f64(), "Selection phase");
        Ok(Gallery {
            session_id,
            facet,
            items,
            exhausted,
        })
    }
    fn commit_auto(
        &mut self,
        lifecycle: &mut SelectionState<AutoState>,
        generation: &FitResult,
        plan: selection::AutoPlan,
    ) -> Result<AutoPage> {
        let phase = Instant::now();
        if plan.rows.is_empty()
            && let Some(mut proposed) = lifecycle.take_proposal()
        {
            let original = lifecycle.original_mut();
            original.candidates = std::mem::take(&mut proposed.candidates);
            original.candidate_generation = proposed.candidate_generation.take();
            original.candidate_revision = proposed.candidate_revision;
            original.candidate_blacklist = std::mem::take(&mut proposed.candidate_blacklist);
        }
        let serving = lifecycle.selected_mut();
        let items: Vec<_> = plan
            .rows
            .iter()
            .zip(&plan.scores)
            .enumerate()
            .map(|(index, (&row, &score))| {
                self.card(
                    generation,
                    "play",
                    &serving.session,
                    serving.epoch,
                    serving.emitted.len() + index,
                    serving.candidates.len(),
                    row,
                    score,
                )
            })
            .collect();
        if !items.is_empty() {
            if !serving.emitted.is_empty() {
                self.store
                    .append_auto(&serving.session, serving.emitted.len(), &items)?;
            } else {
                self.store.create_auto(
                    &serving.session,
                    &serving.preset,
                    &serving.filter.canonical,
                    &generation.model_id,
                    serving.seed,
                    &items,
                )?;
            }
            serving.emitted.extend(plan.rows);
        }
        let page = AutoPage {
            session_id: serving.session.clone(),
            items,
            seed: serving.seed,
            epoch: serving.epoch,
            universe_size: serving.candidates.len(),
            emitted_count: serving.emitted.len(),
            exhausted: plan.exhausted,
        };
        tracing::info!(
            phase = "auto_commit",
            elapsed_seconds = phase.elapsed().as_secs_f64(),
            "Selection phase"
        );
        Ok(page)
    }
    fn rate(&mut self, event_id: String, card: &Card, grade: Grade) -> Result<RatingResult> {
        let row = self
            .corpus
            .find_post(card.post_id)
            .context("Post is absent from corpus")?;
        let prediction = self
            .generation
            .scores
            .as_ref()
            .map(|scores| RatingPrediction {
                predicted_grade: selection::median_grade(scores, row),
                model_id: self.generation.model_id.clone(),
            });
        let result =
            self.store
                .rate(&event_id, card.post_id, grade, &card.session_id, prediction)?;
        if result.changed {
            if self.grades[row].is_none() {
                for &artist in self.corpus.post_artists(row) {
                    self.live_artist_counts[artist as usize] += 1;
                }
                self.rating_count += 1;
                self.rated_rows.push(row);
            }
            if self.grades[row] != Some(grade) {
                self.pending_grades.insert(row, grade);
            }
            self.grades[row] = Some(grade);
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "../tests/unit/app/mod.rs"]
mod tests;

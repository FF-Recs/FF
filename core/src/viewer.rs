//! Pure browsing transitions. Offered cards retain their original durable provenance;
//! selection generations, navigation visits, and media attempts are separate identities.
use crate::types::{
    AutoNextRequest, AutoPage, BrowseContext, Card, Gallery, GalleryNextRequest, Grade,
    RatingResult,
};
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

pub struct Page {
    pub session_id: String,
    pub items: Vec<Card>,
    pub exhausted: bool,
}

impl From<AutoPage> for Page {
    fn from(page: AutoPage) -> Self {
        Self {
            session_id: page.session_id,
            items: page.items,
            exhausted: page.exhausted,
        }
    }
}
impl From<Gallery> for Page {
    fn from(page: Gallery) -> Self {
        Self {
            session_id: page.session_id,
            items: page.items,
            exhausted: page.exhausted,
        }
    }
}

const LOOKAHEAD: usize = 4;
const LOW_WATER: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    Unavailable,
    Ready,
    Deferred,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NavigationIntent {
    Back,
    Next,
    Context {
        mode: String,
        filter: String,
        fresh: bool,
    },
}

struct PendingNavigation {
    generation: u64,
    visit: u64,
    intent: NavigationIntent,
    automatic: bool,
}

pub struct ViewerItem {
    pub card: Card,
    /// None is replaceable, never-presented lookahead.
    pub entry_id: Option<String>,
}

#[derive(Default)]
enum Presentation {
    #[default]
    Unrecorded,
    Pending {
        event: String,
    },
    Recorded {
        event: String,
    },
    Failed {
        event: String,
        error: String,
    },
}

struct VoteCommand {
    event: String,
    generation: u64,
    visit: u64,
    card: Card,
    grade: Grade,
}

#[derive(Default)]
enum Rating {
    #[default]
    Idle,
    Pending {
        command: VoteCommand,
    },
    Failed {
        command: VoteCommand,
        error: String,
    },
    Rejected {
        error: String,
    },
}

pub struct ViewerState {
    pub mode: String,
    pub filter: String,
    pub generation: u64,
    /// Read-only lookahead ownership is independent of durable visit acknowledgements.
    pub lookahead_generation: u64,
    pub visit: u64,
    pub items: Vec<ViewerItem>,
    pub context_id: String,
    pub context_pending: bool,
    pub policy_pending: bool,
    /// Acknowledged live grades overlay delayed offer snapshots; offer identity stays fixed.
    pub acknowledged_grades: HashMap<i64, (i64, Grade)>,
    pub index: usize,
    pub furthest: usize,
    pub session: String,
    pub loading: bool,
    pub refilling: bool,
    pub more: bool,
    /// Delivery authority, independent of the retry-stable backend event UUID.
    pub vote_attempt: u64,
    /// A no-progress response must not trigger an empty polling loop.
    pub refill_stalled: bool,
    /// Every accepted presentation retry has distinct acknowledgement authority.
    pub view_attempt: u64,
    pub media_generation: u64,
    pub media_ready: bool,
    pub ready_since: Option<Instant>,
    pub advance: bool,
    pub selection_error: String,
    pub refill_error: String,
    pending_navigation: Option<PendingNavigation>,
    presentation: Presentation,
    rating: Rating,
}
impl Default for ViewerState {
    fn default() -> Self {
        Self {
            mode: "discover".into(),
            filter: String::new(),
            generation: 0,
            lookahead_generation: 0,
            visit: 0,
            items: Vec::new(),
            context_id: String::new(),
            context_pending: false,
            policy_pending: false,
            acknowledged_grades: HashMap::new(),
            index: 0,
            furthest: 0,
            session: String::new(),
            loading: true,
            refilling: false,
            more: false,
            vote_attempt: 0,
            refill_stalled: false,
            view_attempt: 0,
            media_generation: 0,
            media_ready: false,
            ready_since: None,
            advance: false,
            selection_error: String::new(),
            refill_error: String::new(),
            pending_navigation: None,
            presentation: Presentation::Unrecorded,
            rating: Rating::Idle,
        }
    }
}
impl ViewerState {
    pub fn current(&self) -> Option<&Card> {
        self.items.get(self.index).map(|item| &item.card)
    }
    /// Existing navigation sequence only: current, five ahead, then five behind.
    /// Distances are absolute navigation steps, independent of fetch priority.
    /// Reading cache targets never selects offers or promotes them to history.
    pub fn media_window(&self) -> impl Iterator<Item = (&Card, usize)> {
        self.current()
            .filter(|_| !self.blocked() && !self.loading)
            .into_iter()
            .flat_map(move |_| {
                let (behind, forward) = self.items.split_at(self.index);
                forward
                    .iter()
                    .enumerate()
                    .take(6)
                    .map(|(distance, item)| (&item.card, distance))
                    .chain(
                        behind
                            .iter()
                            .rev()
                            .take(5)
                            .enumerate()
                            .map(|(index, item)| (&item.card, index + 1)),
                    )
            })
    }
    pub fn blocked(&self) -> bool {
        self.context_pending || self.policy_pending
    }
    pub fn current_entry_id(&self) -> Option<&str> {
        self.items
            .get(self.index)
            .and_then(|item| item.entry_id.as_deref())
    }
    pub fn viewing(&self) -> bool {
        matches!(self.presentation, Presentation::Pending { .. })
    }
    pub fn viewed(&self) -> bool {
        matches!(self.presentation, Presentation::Recorded { .. })
    }
    pub fn view_event(&self) -> Option<&str> {
        match &self.presentation {
            Presentation::Unrecorded => None,
            Presentation::Pending { event }
            | Presentation::Recorded { event }
            | Presentation::Failed { event, .. } => Some(event),
        }
    }
    pub fn view_error(&self) -> &str {
        match &self.presentation {
            Presentation::Failed { error, .. } => error,
            _ => "",
        }
    }
    pub fn voting(&self) -> bool {
        matches!(self.rating, Rating::Pending { .. })
    }
    pub fn vote_event(&self) -> Option<&str> {
        match &self.rating {
            Rating::Pending { command } | Rating::Failed { command, .. } => Some(&command.event),
            _ => None,
        }
    }
    pub fn vote_error(&self) -> &str {
        match &self.rating {
            Rating::Failed { error, .. } | Rating::Rejected { error } => error,
            _ => "",
        }
    }
    /// Original accepted payload for pending-write diagnostics and failed retries.
    pub fn vote_request(&self) -> Option<(&Card, Grade)> {
        match &self.rating {
            Rating::Pending { command } | Rating::Failed { command, .. } => {
                Some((&command.card, command.grade))
            }
            _ => None,
        }
    }
    pub fn failed_vote(&self) -> Option<(&Card, Grade)> {
        match &self.rating {
            Rating::Failed { command, .. } => Some((&command.card, command.grade)),
            _ => None,
        }
    }
    pub fn reject_vote(&mut self, error: String) {
        match &mut self.rating {
            Rating::Pending { .. } => {}
            Rating::Failed {
                error: previous, ..
            }
            | Rating::Rejected { error: previous } => {
                *previous = error;
            }
            Rating::Idle => self.rating = Rating::Rejected { error },
        }
    }
    pub fn grade_admission(&self) -> Admission {
        if self.current().is_none()
            || !self.media_ready
            || self.loading
            || self.blocked()
            || self.voting()
            || !self.view_error().is_empty()
        {
            Admission::Unavailable
        } else if self.viewed() {
            Admission::Ready
        } else {
            Admission::Deferred
        }
    }
    /// Context replacement may cancel selection reads, never durable writes.
    pub fn context_admission(&self) -> Admission {
        if self.blocked() {
            Admission::Unavailable
        } else if self.viewing()
            || self.voting()
            || !self.view_error().is_empty()
            || !self.vote_error().is_empty()
        {
            Admission::Deferred
        } else {
            Admission::Ready
        }
    }
    pub fn navigation_admission(&self, intent: &NavigationIntent) -> Admission {
        let admission = self.context_admission();
        if admission == Admission::Unavailable {
            return admission;
        }
        let available = match intent {
            NavigationIntent::Context { .. } => return admission,
            NavigationIntent::Back => self.current().is_some() && self.index > 0,
            NavigationIntent::Next => {
                self.current().is_some()
                    && (self.index + 1 < self.items.len()
                        || self.more
                        || self.refilling
                        || self.loading
                        || (self.session.is_empty()
                            && !self.refill_stalled
                            && self.selection_error.is_empty()))
            }
        };
        if !available {
            return Admission::Unavailable;
        }
        if admission == Admission::Deferred
            || self.loading
            || matches!(intent, NavigationIntent::Next)
                && (self.index + 1 >= self.items.len()
                    || !self.selection_error.is_empty()
                    || !self.refill_error.is_empty())
        {
            Admission::Deferred
        } else {
            Admission::Ready
        }
    }
    pub fn request_navigation(&mut self, intent: NavigationIntent) -> Admission {
        let admission = self.navigation_admission(&intent);
        if admission == Admission::Unavailable {
            return admission;
        }
        match &intent {
            NavigationIntent::Back => self.advance = false,
            NavigationIntent::Next => self.resume_refill(),
            NavigationIntent::Context { .. } => {}
        }
        self.pending_navigation = Some(PendingNavigation {
            generation: self.generation,
            visit: self.visit,
            intent,
            automatic: false,
        });
        admission
    }
    /// Latest queued context, for composing another mode/filter/fresh command.
    pub fn pending_context(&self) -> Option<(&str, &str, bool)> {
        let pending = self.pending_navigation.as_ref()?;
        if pending.generation != self.generation || pending.visit != self.visit {
            return None;
        }
        match &pending.intent {
            NavigationIntent::Context {
                mode,
                filter,
                fresh,
            } => Some((mode, filter, *fresh)),
            _ => None,
        }
    }
    pub fn take_navigation(&mut self) -> Option<NavigationIntent> {
        let pending = self.pending_navigation.as_ref()?;
        if pending.generation != self.generation || pending.visit != self.visit {
            self.pending_navigation = None;
            return None;
        }
        if self.blocked()
            || self.loading && !matches!(&pending.intent, NavigationIntent::Context { .. })
            || self.viewing()
            || self.voting()
            || !self.view_error().is_empty()
            || !self.vote_error().is_empty()
        {
            return None;
        }
        if matches!(&pending.intent, NavigationIntent::Next) {
            if !self.selection_error.is_empty() || !self.refill_error.is_empty() {
                return None;
            }
            if self.index + 1 >= self.items.len() {
                // An exhausted or non-progressing frontier cancels this command, not
                // a later publication. Only another explicit Next may resume it.
                if !self.refilling
                    && (self.refill_stalled || !self.more && !self.session.is_empty())
                {
                    self.pending_navigation = None;
                }
                return None;
            }
        }
        self.pending_navigation.take().map(|pending| pending.intent)
    }
    /// Invalidate reads and frames immediately, without deleting durable visits.
    pub fn suspend(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.lookahead_generation = self.lookahead_generation.wrapping_add(1);
        self.refilling = false;
        self.more = false;
        self.session.clear();
        self.advance = false;
        self.pending_navigation = None;
        self.retry_media();
    }
    pub fn begin_context(&mut self) -> u64 {
        self.suspend();
        self.context_pending = true;
        self.generation
    }
    /// Only the admitted activation/restore may replace the visible projection.
    pub fn restore(&mut self, generation: u64, context: BrowseContext) -> bool {
        if generation != self.generation || !self.context_pending {
            return false;
        }
        self.context_pending = false;
        self.policy_pending = false;
        self.context_id = context.context_id;
        self.mode = context.mode;
        self.filter = context.filter;
        self.index = context
            .cursor
            .as_ref()
            .and_then(|cursor| {
                context
                    .entries
                    .iter()
                    .position(|entry| &entry.entry_id == cursor)
            })
            .unwrap_or(0);
        self.items = context
            .entries
            .into_iter()
            .map(|entry| ViewerItem {
                card: entry.card,
                entry_id: Some(entry.entry_id),
            })
            .collect();
        self.furthest = self.items.len().saturating_sub(1);
        self.loading = false;
        self.refill_stalled = false;
        self.selection_error.clear();
        self.refill_error.clear();
        self.begin_visit();
        true
    }
    /// A failed accessibility projection must not reopen the pre-policy presentation.
    pub fn restore_failed(&mut self, generation: u64, error: anyhow::Error) -> bool {
        if generation != self.generation || !self.context_pending {
            return false;
        }
        self.context_pending = false;
        self.loading = false;
        self.selection_error = format!("Context not restored: {error:#}");
        true
    }
    pub fn is_video(&self) -> bool {
        self.current()
            .is_some_and(|c| matches!(c.file_extension.as_str(), "webm" | "mp4"))
    }
    pub fn is_auto(&self) -> bool {
        matches!(self.mode.as_str(), "balanced" | "favorites" | "fresh")
    }
    pub fn selection_page_size(&self) -> usize {
        LOOKAHEAD + 1
    }
    /// Exclude only the active offer and disposable lookahead, not lifetime history.
    pub fn selection_exclusions(&self) -> Vec<i64> {
        self.selection_exclusion_ids().collect()
    }
    fn selection_exclusion_ids(&self) -> impl Iterator<Item = i64> + '_ {
        self.current().map(|card| card.post_id).into_iter().chain(
            self.items
                .iter()
                .enumerate()
                .skip(self.furthest)
                .filter(|(index, item)| *index != self.index && item.entry_id.is_none())
                .map(|(_, item)| item.card.post_id),
        )
    }
    /// Invalidate unpublished work, returning whether live future work needs replacement.
    /// Publication alone never reopens an exhausted, failed, or no-progress frontier.
    pub fn invalidate_lookahead(&mut self) -> bool {
        let refresh = !self.refill_stalled
            && self.selection_error.is_empty()
            && self.refill_error.is_empty()
            && (self.more
                || self.refilling
                || self.loading
                || (self.session.is_empty() && self.unvisited() > 0));
        self.lookahead_generation = self.lookahead_generation.wrapping_add(1);
        let current = self.index;
        let new_index = self
            .items
            .iter()
            .take(current)
            .filter(|item| item.entry_id.is_some())
            .count();
        let mut index = 0;
        self.items.retain(|item| {
            let keep = index == current || item.entry_id.is_some();
            index += 1;
            keep
        });
        self.index = new_index;
        self.furthest = self.items.len().saturating_sub(1);
        self.refilling = false;
        refresh
    }
    pub fn auto_next_request(&self) -> Option<AutoNextRequest> {
        if !self.is_auto() || !self.needs_refill() {
            return None;
        }
        Some(AutoNextRequest {
            session_id: self.session.clone(),
            page_size: LOOKAHEAD.saturating_sub(self.unvisited()),
            exclude_post_ids: self.selection_exclusions(),
        })
    }
    pub fn gallery_next_request(&self) -> Option<GalleryNextRequest> {
        if self.is_auto() || !self.needs_refill() {
            return None;
        }
        Some(GalleryNextRequest {
            session_id: self.session.clone(),
            page_size: LOOKAHEAD.saturating_sub(self.unvisited()),
            exclude_post_ids: self.selection_exclusions(),
        })
    }
    fn unvisited(&self) -> usize {
        self.items.len().saturating_sub(self.furthest + 1)
    }
    pub fn select(&mut self, mode: String, filter: String) {
        self.generation = self.generation.wrapping_add(1);
        self.lookahead_generation = self.lookahead_generation.wrapping_add(1);
        self.mode = mode;
        self.filter = filter;
        self.loading = true;
        self.refilling = false;
        self.more = false;
        self.refill_stalled = false;
        self.context_id.clear();
        self.items.clear();
        self.index = 0;
        self.furthest = 0;
        self.session.clear();
        self.selection_error.clear();
        self.refill_error.clear();
        if self.mode == "rating_review" {
            self.advance = false;
        }
        self.begin_visit();
    }
    fn append_offers(&mut self, items: Vec<Card>) {
        let mut retained: HashSet<_> = self.selection_exclusion_ids().collect();
        for mut card in items {
            if !retained.insert(card.post_id) {
                continue;
            }
            if let Some((_, grade)) = self.acknowledged_grades.get(&card.post_id) {
                card.current_grade = Some(*grade);
            }
            self.items.push(ViewerItem {
                card,
                entry_id: None,
            });
        }
    }
    pub fn selection_owned(
        &mut self,
        generation: u64,
        lookahead: u64,
        result: anyhow::Result<Page>,
    ) -> bool {
        if lookahead != self.lookahead_generation || self.blocked() {
            return false;
        }
        self.selection(generation, result)
    }
    pub fn selection(&mut self, generation: u64, result: anyhow::Result<Page>) -> bool {
        if generation != self.generation {
            return false;
        }
        self.loading = false;
        match result {
            Ok(page) => {
                self.session = page.session_id;
                let had_current = self.current().is_some();
                let before = self.items.len();
                self.append_offers(page.items);
                self.more = !page.exhausted;
                self.refill_stalled = self.items.len() == before;
                if !had_current {
                    self.begin_visit();
                }
                true
            }
            Err(error) => {
                self.selection_error = format!("Selection failed: {error:#}");
                false
            }
        }
    }
    pub fn start_refill(&mut self) -> bool {
        if self.loading
            || self.blocked()
            || self.refilling
            || !self.more
            || !self.refill_error.is_empty()
            || !self.needs_refill()
        {
            return false;
        }
        self.refilling = true;
        true
    }
    /// Existing explicit Next/Retry controls may resume a stalled continuation.
    pub fn resume_refill(&mut self) {
        self.refill_stalled = false;
    }
    pub fn refill_owned(
        &mut self,
        generation: u64,
        lookahead: u64,
        previous: &str,
        result: anyhow::Result<Page>,
    ) {
        if lookahead == self.lookahead_generation && !self.blocked() {
            self.refill(generation, previous, result);
        }
    }
    pub fn refill(&mut self, generation: u64, previous: &str, result: anyhow::Result<Page>) {
        if generation != self.generation || previous != self.session {
            return;
        }
        self.refilling = false;
        match result {
            Ok(page) => {
                let before = self.items.len();
                self.append_offers(page.items);
                self.session = page.session_id;
                self.more = !page.exhausted;
                self.refill_stalled = self.items.len() == before;
            }
            Err(error) => self.refill_error = format!("Upcoming selections failed: {error:#}"),
        }
    }
    pub fn needs_refill(&self) -> bool {
        self.current().is_some()
            && !self.blocked()
            && self.index == self.furthest
            && !self.refill_stalled
            && self.unvisited() <= LOW_WATER
    }
    pub fn begin_visit(&mut self) {
        self.visit = self.visit.wrapping_add(1);
        self.pending_navigation = None;
        self.presentation = Presentation::Unrecorded;
        self.rating = Rating::Idle;
        self.ready_since = None;
        self.retry_media();
    }
    pub fn retry_media(&mut self) {
        self.media_generation = self.media_generation.wrapping_add(1);
        self.media_ready = false;
        self.ready_since = None;
    }
    pub fn media_ready(&mut self, generation: u64, now: Instant) -> bool {
        if self.blocked() || generation != self.media_generation || self.media_ready {
            return false;
        }
        self.media_ready = true;
        self.ready_since = Some(now);
        true
    }
    pub fn navigate(&mut self, delta: isize) -> bool {
        if self.loading || self.blocked() || self.voting() || self.items.is_empty() {
            return false;
        }
        if delta < 0 {
            self.advance = false;
        }
        let next = self
            .index
            .saturating_add_signed(delta)
            .min(self.items.len() - 1);
        if next == self.index {
            return false;
        }
        if next > self.furthest {
            self.refill_stalled = false;
        }
        if !self.context_id.is_empty()
            && !self.viewed()
            && self.view_event().is_none()
            && self.current_entry_id().is_none()
            && delta > 0
        {
            self.items.remove(self.index);
            self.furthest = self.furthest.saturating_sub(1);
            self.index = next
                .saturating_sub(1)
                .min(self.items.len().saturating_sub(1));
            self.furthest = self.furthest.max(self.index);
            self.begin_visit();
            return true;
        }
        self.index = next;
        self.furthest = self.furthest.max(next);
        // Durable history is never trimmed; only selection lookahead is disposable.
        self.begin_visit();
        true
    }
    pub fn start_vote(&mut self, card: Card, grade: Grade) -> bool {
        let scope = (self.generation, self.visit);
        let same_offer = |offer: &Card| {
            offer.post_id == card.post_id
                && offer.session_id == card.session_id
                && offer.epoch == card.epoch
                && offer.position == card.position
                && offer.model_id == card.model_id
        };
        let retry = matches!(
            &self.rating,
            Rating::Failed { command, .. }
                if (command.generation, command.visit) == scope
                    && same_offer(&command.card)
                    && command.grade == grade
        );
        if !self.current().is_some_and(same_offer)
            || self.blocked()
            || self.loading
            || self.voting()
            || !self.view_error().is_empty()
            || (!retry && self.grade_admission() == Admission::Unavailable)
        {
            return false;
        }
        let correction = self
            .current()
            .is_some_and(|offer| offer.current_grade.is_some());
        if correction {
            self.advance = false;
            if self
                .pending_navigation
                .as_ref()
                .is_some_and(|pending| pending.automatic)
            {
                self.pending_navigation = None;
            }
        } else if self.index == self.furthest
            && self.mode != "rating_review"
            && self.pending_navigation.is_none()
        {
            self.pending_navigation = Some(PendingNavigation {
                generation: self.generation,
                visit: self.visit,
                intent: NavigationIntent::Next,
                automatic: true,
            });
        }
        self.vote_attempt = self.vote_attempt.wrapping_add(1);
        let event = match std::mem::take(&mut self.rating) {
            Rating::Failed { command, .. } if command.grade == grade => command.event,
            _ => uuid::Uuid::new_v4().to_string(),
        };
        self.rating = Rating::Pending {
            command: VoteCommand {
                event,
                generation: self.generation,
                visit: self.visit,
                card,
                grade,
            },
        };
        true
    }
    /// Acknowledge persistence without consuming the visit's navigation command.
    pub fn vote(
        &mut self,
        generation: u64,
        visit: u64,
        attempt: u64,
        card: &Card,
        result: anyhow::Result<RatingResult>,
    ) {
        if let Ok(result) = &result {
            let latest = self
                .acknowledged_grades
                .entry(result.post_id)
                .or_insert((result.revision, result.grade));
            if result.revision > latest.0 {
                *latest = (result.revision, result.grade);
            }
        }
        if generation != self.generation {
            return;
        }
        if visit != self.visit
            || attempt != self.vote_attempt
            || !matches!(
                &self.rating,
                Rating::Pending { command }
                    if command.generation == generation
                        && command.visit == visit
                        && command.card.post_id == card.post_id
                        && command.card.session_id == card.session_id
                        && command.card.epoch == card.epoch
                        && command.card.position == card.position
                        && command.card.model_id == card.model_id
            )
        {
            // Global revisions remain useful even after the originating visit ends.
            // Only its accepted in-flight command may change durable-write state.
            if result.is_ok() {
                for item in &mut self.items {
                    if let Some((_, grade)) = self.acknowledged_grades.get(&item.card.post_id) {
                        item.card.current_grade = Some(*grade);
                    }
                }
            }
            return;
        }
        let Rating::Pending { command } = std::mem::take(&mut self.rating) else {
            unreachable!("only an admitted pending rating may be acknowledged");
        };
        self.rating = match result {
            Ok(result) => {
                let grade = self.acknowledged_grades[&result.post_id].1;
                for item in &mut self.items {
                    if item.card.post_id == result.post_id {
                        item.card.current_grade = Some(grade);
                    }
                }
                Rating::Idle
            }
            Err(error) => Rating::Failed {
                command,
                error: format!("Vote not saved: {error:#}"),
            },
        };
    }
    pub fn start_view(&mut self) -> Option<(u64, String, Card)> {
        if self.viewing() || self.viewed() || !self.media_ready {
            return None;
        }
        if self.blocked() {
            return None;
        }
        let card = self.current()?.clone();
        let event = match std::mem::take(&mut self.presentation) {
            Presentation::Failed { event, .. } => event,
            Presentation::Unrecorded => uuid::Uuid::new_v4().to_string(),
            _ => unreachable!("presentation admission excludes pending and recorded visits"),
        };
        self.presentation = Presentation::Pending {
            event: event.clone(),
        };
        self.view_attempt = self.view_attempt.wrapping_add(1);
        Some((self.visit, event, card))
    }
    pub fn view(
        &mut self,
        visit: u64,
        attempt: u64,
        event: &str,
        result: anyhow::Result<()>,
        now: Instant,
    ) {
        if visit != self.visit
            || attempt != self.view_attempt
            || !self.viewing()
            || self.view_event() != Some(event)
        {
            return;
        }
        let Presentation::Pending { event } = std::mem::take(&mut self.presentation) else {
            unreachable!("only an admitted pending presentation may be acknowledged");
        };
        self.presentation = match result {
            Ok(()) => {
                if self.items[self.index].entry_id.is_none() {
                    self.items[self.index].entry_id = Some(event.clone());
                }
                self.ready_since = Some(now);
                Presentation::Recorded { event }
            }
            Err(error) => Presentation::Failed {
                event,
                error: format!("View not recorded: {error:#}"),
            },
        };
    }
    pub fn toggle_advance(&mut self, now: Instant) {
        if !self.blocked() && self.mode != "rating_review" {
            self.advance = !self.advance;
            self.ready_since = Some(now);
        }
    }
    pub fn advance_deadline(&self, delay: u32) -> Option<Instant> {
        if !self.advance
            || self.blocked()
            || self.mode == "rating_review"
            || self.is_video()
            || !self.media_ready
            || !self.viewed()
            || self.voting()
            || self.viewing()
            || self.pending_navigation.is_some()
            || !self.view_error().is_empty()
            || !self.vote_error().is_empty()
            || self.loading
            || self.index + 1 >= self.items.len()
        {
            return None;
        }
        self.ready_since
            .map(|start| start + Duration::from_secs(delay as u64))
    }
    pub fn timed_advance(&mut self, now: Instant, delay: u32) -> bool {
        self.advance_deadline(delay).is_some_and(|due| now >= due) && self.navigate(1)
    }
}

pub fn fresh_seed() -> i32 {
    let bytes = *uuid::Uuid::new_v4().as_bytes();
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
#[path = "../tests/unit/viewer/mod.rs"]
mod tests;

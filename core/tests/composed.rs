//! CI-only owner-level smoke; scripts/verify-ci.sh supplies its single online backup.
use anyhow::{Context, Result, bail, ensure};
use ff_core::{
    app::{AppConfig, AppHandle},
    corpus,
    model::ModelRoute,
    store::Store,
    types::{AutoStartRequest, Card, Gallery, GalleryStartRequest, Grade},
    viewer::{NavigationIntent, ViewerState},
};
use rusqlite::{Connection, OpenFlags, params};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::time::{Instant, timeout};

const DEADLINE: Duration = Duration::from_secs(180);
const FAILURE: &str = "ff-ci-composed-rollback";

fn config() -> Result<AppConfig> {
    ensure!(
        std::env::var("CI").as_deref() == Ok("true"),
        "Composed smoke requires CI=true"
    );
    ensure!(
        std::env::var("FF_CI_COMPOSED_SMOKE").as_deref() == Ok("true"),
        "Enable composed smoke explicitly through scripts/verify-ci.sh"
    );
    let state = std::env::var_os("FF_CI_STATE_DIR")
        .context("Explicit owned FF_CI_STATE_DIR is required")?;
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("Repository root")?;
    let root = repository.join(".workdir/ci").canonicalize()?;
    let state = PathBuf::from(state).canonicalize()?;
    ensure!(
        state.parent() == Some(root.as_path())
            && state
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("native-")),
        "State must be a script-owned native-* CI directory"
    );
    let fixture = PathBuf::from(
        std::env::var_os("FF_CI_INTERACTION_PATH")
            .context("Explicit non-personal FF_CI_INTERACTION_PATH is required")?,
    )
    .canonicalize()?;
    let interaction_path = state.join("interactions.sqlite3").canonicalize()?;
    ensure!(
        interaction_path.parent() == Some(state.as_path()) && interaction_path != fixture,
        "Use the script's isolated online backup, never the source fixture"
    );
    let settings_path = state.join("config.toml");
    ensure!(
        !settings_path.is_symlink(),
        "CI settings cannot be a symlink"
    );
    let corpus_path = match std::env::var_os("FF_CORPUS_PATH").filter(|path| !path.is_empty()) {
        Some(path) => PathBuf::from(path),
        None => corpus::resolve_corpus(repository)?,
    };
    Ok(AppConfig {
        corpus_path,
        interaction_path,
        settings_path,
    })
}

// Hash exact stored offer bytes, including genuinely historical policies if present.
// The original sequence bound excludes this smoke's new commands; family and
// sequence/ordinal retain stable identities even for overlapping imported offers.
fn offer_digest(db: &Connection, family: &str, upper: i64) -> Result<[u8; 32]> {
    let mut statement = db.prepare(
        "SELECT sequence,ordinal,payload_json FROM offer_facts WHERE family=?1 AND sequence<=?2 ORDER BY sequence,ordinal"
    )?;
    let mut rows = statement.query(params![family, upper])?;
    let mut hash = Sha256::new();
    hash.update((family.len() as u64).to_le_bytes());
    hash.update(family.as_bytes());
    while let Some(row) = rows.next()? {
        let sequence: i64 = row.get(0)?;
        let ordinal: i64 = row.get(1)?;
        let payload: String = row.get(2)?;
        hash.update(sequence.to_le_bytes());
        hash.update(ordinal.to_le_bytes());
        hash.update((payload.len() as u64).to_le_bytes());
        hash.update(payload.as_bytes());
    }
    Ok(hash.finalize().into())
}

fn gallery_digest(db: &Connection) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    for sql in [
        "SELECT json_array(sequence,ordinal,session_id,operation,created_at,updated_at,status,mode,filter_text,model_id,requested_size,returned_size,cursor,exhausted,seed) FROM selection_facts WHERE family='gallery' ORDER BY sequence,ordinal",
        "SELECT json_array(sequence,ordinal,session_id,position,post_id,facet,payload_json,card_json,selection_policy,initial_state,initial_delivered_at) FROM offer_facts WHERE family='gallery' ORDER BY sequence,ordinal",
        "SELECT json_array(sequence,ordinal,origin,post_id,action,first_rated_at,updated_at,revision,context_json,family,session_id,predicted_grade,recorded_action,rating_revision,judgment_model_id,checkpoint_review_sequence,checkpoint_updated_at) FROM rating_facts WHERE family='gallery' ORDER BY sequence,ordinal",
        "SELECT json_array(sequence,ordinal,kind,event_id,session_id,post_id,viewed_at,context_id,entry_id,new_entry,entry_position,offer_family,offer_position,view_family,seed_card_json,is_view) FROM presentation_facts WHERE view_family='gallery' OR offer_family='gallery' ORDER BY sequence,ordinal",
    ] {
        let mut statement = db.prepare(sql)?;
        for row in statement.query_map([], |row| row.get::<_, String>(0))? {
            let row = row?;
            hash.update((row.len() as u64).to_le_bytes());
            hash.update(row.as_bytes());
        }
    }
    Ok(hash.finalize().into())
}

fn payloads(db: &Connection, family: &str, session: &str) -> Result<Vec<String>> {
    let mut statement = db.prepare(
        "SELECT payload_json FROM offer_facts WHERE family=?1 AND session_id=?2 ORDER BY position",
    )?;
    Ok(statement
        .query_map(params![family, session], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?)
}

fn cards_bytes<'a>(cards: impl ExactSizeIterator<Item = &'a Card>) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::new(&mut bytes);
    let mut sequence = serde::Serializer::serialize_seq(&mut serializer, Some(cards.len()))?;
    for card in cards {
        serde::ser::SerializeSeq::serialize_element(&mut sequence, card)?;
    }
    serde::ser::SerializeSeq::end(sequence)?;
    Ok(bytes)
}

fn assert_offers(
    cards: &[Card],
    session: &str,
    start: usize,
    count: usize,
    model: &str,
) -> Result<()> {
    ensure!(
        cards.len() == count,
        "Expected {count} issued offers, got {}",
        cards.len()
    );
    for (offset, card) in cards.iter().enumerate() {
        ensure!(
            card.session_id == session && card.position == start + offset && card.model_id == model,
            "Offer group, global position or serving provenance changed"
        );
    }
    Ok(())
}

fn assert_cohort(
    db: &Connection,
    session: &str,
    model: &str,
    capacity: usize,
    issued: usize,
    completed: bool,
) -> Result<()> {
    let header: (String, usize, usize, String, bool) = db.query_row(
        "SELECT model_id,requested_size,returned_size,status,(SELECT exhausted FROM selection_facts WHERE family='gallery' AND session_id=?1 AND exhausted IS NOT NULL ORDER BY sequence DESC,ordinal DESC LIMIT 1) FROM q_gallery_sessions WHERE session_id=?1",
        [session], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)))?;
    ensure!(
        header
            == (
                model.to_owned(),
                capacity,
                issued,
                if completed { "completed" } else { "active" }.to_owned(),
                false
            ),
        "Cohort header/capacity/count/completion/exhaustion mismatch: {header:?}"
    );
    let positions: (usize, i64, i64) = db.query_row(
        "SELECT COUNT(*),MIN(position),MAX(position) FROM offer_facts WHERE family='gallery' AND session_id=?1",
        [session],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    ensure!(
        positions == (issued, 0, issued as i64 - 1),
        "Persisted cohort positions are not contiguous"
    );
    Ok(())
}

fn frontier(viewer: &mut ViewerState) -> Result<()> {
    // Progress is determined by the viewer's semantic low-water mark, not sleeps.
    for _ in 0..5 {
        if viewer.needs_refill() {
            return Ok(());
        }
        ensure!(
            viewer.navigate(1),
            "No progress to the next refill boundary"
        );
    }
    bail!("Viewer failed to reach its bounded low-water boundary")
}

async fn publication(app: &AppHandle, previous: &str) -> Result<String> {
    let due = Instant::now() + DEADLINE;
    loop {
        let health = timeout(due.saturating_duration_since(Instant::now()), app.health())
            .await
            .context("Deadline waiting for model health")??;
        ensure!(
            health.refresh_error.is_none(),
            "Actual refresh failed: {:?}",
            health.refresh_error
        );
        if health.model_id != previous && !health.refreshing {
            ensure!(
                health.model_status.route == ModelRoute::Ensemble,
                "Fixture lost supported ensemble route"
            );
            return Ok(health.model_id);
        }
        ensure!(
            Instant::now() < due,
            "Actual model publication deadline elapsed"
        );
        // Poll cadence is not a correctness assertion; publication is the condition.
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn refill(app: &AppHandle, viewer: &mut ViewerState) -> Result<Gallery> {
    let request = viewer
        .gallery_next_request()
        .context("Gallery deficit request")?;
    let previous = viewer.session.clone();
    let current = cards_bytes(std::iter::once(viewer.current().context("Current card")?))?;
    let visit = viewer.visit;
    let media = viewer.media_generation;
    let retained_len = viewer.items.len();
    let retained = cards_bytes(viewer.items.iter().map(|item| &item.card))?;
    ensure!(viewer.start_refill(), "Viewer refused refill");
    let page = app.next_gallery(request).await?;
    viewer.refill(viewer.generation, &previous, Ok(page.clone().into()));
    ensure!(
        cards_bytes(std::iter::once(
            viewer.current().context("Retained current")?
        ))? == current
            && viewer.visit == visit
            && viewer.media_generation == media
            && cards_bytes(
                viewer
                    .items
                    .get(..retained_len)
                    .context("Retained Gallery history was lost")?
                    .iter()
                    .map(|item| &item.card)
            )? == retained,
        "Refill changed the current visit, media attempt or a retained offer"
    );
    Ok(page)
}

async fn rollback_retry(
    app: &AppHandle,
    db: &Connection,
    viewer: &mut ViewerState,
    boundary: bool,
) -> Result<Gallery> {
    let request = viewer
        .gallery_next_request()
        .context("Rollback deficit request")?;
    // Failure is inside the actor-owned SQLite transaction. Append fails on its
    // second insert; boundary fails after inserting the new cohort and its rows.
    let sql = if boundary {
        "CREATE TRIGGER ff_ci_composed_failure BEFORE INSERT ON selection_facts WHEN NEW.family='gallery' AND NEW.session_id=?1 AND NEW.operation='retire' BEGIN SELECT RAISE(ABORT,'ff-ci-composed-rollback'); END"
    } else {
        "CREATE TRIGGER ff_ci_composed_failure BEFORE INSERT ON offer_facts WHEN NEW.family='gallery' AND NEW.session_id=?1 AND NEW.position=8 BEGIN SELECT RAISE(ABORT,'ff-ci-composed-rollback'); END"
    };
    // SQLite DDL does not accept parameters; quote the genuine generated owner ID.
    let session_literal = format!("'{}'", viewer.session.replace('\'', "''"));
    db.execute_batch(&sql.replace("?1", &session_literal))?;
    let before = gallery_digest(db)?;
    let cards = cards_bytes(viewer.items.iter().map(|item| &item.card))?;
    let previous = viewer.session.clone();
    let visit = viewer.visit;
    ensure!(viewer.start_refill(), "Viewer refused failure probe");
    let result = app.next_gallery(request.clone()).await;
    // Always remove the owned trigger, even when the expected failure is absent.
    db.execute_batch("DROP TRIGGER ff_ci_composed_failure")?;
    let error = match result {
        Err(error) => error,
        Ok(_) => bail!("SQLite trigger did not reject the real commit"),
    };
    ensure!(
        format!("{error:#}").contains(FAILURE),
        "Unexpected failure: {error:#}"
    );
    viewer.refill(viewer.generation, &previous, Err(error));
    ensure!(
        gallery_digest(db)? == before,
        "Failed transaction changed durable cohort state"
    );
    ensure!(
        viewer.session == previous
            && viewer.visit == visit
            && cards_bytes(viewer.items.iter().map(|item| &item.card))? == cards,
        "Failed continuation changed viewer ownership/history"
    );
    let retry = viewer
        .gallery_next_request()
        .context("Original continuation must remain valid")?;
    ensure!(
        retry.session_id == request.session_id
            && retry.page_size == request.page_size
            && retry.exclude_post_ids == request.exclude_post_ids,
        "Rollback changed the retry request"
    );
    viewer.refill_error.clear();
    viewer.resume_refill();
    refill(app, viewer).await
}

async fn scenario(app: &AppHandle, path: &Path) -> Result<()> {
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
    db.busy_timeout(Duration::from_secs(5))?;
    let history = ["gallery", "auto"]
        .into_iter()
        .map(|family| -> Result<_> {
            let upper: i64 = db.query_row(
                "SELECT COALESCE(MAX(sequence),0) FROM offer_facts WHERE family=?1",
                [family],
                |row| row.get(0),
            )?;
            Ok((family, upper, offer_digest(&db, family, upper)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let initial = app.health().await?;
    ensure!(
        initial.model_status.route == ModelRoute::Ensemble && !initial.refreshing,
        "FF_CI_INTERACTION_PATH must supply rich non-personal ratings with actual Ensemble support"
    );
    let m0 = initial.model_id;
    let capacity = app
        .facets()
        .await?
        .into_iter()
        .find(|facet| facet.slug == "train_broadly")
        .context("Production Train Broadly facet")?
        .cohort_size;
    let mut gallery = ViewerState::default();
    gallery.select("train_broadly".into(), String::new());
    let first = app
        .start_gallery(GalleryStartRequest {
            facet: gallery.mode.clone(),
            page_size: gallery.selection_page_size(),
            filter: gallery.filter.clone(),
            exclude_post_ids: vec![],
        })
        .await?;
    let cohort = first.session_id.clone();
    assert_offers(&first.items, &cohort, 0, 5, &m0)
        .context("CI fixture/corpus must have enough eligible Train Broadly rows for a full cohort and rollover")?;
    ensure!(
        first.items.iter().all(|card| card.universe_size == 5) && !first.exhausted,
        "Initial Gallery offer window mismatch"
    );
    ensure!(
        gallery.selection(gallery.generation, Ok(first.into())),
        "Adopt initial Gallery"
    );
    ensure!(
        gallery.gallery_next_request().is_none(),
        "Initial five must not trigger another page"
    );
    let old_gallery = payloads(&db, "gallery", &cohort)?;
    assert_cohort(&db, &cohort, &m0, capacity, 5, false)?;

    let mut auto = ViewerState::default();
    auto.select("balanced".into(), String::new());
    let auto_first = app
        .start_auto(AutoStartRequest {
            preset: auto.mode.clone(),
            page_size: auto.selection_page_size(),
            seed: 314159,
            filter: auto.filter.clone(),
            exclude_post_ids: vec![],
        })
        .await?;
    let auto_session = auto_first.session_id.clone();
    let epoch = auto_first.epoch;
    assert_offers(&auto_first.items, &auto_session, 0, 5, &m0)
        .context("CI fixture/corpus must have at least seven eligible balanced Auto offers")?;
    ensure!(
        auto_first.emitted_count == 5 && !auto_first.exhausted,
        "Auto initial emission mismatch"
    );
    ensure!(
        auto.selection(auto.generation, Ok(auto_first.into())),
        "Adopt initial Auto"
    );
    let old_auto = payloads(&db, "auto", &auto_session)?;
    let auto_retained = cards_bytes(auto.items.iter().map(|item| &item.card))?;
    println!("composed-smoke phase complete: M0 Gallery five and Auto five issued");

    let voted = gallery.current().context("Gallery rating card")?.clone();
    ensure!(
        voted.current_grade.is_none(),
        "Fixture needs an unrated Gallery offer"
    );
    // This headless consumer acknowledges its presentation explicitly; actual
    // decoder/media acceptance belongs to the desktop verification path.
    gallery.media_ready(gallery.media_generation, std::time::Instant::now());
    let (visit, view_event, viewed) = gallery.start_view().context("Begin offered view")?;
    let view_attempt = gallery.view_attempt;
    app.viewed(view_event.clone(), viewed).await?;
    gallery.view(
        visit,
        view_attempt,
        &view_event,
        Ok(()),
        std::time::Instant::now(),
    );
    ensure!(
        gallery.start_vote(voted.clone(), Grade::Love),
        "Start real vote"
    );
    let vote_attempt = gallery.vote_attempt;
    let vote_event = gallery
        .vote_event()
        .map(str::to_owned)
        .context("Accepted grade identity")?;
    let saved = app
        .rate(vote_event.clone(), voted.clone(), Grade::Love)
        .await?;
    ensure!(
        saved.changed && saved.grade == Grade::Love,
        "Real changed rating was not saved"
    );
    let revision = saved.revision;
    let stored = Store::open(path)?
        .ratings()?
        .into_iter()
        .find(|rating| rating.post_id == voted.post_id)
        .context("Saved rating must be durable")?;
    ensure!(
        stored.grade == Grade::Love && stored.revision == revision,
        "Durable rating mismatch"
    );
    let context: serde_json::Value = serde_json::from_str(&stored.context_json)?;
    ensure!(
        context["gallery_session_id"] == cohort
            && context["model_id"] == m0
            && context["selection_policy"] == ff_core::selection::GALLERY_POLICY,
        "Saved rating lost actual offer provenance"
    );
    gallery.vote(
        gallery.generation,
        gallery.visit,
        vote_attempt,
        &voted,
        Ok(saved),
    );
    ensure!(
        gallery.take_navigation() == Some(NavigationIntent::Next) && gallery.navigate(1),
        "First saved vote must advance once"
    );
    let m1 = publication(app, &m0).await?;
    let retry = app.rate(vote_event, voted.clone(), Grade::Love).await?;
    ensure!(
        !retry.changed && retry.revision == revision,
        "Exact vote retry changed history"
    );
    ensure!(
        payloads(&db, "gallery", &cohort)? == old_gallery,
        "Publication rewrote M0 offers"
    );
    println!("composed-smoke phase complete: real saved rating and M1 publication ({m0} -> {m1})");

    frontier(&mut gallery)?;
    let request = gallery
        .gallery_next_request()
        .context("Low-water deficit")?;
    ensure!(
        request.page_size == 2 && request.exclude_post_ids.len() == 5,
        "Refill must request only the two-card deficit"
    );
    let next = refill(app, &mut gallery).await?;
    ensure!(
        next.session_id == cohort && !next.exhausted,
        "Publication must not roll Gallery cohort"
    );
    assert_offers(&next.items, &cohort, 5, 2, &m1)?;
    ensure!(
        next.items.iter().all(|card| card.universe_size == 7),
        "Appended universe must be issued count"
    );
    assert_cohort(&db, &cohort, &m0, capacity, 7, false)?;
    ensure!(
        payloads(&db, "gallery", &cohort)?.get(..5) == Some(old_gallery.as_slice()),
        "M0 payload bytes changed on M1 append"
    );
    let current = gallery.current().context("Explicit viewed card")?.clone();
    app.viewed(uuid::Uuid::new_v4().to_string(), current)
        .await?;
    ensure!(
        gallery.navigate(-1) && gallery.gallery_next_request().is_none() && gallery.navigate(1),
        "Backtracking must not request a page"
    );
    println!(
        "composed-smoke phase complete: same-cohort M1 two-card deficit and immutable M0 provenance"
    );

    frontier(&mut gallery)?;
    let retried = rollback_retry(app, &db, &mut gallery, false).await?;
    assert_offers(&retried.items, &cohort, 7, 2, &m1)?;
    assert_cohort(&db, &cohort, &m0, capacity, 9, false)?;
    for _ in 0..capacity {
        if gallery.items.len() == capacity {
            break;
        }
        frontier(&mut gallery)?;
        let issued = gallery.items.len();
        let page = refill(app, &mut gallery).await?;
        ensure!(
            page.session_id == cohort && !page.exhausted,
            "CI fixture/corpus must support a complete Train Broadly cohort before genuine rollover"
        );
        assert_offers(&page.items, &cohort, issued, 2.min(capacity - issued), &m1)
            .context("CI fixture/corpus has insufficient eligible rows for the required full-cohort boundary")?;
        ensure!(
            page.items
                .iter()
                .all(|card| card.universe_size == gallery.items.len()),
            "Wrong issued universe"
        );
    }
    ensure!(
        gallery.items.len() == capacity,
        "Bounded cohort did not fill"
    );
    assert_cohort(&db, &cohort, &m0, capacity, capacity, true)?;
    frontier(&mut gallery)?;
    let transition = rollback_retry(app, &db, &mut gallery, true).await?;
    ensure!(
        transition.session_id != cohort && !transition.exhausted,
        "Genuine continuation must start a new cohort"
    );
    assert_offers(&transition.items, &transition.session_id, 0, 2, &m1)?;
    assert_cohort(&db, &cohort, &m0, capacity, capacity, true)?;
    assert_cohort(&db, &transition.session_id, &m1, capacity, 2, false)?;
    ensure!(
        payloads(&db, "gallery", &cohort)?.get(..5) == Some(old_gallery.as_slice()),
        "Boundary rewrote M0 bytes"
    );
    println!(
        "composed-smoke phase complete: atomic append/boundary rollback, original-token retries and real cohort rollover"
    );

    frontier(&mut auto)?;
    let auto_request = auto.auto_next_request().context("Auto deficit request")?;
    ensure!(
        auto_request.page_size == 2 && auto_request.exclude_post_ids.len() == 5,
        "Auto must request only deficit"
    );
    let visit = auto.visit;
    let current = cards_bytes(std::iter::once(auto.current().context("Current Auto")?))?;
    ensure!(auto.start_refill(), "Start Auto refill");
    let page = app.next_auto(auto_request).await?;
    ensure!(
        page.session_id == auto_session
            && page.epoch == epoch
            && page.emitted_count == 7
            && !page.exhausted,
        "Generation publication reset Auto session/epoch/emission history"
    );
    assert_offers(&page.items, &auto_session, 5, 2, &m1)?;
    ensure!(
        page.items.iter().all(|card| card.epoch == epoch),
        "Auto card epoch changed"
    );
    auto.refill(auto.generation, &auto_session, Ok(page.into()));
    ensure!(
        auto.visit == visit
            && cards_bytes(std::iter::once(auto.current().context("Retained Auto")?))? == current
            && cards_bytes(
                auto.items
                    .get(..5)
                    .context("Retained Auto history was lost")?
                    .iter()
                    .map(|item| &item.card)
            )? == auto_retained,
        "Auto refill changed current visit/retained provenance"
    );
    ensure!(
        payloads(&db, "auto", &auto_session)?.get(..5) == Some(old_auto.as_slice()),
        "Auto publication rewrote M0 bytes"
    );
    let emitted: usize = db.query_row(
        "SELECT returned_size FROM q_auto_sessions WHERE session_id=?1",
        params![auto_session],
        |row| row.get(0),
    )?;
    ensure!(emitted == 7, "Durable Auto emission count mismatch");
    for (family, upper, digest) in history {
        ensure!(
            offer_digest(&db, family, upper)? == digest,
            "Historical {family} payload bytes changed"
        );
    }
    println!(
        "composed-smoke phase complete: bounded Auto M0-to-M1 continuation and historical offer-byte preservation"
    );
    Store::open(path)?
        .verify_history()
        .context("Complete composed journal/fact verification")?;
    println!("composed-smoke phase complete: complete journal/fact history verified");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "Full-data CI only; run through scripts/verify-ci.sh"]
async fn composed_smoke() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ff_core=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let config = config()?;
    let path = config.interaction_path.clone();
    // Keep initialization owned if its deadline expires: recover the eventual
    // handle and shut it down rather than abandoning a starting actor.
    let mut opening = tokio::spawn(AppHandle::open(config));
    let app = match timeout(DEADLINE, &mut opening).await {
        Ok(result) => result.context("Backend initialization task")??,
        Err(_) => {
            if let Ok(Ok(app)) = opening.await {
                app.shutdown().await?;
            }
            bail!("Backend initialization exceeded its deadline (owned startup drained)");
        }
    };
    let result = timeout(DEADLINE, scenario(&app, &path))
        .await
        .context("Composed scenario deadline elapsed")
        .and_then(|result| result);
    // Cancellation/failure still drains admitted production work before the
    // script may delete its backup. A deadline failure is not permission to leak.
    let mut closing = std::pin::pin!(app.shutdown());
    let shutdown = match timeout(DEADLINE, &mut closing).await {
        Ok(result) => result,
        Err(_) => {
            closing.await.context("Drain overdue shutdown")?;
            Err(anyhow::anyhow!(
                "Shutdown exceeded its deadline (owned work drained)"
            ))
        }
    };
    if let Err(error) = &result {
        eprintln!("composed-smoke scenario failed: {error:#}");
    }
    shutdown.context("Clean production-owner shutdown")?;
    result?;
    println!("composed-smoke phase complete: clean shutdown; all owner-level assertions passed");
    Ok(())
}

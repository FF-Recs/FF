//! Deterministic event details and on-demand reads. No mutable projections live here.
use super::*;

#[derive(Deserialize)]
struct NativeOffer {
    format_version: u8,
    selection_policy: String,
    card: Card,
}

// Card's f32 encoding is deliberately produced by Card's serializer, never by
// extracting/re-encoding a serde_json::Value (which would widen its number).
fn canonical_offer(payload: &str) -> Result<(Option<String>, Option<String>)> {
    match serde_json::from_str::<NativeOffer>(payload) {
        Ok(offer) => {
            ensure!(
                offer.format_version == 1,
                "Unsupported native offered-card format"
            );
            Ok((
                Some(serde_json::to_string(&offer.card)?),
                Some(offer.selection_policy),
            ))
        }
        Err(error) => {
            // Only opaque imported offers take this path. Malformed native
            // evidence must not silently turn into an unpresentable offer.
            let value = serde_json::from_str::<Value>(payload).ok();
            ensure!(
                value
                    .as_ref()
                    .is_none_or(|value| value.get("format_version").is_none()),
                "Invalid native offered card: {error}"
            );
            Ok((None, None))
        }
    }
}

fn initial(connection: &Connection, event_sequence: i64) -> Result<()> {
    ensure!(
        event_sequence == 1,
        "Initial interaction event must have sequence 1"
    );
    let occupied: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM interaction_events UNION ALL SELECT 1 FROM selection_facts UNION ALL SELECT 1 FROM offer_facts UNION ALL SELECT 1 FROM rating_facts UNION ALL SELECT 1 FROM review_facts UNION ALL SELECT 1 FROM presentation_facts UNION ALL SELECT 1 FROM activation_facts UNION ALL SELECT 1 FROM preference_facts UNION ALL SELECT 1 FROM legacy_delivery_facts)",
        [], |row| row.get(0),
    )?;
    ensure!(!occupied, "Initial event requires empty fact history");
    Ok(())
}

pub(super) fn apply(
    connection: &Connection,
    event_id: &str,
    event_sequence: i64,
    timestamp: f64,
    event: &Event,
) -> Result<Value> {
    match event {
        Event::Initialized { facts } => {
            initial(connection, event_sequence)?;
            seed::apply(connection, event_sequence, facts)?;
        }
        Event::GalleryCommitted {
            owner,
            model_id,
            offers,
            completed,
            exhausted,
            previous,
        } => {
            let owner = owner.borrowed();
            required_text(owner.session, "Gallery session ID")?;
            required_text(owner.facet, "Gallery facet")?;
            required_text(model_id, "Gallery model ID")?;
            ensure!(owner.capacity > 0, "Gallery capacity must be positive");
            let end = owner
                .issued
                .checked_add(offers.len())
                .context("Gallery position overflow")?;
            ensure!(
                end <= owner.capacity,
                "Gallery offers exceed cohort capacity"
            );
            ensure!(
                !exhausted || *completed,
                "Global exhaustion must close the cohort"
            );
            ensure!(
                end < owner.capacity || *completed,
                "A full cohort must be completed"
            );
            validate_offers(owner.session, owner.issued, offers)?;
            ensure!(
                offers.iter().all(|offer| offer.facet == owner.facet),
                "Gallery card facet disagrees with its session"
            );
            if owner.issued == 0 {
                ensure!(!offers.is_empty(), "Cannot create an empty Gallery cohort");
                ensure_new_session(connection, owner.session)?;
                connection.execute(
                    "INSERT INTO selection_facts(sequence,ordinal,family,session_id,operation,created_at,updated_at,status,mode,filter_text,model_id,requested_size,returned_size,cursor,exhausted) VALUES(?1,0,'gallery',?2,'create',?3,?3,?4,?5,?6,?7,?8,?9,-1,?10)",
                    params![event_sequence, owner.session, timestamp, if *completed { "completed" } else { "active" }, owner.facet, owner.filter, model_id, i64::try_from(owner.capacity)?, i64::try_from(end)?, exhausted],
                )?;
            } else {
                validate_gallery_owner(connection, &owner, offers.is_empty())?;
                connection.execute(
                    "INSERT INTO selection_facts(sequence,ordinal,family,session_id,operation,updated_at,status,returned_size,exhausted) VALUES(?1,0,'gallery',?2,'append',?3,?4,?5,?6)",
                    params![event_sequence, owner.session, timestamp, if *completed { "completed" } else { "active" }, i64::try_from(end)?, exhausted],
                )?;
            }
            insert_offers(connection, event_sequence, "gallery", owner.session, offers)?;
            if let Some((previous, completed)) = previous {
                ensure!(
                    previous.session != owner.session,
                    "Gallery transition must change cohort identity"
                );
                retire(
                    connection,
                    event_sequence,
                    1,
                    &previous.borrowed(),
                    *completed,
                    timestamp,
                )?;
            }
        }
        Event::GalleryRetired { owner } => {
            retire(
                connection,
                event_sequence,
                0,
                &owner.borrowed(),
                false,
                timestamp,
            )?;
        }
        Event::AutoCreated {
            session,
            preset,
            filter,
            model_id,
            seed,
            offers,
        } => {
            required_text(session, "Auto session ID")?;
            required_text(model_id, "Auto model ID")?;
            ensure!(
                ["favorites", "balanced", "fresh"].contains(&preset.as_str()),
                "Unknown Auto preset '{preset}'"
            );
            validate_offers(session, 0, offers)?;
            ensure_new_session(connection, session)?;
            connection.execute(
                "INSERT INTO selection_facts(sequence,ordinal,family,session_id,operation,created_at,updated_at,status,model_id,mode,filter_text,seed,returned_size,cursor) VALUES(?1,0,'auto',?2,'create',?3,?3,'active',?4,?5,?6,?7,?8,-1)",
                params![event_sequence, session, timestamp, model_id, preset, filter, seed, i64::try_from(offers.len())?],
            )?;
            insert_offers(connection, event_sequence, "auto", session, offers)?;
        }
        Event::AutoAppended {
            session,
            start,
            offers,
        } => {
            required_text(session, "Auto session ID")?;
            ensure!(!offers.is_empty(), "An Auto append cannot be empty");
            validate_offers(session, *start, offers)?;
            let end = i64::try_from(
                start
                    .checked_add(offers.len())
                    .context("Auto position overflow")?,
            )?;
            let previous: Option<(i64, String)> = connection
                .query_row(
                    "SELECT returned_size,status FROM q_auto_sessions WHERE session_id=?1",
                    [session],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            let (returned, status) = previous.context("Unknown Auto session")?;
            ensure!(status == "active", "Auto session is not active");
            ensure!(
                returned == i64::try_from(*start)?,
                "Auto append position is stale"
            );
            insert_offers(connection, event_sequence, "auto", session, offers)?;
            connection.execute(
                "INSERT INTO selection_facts(sequence,ordinal,family,session_id,operation,updated_at,returned_size) VALUES(?1,0,'auto',?2,'append',?3,?4)",
                params![event_sequence, session, timestamp, end],
            )?;
        }
        Event::RatingRecorded {
            post_id,
            grade,
            session,
            prediction,
        } => {
            return reduce_rating(
                connection,
                event_sequence,
                *post_id,
                *grade,
                session,
                prediction
                    .as_ref()
                    .map(|p| (p.predicted_grade, p.model_id.as_str())),
                timestamp,
            );
        }
        Event::Viewed { post_id, session } => {
            record_view(
                connection,
                event_id,
                event_sequence,
                *post_id,
                session,
                timestamp,
            )?;
        }
        Event::PreferencesSaved { settings } => {
            connection.execute(
                "INSERT INTO preference_facts(sequence,ordinal,settings_json) VALUES(?1,0,?2)",
                params![event_sequence, serde_json::to_string(settings)?],
            )?;
        }
        Event::ContextActivated {
            mode,
            filter,
            fresh,
        } => {
            required_text(mode, "Browse mode")?;
            let previous: Option<String> = if *fresh {
                None
            } else {
                connection.query_row(
                    "SELECT context_id FROM q_browse_contexts WHERE mode=?1 AND filter_text=?2 ORDER BY activated_sequence DESC,context_id LIMIT 1",
                    params![mode, filter], |row| row.get(0),
                ).optional()?
            };
            let context_id = previous.as_deref().unwrap_or(event_id);
            connection.execute(
                "INSERT INTO activation_facts(sequence,ordinal,origin,context_id,mode,filter_text,activated_sequence) VALUES(?1,0,'activation',?2,?3,?4,?1)",
                params![event_sequence, context_id, mode, filter],
            )?;
            return Ok(json!({"context_id":context_id}));
        }
        Event::Presented {
            context_id,
            entry_id,
            card,
        } => {
            present(
                connection,
                event_id,
                event_sequence,
                context_id,
                entry_id.as_deref(),
                card,
                timestamp,
            )?;
        }
    }
    Ok(Value::Null)
}

fn insert_offers(
    connection: &Connection,
    sequence: i64,
    family: &str,
    session: &str,
    offers: &[OfferFact],
) -> Result<()> {
    let mut insert = connection.prepare(
        "INSERT INTO offer_facts(sequence,ordinal,family,session_id,position,post_id,facet,payload_json,card_json,selection_policy,initial_state) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
    )?;
    for (ordinal, offer) in offers.iter().enumerate() {
        let (card, policy) = canonical_offer(&offer.payload_json)?;
        insert.execute(params![
            sequence,
            i64::try_from(ordinal)?,
            family,
            session,
            i64::try_from(offer.position)?,
            offer.post_id,
            offer.facet,
            offer.payload_json,
            card,
            policy,
            if family == "gallery" {
                Some("selected")
            } else {
                None
            }
        ])?;
    }
    Ok(())
}

fn reduce_rating(
    connection: &Connection,
    sequence: i64,
    post_id: i64,
    grade: Grade,
    session: &str,
    prediction: Option<(u8, &str)>,
    timestamp: f64,
) -> Result<Value> {
    if let Some((predicted_grade, model_id)) = prediction {
        Grade::from_u8(predicted_grade)?;
        required_text(model_id, "Judgment model ID")?;
    }
    let context = context(connection, session, post_id)?;
    let previous: Option<(String, String, i64, f64)> = connection
        .query_row(
            "SELECT action,context_json,revision,first_rated_at FROM q_ratings WHERE post_id=?1",
            [post_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()?;
    if let Some((action, json, revision, _)) = &previous {
        if action == grade.as_str() && json == &context.json {
            return Ok(json!({"revision":revision,"changed":false}));
        }
    }
    let (revision, first_rated_at) = match previous {
        Some((_, _, revision, first)) => (
            revision
                .checked_add(1)
                .context("Rating revision overflow")?,
            first,
        ),
        None => (1, timestamp),
    };
    let is_review = context.facet.as_deref() == Some("rating_review");
    let mut review_sequence = review_sequence(connection)?;
    if is_review {
        let voted: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM review_facts WHERE kind IN ('vote_seed','vote') AND session_id=?1 AND post_id=?2)",
            params![session, post_id], |row| row.get(0),
        )?;
        if !voted {
            review_sequence = review_sequence
                .checked_add(1)
                .context("Review sequence overflow")?;
            connection.execute(
                "INSERT INTO review_facts(sequence,ordinal,kind,post_id,session_id,review_sequence) VALUES(?1,0,'vote',?2,?3,?4)",
                params![sequence, post_id, session, review_sequence],
            )?;
        }
    }
    connection.execute(
        "INSERT INTO rating_facts(sequence,ordinal,origin,post_id,action,first_rated_at,updated_at,revision,context_json,family,session_id,predicted_grade,recorded_action,rating_revision,judgment_model_id,checkpoint_review_sequence,checkpoint_updated_at) VALUES(?1,0,'change',?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
        params![sequence, post_id, grade.as_str(), first_rated_at, timestamp, revision, context.json, if context.is_auto { "auto" } else { "gallery" }, session, prediction.map(|p| p.0), prediction.map(|_| grade.as_str()), prediction.map(|_| revision), prediction.map(|p| p.1), prediction.map(|_| review_sequence), prediction.map(|_| timestamp)],
    )?;
    if is_review && grade == Grade::Dislike {
        let offered: Value;
        let model_id = if let Some((_, model_id)) = prediction {
            model_id
        } else {
            offered = serde_json::from_str(&context.json)?;
            offered["model_id"]
                .as_str()
                .context("Review offer lacks model provenance")?
        };
        connection.execute(
            "INSERT INTO review_facts(sequence,ordinal,kind,post_id,session_id,model_id,excluded_at) VALUES(?1,1,'exclusion',?2,?3,?4,?5)",
            params![sequence, post_id, session, model_id, timestamp],
        )?;
    }
    Ok(json!({"revision":revision,"changed":true}))
}

fn existing_view(
    connection: &Connection,
    event_id: &str,
    session: &str,
    post: i64,
) -> Result<bool> {
    let existing: Option<(String, i64)> = connection
        .query_row(
            "SELECT session_id,post_id FROM presentation_facts WHERE event_id=?1 AND is_view=1",
            [event_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    if let Some((saved_session, saved_post)) = existing {
        ensure!(
            saved_session == session && saved_post == post,
            "View event ID belongs to another offer"
        );
        return Ok(true);
    }
    Ok(false)
}

fn record_view(
    connection: &Connection,
    event_id: &str,
    sequence: i64,
    post: i64,
    session: &str,
    timestamp: f64,
) -> Result<()> {
    let context = context(connection, session, post)?;
    if existing_view(connection, event_id, session, post)? {
        return Ok(());
    }
    let family = if context.is_auto { "auto" } else { "gallery" };
    connection.execute(
        "INSERT INTO presentation_facts(sequence,ordinal,kind,event_id,session_id,post_id,viewed_at,new_entry,is_view,offer_family,offer_position,view_family) SELECT ?1,0,'view',?2,?3,?4,?5,0,1,?6,position,?6 FROM offer_facts WHERE family=?6 AND session_id=?3 AND post_id=?4",
        params![sequence, event_id, session, post, timestamp, family],
    )?;
    Ok(())
}

fn present(
    connection: &Connection,
    event_id: &str,
    sequence: i64,
    context_id: &str,
    entry_id: Option<&str>,
    card: &Card,
    timestamp: f64,
) -> Result<()> {
    let active: String = connection
        .query_row(
            "SELECT active_context_id FROM q_browse_state WHERE singleton=1",
            [],
            |row| row.get(0),
        )
        .context("Presentation requires an active context")?;
    ensure!(
        active == context_id,
        "Presentation belongs to an inactive context"
    );
    let (mode, filter): (String, String) = connection.query_row(
        "SELECT mode,filter_text FROM q_browse_contexts WHERE context_id=?1",
        [context_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    // Validation is Gallery-first even though actual-view provenance is Auto-first.
    let offer: Option<(String, String, Option<String>, String, i64)> = connection.query_row(
        "SELECT s.facet,s.filter_text,o.card_json,o.family,o.position FROM offer_facts o JOIN q_gallery_sessions s USING(session_id) WHERE o.family='gallery' AND o.session_id=?1 AND o.post_id=?2 UNION ALL SELECT s.preset,s.filter_text,o.card_json,o.family,o.position FROM offer_facts o JOIN q_auto_sessions s USING(session_id) WHERE o.family='auto' AND o.session_id=?1 AND o.post_id=?2",
        params![card.session_id, card.post_id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
    ).optional()?;
    let (offer_mode, offer_filter, original, family, offer_position) =
        offer.context("Presentation card was never offered")?;
    ensure!(
        offer_mode == mode && offer_filter == filter,
        "Presentation offer belongs to another mode/filter"
    );
    let mut comparable: Card =
        serde_json::from_str(&original.context("Presentation requires a native offer")?)?;
    comparable.current_grade = None;
    ensure!(
        serde_json::to_value(&comparable)? == serde_json::to_value(card)?,
        "Presentation card differs from its immutable offer"
    );
    let entry = entry_id.unwrap_or(event_id);
    let position = if entry_id.is_some() {
        let (saved_context, saved_card): (String, String) = connection
            .query_row(
                "SELECT context_id,card_json FROM q_browse_entries WHERE entry_id=?1",
                [entry],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?
            .context("Unknown visited history entry")?;
        ensure!(
            saved_context == context_id,
            "History entry belongs to another context"
        );
        let mut saved_card: Card = serde_json::from_str(&saved_card)?;
        saved_card.current_grade = None;
        ensure!(
            serde_json::to_value(&saved_card)? == serde_json::to_value(&comparable)?,
            "History entry refers to another offer"
        );
        None
    } else {
        Some(connection.query_row(
            "SELECT COALESCE(MAX(position),-1)+1 FROM q_browse_entries WHERE context_id=?1",
            [context_id],
            |row| row.get::<_, i64>(0),
        )?)
    };
    let view = context(connection, &card.session_id, card.post_id)?;
    ensure!(
        !existing_view(connection, event_id, &card.session_id, card.post_id)?,
        "Presentation event ID is already an actual view"
    );
    connection.execute(
        "INSERT INTO presentation_facts(sequence,ordinal,kind,event_id,session_id,post_id,viewed_at,context_id,entry_id,new_entry,entry_position,offer_family,offer_position,view_family,is_view) VALUES(?1,0,'presented',?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,1)",
        params![sequence, event_id, card.session_id, card.post_id, timestamp, context_id, entry, entry_id.is_none(), position, family, offer_position, if view.is_auto { "auto" } else { "gallery" }],
    )?;
    Ok(())
}

fn retire(
    connection: &Connection,
    sequence: i64,
    ordinal: i64,
    owner: &GalleryOwner<'_>,
    completed: bool,
    timestamp: f64,
) -> Result<()> {
    validate_gallery_owner(connection, owner, true)?;
    let status: String = connection.query_row(
        "SELECT status FROM q_gallery_sessions WHERE session_id=?1",
        [owner.session],
        |row| row.get(0),
    )?;
    connection.execute(
        "INSERT INTO selection_facts(sequence,ordinal,family,session_id,operation,updated_at,status) VALUES(?1,?2,'gallery',?3,'retire',?4,?5)",
        params![sequence, ordinal, owner.session, timestamp, if completed || status == "completed" { "completed" } else { "abandoned" }],
    )?;
    Ok(())
}

pub(super) fn unavailable(connection: &Connection) -> Result<HashSet<i64>> {
    post_ids(
        connection,
        "SELECT post_id FROM rating_facts UNION SELECT post_id FROM presentation_facts WHERE is_view=1",
    )
}

fn post_ids(connection: &Connection, sql: &str) -> Result<HashSet<i64>> {
    let mut statement = connection.prepare(sql)?;
    Ok(statement
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?)
}

pub(super) fn recent_views(connection: &Connection, limit: usize) -> Result<Vec<i64>> {
    ensure!(limit > 0, "Recent view limit must be positive");
    i64::try_from(limit).context("Recent view limit exceeds SQLite integer range")?;
    let mut statement = connection.prepare(
        "SELECT post_id FROM presentation_facts WHERE is_view=1 ORDER BY viewed_at DESC,post_id",
    )?;
    let mut rows = statement.query([])?;
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    while let Some(row) = rows.next()? {
        let post = row.get(0)?;
        if seen.insert(post) {
            ordered.push(post);
            if ordered.len() == limit {
                break;
            }
        }
    }
    Ok(ordered)
}

pub(super) fn checkpoints(connection: &Connection) -> Result<HashMap<i64, Checkpoint>> {
    let mut statement = connection.prepare("SELECT post_id,predicted_grade,recorded_action,rating_revision,model_id,review_sequence,updated_at FROM q_rating_prediction_checkpoints")?;
    let mut rows = statement.query([])?;
    let mut checkpoints = HashMap::new();
    while let Some(row) = rows.next()? {
        let post_id = row.get(0)?;
        let predicted_grade: u8 = row.get(1)?;
        Grade::from_u8(predicted_grade)
            .with_context(|| format!("Invalid checkpoint for post {post_id}"))?;
        checkpoints.insert(
            post_id,
            Checkpoint {
                post_id,
                predicted_grade,
                recorded_grade: Grade::parse(&row.get::<_, String>(2)?)?,
                rating_revision: row.get(3)?,
                model_id: row.get(4)?,
                review_sequence: row.get(5)?,
                updated_at: row.get(6)?,
            },
        );
    }
    Ok(checkpoints)
}

pub(super) fn review_exclusions(connection: &Connection) -> Result<HashSet<i64>> {
    post_ids(
        connection,
        "SELECT DISTINCT post_id FROM review_facts WHERE kind IN ('exclusion_seed','exclusion')",
    )
}

pub(super) fn review_sequence(connection: &Connection) -> Result<i64> {
    let sequence: i64 = connection.query_row(
        "SELECT review_sequence FROM q_rating_review_state WHERE singleton=1",
        [],
        |row| row.get(0),
    )?;
    ensure!(sequence >= 0, "Invalid negative Review sequence");
    Ok(sequence)
}

pub(super) fn preferences(connection: &Connection) -> Result<Option<UserSettings>> {
    let json: Option<String> = connection
        .query_row(
            "SELECT settings_json FROM q_user_preferences WHERE singleton=1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    json.map(|json| serde_json::from_str(&json).map_err(Into::into))
        .transpose()
}

pub(super) fn active_context(connection: &Connection) -> Result<Option<BrowseContext>> {
    let id: Option<String> = connection
        .query_row(
            "SELECT active_context_id FROM q_browse_state WHERE singleton=1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    id.map(|id| read_context(connection, &id)).transpose()
}

pub(super) fn imported_view_reserved(connection: &Connection, event_id: &str) -> Result<bool> {
    Ok(connection.query_row("SELECT EXISTS(SELECT 1 FROM presentation_facts WHERE kind='view_seed' AND is_view=1 AND event_id=?1)", [event_id], |row| row.get(0))?)
}

pub(super) fn ratings(connection: &Connection) -> Result<Vec<StoredRating>> {
    let mut statement = connection.prepare(
        "SELECT post_id,action,first_rated_at,updated_at,revision,context_json FROM q_ratings ORDER BY first_rated_at,post_id",
    )?;
    let mut rows = statement.query([])?;
    let mut ratings = Vec::new();
    while let Some(row) = rows.next()? {
        let post_id = row.get(0)?;
        ratings.push(StoredRating {
            post_id,
            grade: Grade::parse(row.get_ref(1)?.as_str()?)
                .with_context(|| format!("Decoding stored grade for post {post_id}"))?,
            first_rated_at: row.get(2)?,
            updated_at: row.get(3)?,
            revision: row.get(4)?,
            context_json: row.get(5)?,
        });
    }
    Ok(ratings)
}

pub(super) fn read_context(connection: &Connection, context_id: &str) -> Result<BrowseContext> {
    let (mode, filter, cursor) = connection.query_row(
        "SELECT mode,filter_text,cursor FROM q_browse_contexts WHERE context_id=?1",
        [context_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    let mut statement = connection.prepare(
        "SELECT entry_id,card_json FROM q_browse_entries WHERE context_id=?1 ORDER BY position",
    )?;
    let mut rows = statement.query([context_id])?;
    let mut entries = Vec::new();
    while let Some(row) = rows.next()? {
        entries.push(BrowseEntry {
            entry_id: row.get(0)?,
            card: serde_json::from_str(row.get_ref(1)?.as_str()?)?,
        });
    }
    Ok(BrowseContext {
        context_id: context_id.into(),
        mode,
        filter,
        entries,
        cursor,
    })
}

pub(super) fn context(
    connection: &Connection,
    session: &str,
    post_id: i64,
) -> Result<RatingContext> {
    required_text(session, "Rating session ID")?;
    ensure!(post_id > 0, "Post ID must be positive");
    let auto: Option<(String, String, i32, String)> = connection
        .query_row(
            "SELECT model_id,filter_text,seed,preset FROM q_auto_sessions WHERE session_id=?1",
            [session],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()?;
    if let Some((model, filter, seed, preset)) = auto {
        let (position, payload): (i64, String) = connection.query_row(
            "SELECT position,payload_json FROM offer_facts WHERE family='auto' AND session_id=?1 AND post_id=?2",
            params![session, post_id], |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional()?.with_context(|| format!("Post {post_id} was not offered in Auto session '{session}'"))?;
        let identity = payload_identity(&payload)?;
        let mut context = json!({
            "source": "auto-rust-v1", "media_md5": identity.md5, "media_file_extension": identity.extension,
            "auto_session_id": session, "preset": preset, "filter": filter, "seed": seed,
            "model_id": identity.model_id.as_deref().unwrap_or(&model), "rank": position,
        });
        if let Some(policy) = identity.selection_policy {
            context["selection_policy"] = Value::String(policy);
        }
        Ok(RatingContext {
            json: serde_json::to_string(&context)?,
            is_auto: true,
            facet: None,
        })
    } else {
        let (model, facet, filter): (String, String, String) = connection
            .query_row(
                "SELECT model_id,facet,filter_text FROM q_gallery_sessions WHERE session_id=?1",
                [session],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?
            .with_context(|| format!("Unknown rating session '{session}'"))?;
        let (position, payload): (i64, String) = connection.query_row(
            "SELECT position,payload_json FROM offer_facts WHERE family='gallery' AND session_id=?1 AND post_id=?2",
            params![session, post_id], |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional()?.with_context(|| format!("Post {post_id} was not offered in gallery session '{session}'"))?;
        let identity = payload_identity(&payload)?;
        let mut context = json!({
            "source": "gallery-rust-v1", "media_md5": identity.md5, "media_file_extension": identity.extension,
            "gallery_session_id": session, "facet": facet, "filter": filter,
            "model_id": identity.model_id.as_deref().unwrap_or(&model), "rank": position,
        });
        if let Some(policy) = identity.selection_policy {
            context["selection_policy"] = Value::String(policy);
        }
        Ok(RatingContext {
            json: serde_json::to_string(&context)?,
            is_auto: false,
            facet: Some(facet),
        })
    }
}

fn ensure_new_session(connection: &Connection, session: &str) -> Result<()> {
    let exists: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM selection_facts WHERE session_id=?1)",
        [session],
        |row| row.get(0),
    )?;
    ensure!(!exists, "Session ID '{session}' already exists");
    Ok(())
}

fn validate_gallery_owner(
    connection: &Connection,
    owner: &GalleryOwner<'_>,
    allow_completed: bool,
) -> Result<()> {
    let saved: Option<(String, String, String, i64, i64)> = connection.query_row(
        "SELECT status,facet,filter_text,requested_size,returned_size FROM q_gallery_sessions WHERE session_id=?1",
        [owner.session],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
    ).optional()?;
    let (status, facet, filter, capacity, returned) = saved.context("Unknown Gallery cohort")?;
    ensure!(
        status == "active" || allow_completed && status == "completed",
        "Gallery cohort is not appendable"
    );
    ensure!(
        facet == owner.facet && filter == owner.filter,
        "Gallery cohort ownership disagrees with facet/filter"
    );
    ensure!(
        capacity == i64::try_from(owner.capacity)?,
        "Gallery cohort capacity changed"
    );
    ensure!(
        returned == i64::try_from(owner.issued)?,
        "Gallery append position is stale"
    );
    let (count, minimum, maximum): (i64, i64, i64) = connection.query_row(
        "SELECT COUNT(*),COALESCE(MIN(position),0),COALESCE(MAX(position),-1) FROM offer_facts WHERE family='gallery' AND session_id=?1",
        [owner.session], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    ensure!(
        count == returned && minimum == 0 && maximum == returned - 1,
        "Gallery persisted positions are not contiguous"
    );
    Ok(())
}

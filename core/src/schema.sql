-- Schema 13: canonical immutable journal/details; only runtime q_* reads are installed.
CREATE TABLE interaction_events (
 sequence INTEGER PRIMARY KEY,
 event_id TEXT NOT NULL UNIQUE CHECK(length(trim(event_id)) > 0),
 occurred_at REAL NOT NULL, request_json TEXT NOT NULL,
 payload_json TEXT NOT NULL, result_json TEXT NOT NULL
);
CREATE TABLE selection_facts (
 sequence INTEGER NOT NULL REFERENCES interaction_events(sequence) DEFERRABLE INITIALLY DEFERRED, ordinal INTEGER NOT NULL,
 family TEXT NOT NULL CHECK(family IN ('gallery','auto')), session_id TEXT NOT NULL,
 operation TEXT NOT NULL CHECK(operation IN ('seed','create','append','retire')),
 created_at REAL, updated_at REAL NOT NULL, status TEXT, mode TEXT, filter_text TEXT, model_id TEXT,
 requested_size INTEGER, returned_size INTEGER, cursor INTEGER, exhausted INTEGER, seed INTEGER,
 PRIMARY KEY(sequence,ordinal)
) WITHOUT ROWID;
CREATE INDEX selection_identity ON selection_facts(family,session_id,sequence DESC,ordinal DESC);
CREATE TABLE offer_facts (
 sequence INTEGER NOT NULL REFERENCES interaction_events(sequence) DEFERRABLE INITIALLY DEFERRED, ordinal INTEGER NOT NULL,
 family TEXT NOT NULL CHECK(family IN ('gallery','auto')), session_id TEXT NOT NULL,
 position INTEGER NOT NULL, post_id INTEGER NOT NULL, facet TEXT,
 payload_json TEXT NOT NULL, card_json TEXT, selection_policy TEXT,
 initial_state TEXT, initial_delivered_at REAL,
 PRIMARY KEY(family,session_id,position), UNIQUE(family,session_id,post_id), UNIQUE(sequence,ordinal)
) WITHOUT ROWID;
CREATE INDEX offer_post ON offer_facts(post_id);
CREATE TABLE rating_facts (
 sequence INTEGER NOT NULL REFERENCES interaction_events(sequence) DEFERRABLE INITIALLY DEFERRED, ordinal INTEGER NOT NULL,
 origin TEXT NOT NULL CHECK(origin IN ('seed','change')), post_id INTEGER NOT NULL,
 action TEXT NOT NULL, first_rated_at REAL NOT NULL, updated_at REAL NOT NULL, revision INTEGER NOT NULL,
 context_json TEXT NOT NULL, family TEXT, session_id TEXT,
 predicted_grade INTEGER, recorded_action TEXT, rating_revision INTEGER,
 judgment_model_id TEXT, checkpoint_review_sequence INTEGER, checkpoint_updated_at REAL,
 PRIMARY KEY(sequence,ordinal)
) WITHOUT ROWID;
CREATE UNIQUE INDEX rating_latest ON rating_facts(post_id,sequence DESC);
CREATE INDEX rating_family ON rating_facts(family,post_id,sequence);
CREATE TABLE review_facts (
 sequence INTEGER NOT NULL REFERENCES interaction_events(sequence) DEFERRABLE INITIALLY DEFERRED, ordinal INTEGER NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('counter_seed','vote_seed','vote','exclusion_seed','exclusion')),
 post_id INTEGER, session_id TEXT, review_sequence INTEGER, model_id TEXT, excluded_at REAL,
 PRIMARY KEY(sequence,ordinal)
) WITHOUT ROWID;
CREATE UNIQUE INDEX review_vote_identity ON review_facts(session_id,post_id) WHERE kind IN ('vote_seed','vote');
CREATE UNIQUE INDEX review_vote_sequence ON review_facts(review_sequence) WHERE kind IN ('vote_seed','vote');
CREATE INDEX review_exclusion_latest ON review_facts(post_id,sequence DESC,ordinal DESC) WHERE kind IN ('exclusion_seed','exclusion');
CREATE TABLE presentation_facts (
 sequence INTEGER NOT NULL REFERENCES interaction_events(sequence) DEFERRABLE INITIALLY DEFERRED, ordinal INTEGER NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('view_seed','view','entry_seed','presented')),
 event_id TEXT, session_id TEXT, post_id INTEGER, viewed_at REAL,
 context_id TEXT, entry_id TEXT, new_entry INTEGER NOT NULL CHECK(new_entry IN (0,1)),
 entry_position INTEGER, offer_family TEXT, offer_position INTEGER, view_family TEXT, seed_card_json TEXT,
 is_view INTEGER NOT NULL CHECK(is_view IN (0,1)), PRIMARY KEY(sequence,ordinal)
) WITHOUT ROWID;
CREATE UNIQUE INDEX view_identity ON presentation_facts(event_id) WHERE is_view=1;
CREATE UNIQUE INDEX history_identity ON presentation_facts(entry_id) WHERE new_entry=1;
CREATE INDEX recent_by_time ON presentation_facts(viewed_at DESC,post_id) WHERE is_view=1;
CREATE INDEX context_presentation ON presentation_facts(context_id,sequence DESC,ordinal DESC);
CREATE TABLE activation_facts (
 sequence INTEGER NOT NULL REFERENCES interaction_events(sequence) DEFERRABLE INITIALLY DEFERRED, ordinal INTEGER NOT NULL,
 origin TEXT NOT NULL CHECK(origin IN ('seed','activation')), context_id TEXT NOT NULL,
 mode TEXT NOT NULL, filter_text TEXT NOT NULL, activated_sequence INTEGER NOT NULL,
 seed_cursor TEXT, seed_active INTEGER NOT NULL DEFAULT 0 CHECK(seed_active IN (0,1)),
 PRIMARY KEY(sequence,ordinal)
) WITHOUT ROWID;
CREATE INDEX context_activation ON activation_facts(context_id,sequence DESC,ordinal DESC);
CREATE INDEX context_specification ON activation_facts(mode,filter_text,activated_sequence DESC,context_id);
CREATE TABLE preference_facts (
 sequence INTEGER NOT NULL REFERENCES interaction_events(sequence) DEFERRABLE INITIALLY DEFERRED, ordinal INTEGER NOT NULL,
 settings_json TEXT NOT NULL, PRIMARY KEY(sequence,ordinal)
) WITHOUT ROWID;
CREATE TABLE legacy_delivery_facts (
 sequence INTEGER NOT NULL REFERENCES interaction_events(sequence) DEFERRABLE INITIALLY DEFERRED, ordinal INTEGER NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('gallery_delivery','auto_play')),
 post_id INTEGER NOT NULL, session_id TEXT NOT NULL, occurred_at REAL NOT NULL,
 PRIMARY KEY(sequence,ordinal), UNIQUE(kind,post_id)
) WITHOUT ROWID;
CREATE UNIQUE INDEX selection_initial_identity ON selection_facts(family,session_id) WHERE operation IN ('seed','create');
CREATE INDEX actual_view_post ON presentation_facts(post_id) WHERE is_view=1;
CREATE UNIQUE INDEX history_position ON presentation_facts(context_id,entry_position) WHERE new_entry=1;
CREATE UNIQUE INDEX activation_seed_identity ON activation_facts(context_id) WHERE origin='seed';
CREATE TRIGGER interaction_events_no_update BEFORE UPDATE ON interaction_events
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER interaction_events_no_delete BEFORE DELETE ON interaction_events
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER interaction_events_no_replace BEFORE INSERT ON interaction_events
WHEN EXISTS(SELECT 1 FROM interaction_events WHERE sequence=NEW.sequence OR event_id=NEW.event_id)
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER selection_facts_no_update BEFORE UPDATE ON selection_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER selection_facts_no_delete BEFORE DELETE ON selection_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER selection_facts_no_replace BEFORE INSERT ON selection_facts
WHEN EXISTS(SELECT 1 FROM selection_facts WHERE (sequence=NEW.sequence AND ordinal=NEW.ordinal) OR (family=NEW.family AND session_id=NEW.session_id AND operation IN ('seed','create') AND NEW.operation IN ('seed','create')))
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER offer_facts_no_update BEFORE UPDATE ON offer_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER offer_facts_no_delete BEFORE DELETE ON offer_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER offer_facts_no_replace BEFORE INSERT ON offer_facts
WHEN EXISTS(SELECT 1 FROM offer_facts WHERE (sequence=NEW.sequence AND ordinal=NEW.ordinal) OR (family=NEW.family AND session_id=NEW.session_id AND (position=NEW.position OR post_id=NEW.post_id)))
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER rating_facts_no_update BEFORE UPDATE ON rating_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER rating_facts_no_delete BEFORE DELETE ON rating_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER rating_facts_no_replace BEFORE INSERT ON rating_facts
WHEN EXISTS(SELECT 1 FROM rating_facts WHERE (sequence=NEW.sequence AND (ordinal=NEW.ordinal OR post_id=NEW.post_id)))
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER review_facts_no_update BEFORE UPDATE ON review_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER review_facts_no_delete BEFORE DELETE ON review_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER review_facts_no_replace BEFORE INSERT ON review_facts
WHEN EXISTS(SELECT 1 FROM review_facts WHERE (sequence=NEW.sequence AND ordinal=NEW.ordinal) OR (kind IN ('vote_seed','vote') AND NEW.kind IN ('vote_seed','vote') AND ((session_id=NEW.session_id AND post_id=NEW.post_id) OR review_sequence=NEW.review_sequence)))
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER presentation_facts_no_update BEFORE UPDATE ON presentation_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER presentation_facts_no_delete BEFORE DELETE ON presentation_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER presentation_facts_no_replace BEFORE INSERT ON presentation_facts
WHEN EXISTS(SELECT 1 FROM presentation_facts WHERE (sequence=NEW.sequence AND ordinal=NEW.ordinal) OR (is_view=1 AND NEW.is_view=1 AND event_id=NEW.event_id) OR (new_entry=1 AND NEW.new_entry=1 AND (entry_id=NEW.entry_id OR (context_id=NEW.context_id AND entry_position=NEW.entry_position))))
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER activation_facts_no_update BEFORE UPDATE ON activation_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER activation_facts_no_delete BEFORE DELETE ON activation_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER activation_facts_no_replace BEFORE INSERT ON activation_facts
WHEN EXISTS(SELECT 1 FROM activation_facts WHERE (sequence=NEW.sequence AND ordinal=NEW.ordinal) OR (origin='seed' AND NEW.origin='seed' AND context_id=NEW.context_id))
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER preference_facts_no_update BEFORE UPDATE ON preference_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER preference_facts_no_delete BEFORE DELETE ON preference_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER preference_facts_no_replace BEFORE INSERT ON preference_facts
WHEN EXISTS(SELECT 1 FROM preference_facts WHERE (sequence=NEW.sequence AND ordinal=NEW.ordinal))
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER legacy_delivery_facts_no_update BEFORE UPDATE ON legacy_delivery_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER legacy_delivery_facts_no_delete BEFORE DELETE ON legacy_delivery_facts
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;
CREATE TRIGGER legacy_delivery_facts_no_replace BEFORE INSERT ON legacy_delivery_facts
WHEN EXISTS(SELECT 1 FROM legacy_delivery_facts WHERE (sequence=NEW.sequence AND ordinal=NEW.ordinal) OR (kind=NEW.kind AND post_id=NEW.post_id))
BEGIN SELECT RAISE(ABORT,'immutable interaction fact'); END;

-- On-demand runtime readouts for the current schema.
CREATE VIEW IF NOT EXISTS q_ratings AS 
            SELECT current.post_id, current.action, current.first_rated_at,
                   current.updated_at, current.revision, current.context_json
            FROM rating_facts AS current
            JOIN (
                SELECT post_id, MAX(sequence) AS sequence
                FROM rating_facts GROUP BY post_id
            ) AS latest USING (post_id, sequence)
        ;

CREATE VIEW IF NOT EXISTS q_gallery_sessions AS
    SELECT initial.session_id,
           (SELECT change.status FROM selection_facts AS change
            WHERE change.family = initial.family AND change.session_id = initial.session_id
              AND change.status IS NOT NULL
            ORDER BY change.sequence DESC, change.ordinal DESC LIMIT 1) AS status,
           initial.mode AS facet, initial.filter_text, initial.model_id,
           initial.requested_size,
           (SELECT change.returned_size FROM selection_facts AS change
            WHERE change.family = initial.family AND change.session_id = initial.session_id
              AND change.returned_size IS NOT NULL
            ORDER BY change.sequence DESC, change.ordinal DESC LIMIT 1) AS returned_size
    FROM selection_facts AS initial
    WHERE initial.family = 'gallery' AND initial.operation IN ('seed', 'create');

CREATE VIEW IF NOT EXISTS q_auto_sessions AS
    SELECT initial.session_id,
           (SELECT change.status FROM selection_facts AS change
            WHERE change.family = initial.family AND change.session_id = initial.session_id
              AND change.status IS NOT NULL
            ORDER BY change.sequence DESC, change.ordinal DESC LIMIT 1) AS status,
           initial.model_id, initial.mode AS preset, initial.filter_text, initial.seed,
           (SELECT change.returned_size FROM selection_facts AS change
            WHERE change.family = initial.family AND change.session_id = initial.session_id
              AND change.returned_size IS NOT NULL
            ORDER BY change.sequence DESC, change.ordinal DESC LIMIT 1) AS returned_size
    FROM selection_facts AS initial
    WHERE initial.family = 'auto' AND initial.operation IN ('seed', 'create');


CREATE VIEW IF NOT EXISTS q_rating_prediction_checkpoints AS 
            SELECT current.post_id, current.predicted_grade,
                   current.recorded_action, current.rating_revision,
                   current.judgment_model_id AS model_id,
                   current.checkpoint_review_sequence AS review_sequence,
                   current.checkpoint_updated_at AS updated_at
            FROM rating_facts AS current
            WHERE current.predicted_grade IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM rating_facts AS newer
                  WHERE newer.post_id = current.post_id
                    AND (newer.sequence, newer.ordinal) >
                        (current.sequence, current.ordinal)
              )
        ;


CREATE VIEW IF NOT EXISTS q_rating_review_state AS 
            SELECT 1 AS singleton, COALESCE((
                SELECT review_sequence
                FROM review_facts
                WHERE kind = 'vote'
                ORDER BY sequence DESC, ordinal DESC
                LIMIT 1
            ), (
                SELECT review_sequence
                FROM review_facts
                WHERE kind = 'counter_seed'
                ORDER BY sequence DESC, ordinal DESC
                LIMIT 1
            ), 0) AS review_sequence
        ;


CREATE VIEW IF NOT EXISTS q_user_preferences AS 
            SELECT 1 AS singleton, settings_json
            FROM preference_facts
            ORDER BY sequence DESC, ordinal DESC
            LIMIT 1
        ;

CREATE VIEW IF NOT EXISTS q_browse_contexts AS 
            SELECT current.context_id, current.mode, current.filter_text,
                   current.activated_sequence,
                   COALESCE((
                       SELECT occurrence.entry_id
                       FROM presentation_facts AS occurrence
                       WHERE occurrence.context_id = current.context_id
                         AND occurrence.kind = 'presented'
                       ORDER BY occurrence.sequence DESC, occurrence.ordinal DESC
                       LIMIT 1
                   ), (
                       SELECT seed.seed_cursor
                       FROM activation_facts AS seed
                       WHERE seed.context_id = current.context_id
                         AND seed.origin = 'seed'
                       ORDER BY seed.sequence DESC, seed.ordinal DESC
                       LIMIT 1
                   )) AS cursor
            FROM activation_facts AS current
            WHERE NOT EXISTS (
                SELECT 1 FROM activation_facts AS newer
                WHERE newer.context_id = current.context_id
                  AND (newer.sequence, newer.ordinal) >
                      (current.sequence, current.ordinal)
            )
        ;

CREATE VIEW IF NOT EXISTS q_browse_entries AS 
            SELECT entry.entry_id, entry.context_id,
                   entry.entry_position AS position,
                   COALESCE(entry.seed_card_json, offer.card_json) AS card_json
            FROM presentation_facts AS entry
            LEFT JOIN offer_facts AS offer
              ON offer.family = entry.offer_family
             AND offer.session_id = entry.session_id
             AND offer.position = entry.offer_position
            WHERE entry.new_entry = 1
        ;

CREATE VIEW IF NOT EXISTS q_browse_state AS 
            SELECT 1 AS singleton, active_context_id
            FROM (
                SELECT COALESCE((
                    SELECT context_id
                    FROM activation_facts
                    WHERE origin = 'activation'
                    ORDER BY sequence DESC, ordinal DESC
                    LIMIT 1
                ), (
                    SELECT context_id
                    FROM activation_facts
                    WHERE origin = 'seed' AND seed_active = 1
                    ORDER BY sequence DESC, ordinal DESC
                    LIMIT 1
                )) AS active_context_id
            )
            WHERE active_context_id IS NOT NULL
        ;

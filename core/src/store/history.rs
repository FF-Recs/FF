//! Current-schema admission and read-only verification of immutable history.
use anyhow::{Context, Result, bail, ensure};
use rusqlite::{Connection, TransactionBehavior, params};
use serde_json::Value;

use super::{Event, SCHEMA, SCHEMA_VERSION, decode_event, facts};

struct FactTable {
    name: &'static str,
    ordered_rows: &'static str,
}

// Canonical DDL fixes each table's complete column order. Compare every stored
// field, including the journal's six envelope fields, without a
// second column-name authority or materializing either table in memory.
const FACT_TABLES: [FactTable; 9] = [
    FactTable {
        name: "interaction_events",
        ordered_rows: "SELECT * FROM main.interaction_events ORDER BY sequence",
    },
    FactTable {
        name: "selection_facts",
        ordered_rows: "SELECT * FROM main.selection_facts ORDER BY sequence,ordinal",
    },
    FactTable {
        name: "offer_facts",
        ordered_rows: "SELECT * FROM main.offer_facts ORDER BY sequence,ordinal",
    },
    FactTable {
        name: "rating_facts",
        ordered_rows: "SELECT * FROM main.rating_facts ORDER BY sequence,ordinal",
    },
    FactTable {
        name: "review_facts",
        ordered_rows: "SELECT * FROM main.review_facts ORDER BY sequence,ordinal",
    },
    FactTable {
        name: "presentation_facts",
        ordered_rows: "SELECT * FROM main.presentation_facts ORDER BY sequence,ordinal",
    },
    FactTable {
        name: "activation_facts",
        ordered_rows: "SELECT * FROM main.activation_facts ORDER BY sequence,ordinal",
    },
    FactTable {
        name: "preference_facts",
        ordered_rows: "SELECT * FROM main.preference_facts ORDER BY sequence,ordinal",
    },
    FactTable {
        name: "legacy_delivery_facts",
        ordered_rows: "SELECT * FROM main.legacy_delivery_facts ORDER BY sequence,ordinal",
    },
];

/// Admit only the current layout, or initialize a genuinely empty database.
/// Rejections precede persistent writes and the caller's persistent pragmas.
pub(super) fn open_schema(connection: &mut Connection) -> Result<()> {
    ensure!(
        connection.is_autocommit(),
        "Schema opening requires no active transaction"
    );
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
    let version: i64 =
        transaction.pragma_query_value(Some("main"), "user_version", |row| row.get(0))?;
    ensure!(
        matches!(version, 0 | SCHEMA_VERSION),
        "Unsupported interaction schema {version}; expected {SCHEMA_VERSION}"
    );
    if version == 0 {
        let occupied: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM main.sqlite_schema WHERE name NOT GLOB 'sqlite_*')",
            [],
            |row| row.get(0),
        )?;
        ensure!(
            !occupied,
            "Unsupported nonempty unversioned interaction database"
        );
        transaction.execute_batch(SCHEMA)?;
        append_initial(&transaction)?;
        transaction.pragma_update(Some("main"), "user_version", SCHEMA_VERSION)?;
    }
    let canonical = Connection::open_in_memory()?;
    canonical.execute_batch(SCHEMA)?;
    validate_current(&transaction, &canonical)?;
    transaction.commit()?;
    Ok(())
}

/// Verify a stable source snapshot without changing its data, schemas or pragmas.
/// A caller-owned transaction remains owned by the caller, including on failure.
pub(super) fn verify_history(connection: &Connection) -> Result<()> {
    let owned_read = if connection.is_autocommit() {
        Some(connection.unchecked_transaction()?)
    } else {
        None
    };
    let replay = Connection::open_in_memory()?;
    replay.pragma_update(None, "foreign_keys", true)?;
    replay.execute_batch(SCHEMA)?;
    validate_current(connection, &replay)?;
    let transaction = replay.unchecked_transaction()?;
    replay_journal(connection, &transaction)?;
    foreign_key_check(&transaction)?;
    for table in &FACT_TABLES {
        compare_rows(connection, &transaction, table)?;
    }
    // Only the isolated replay was written. Release only our own source read.
    transaction.commit()?;
    if let Some(read) = owned_read {
        read.commit()?;
    }
    Ok(())
}

fn append_initial(connection: &Connection) -> Result<()> {
    let event = Event::Initialized {
        facts: Default::default(),
    };
    let event_id = uuid::Uuid::new_v4().to_string();
    let timestamp = super::now()?;
    let payload = serde_json::to_string(&event)?;
    let request = serde_json::to_string(&event.request(&payload)?)?;
    let result = facts::apply(connection, &event_id, 1, timestamp, &event)?;
    ensure!(
        result.is_null(),
        "Initial interaction event returned a non-null result"
    );
    connection.execute(
        "INSERT INTO main.interaction_events(sequence,event_id,occurred_at,request_json,payload_json,result_json) VALUES(1,?1,?2,?3,?4,'null')",
        params![event_id, timestamp, request, payload],
    )?;
    Ok(())
}

fn validate_current(connection: &Connection, canonical: &Connection) -> Result<()> {
    let version: i64 =
        connection.pragma_query_value(Some("main"), "user_version", |row| row.get(0))?;
    ensure!(
        version == SCHEMA_VERSION,
        "Unsupported interaction schema {version}; expected {SCHEMA_VERSION}"
    );
    let mut statement = canonical.prepare(
        "SELECT type,name,sql FROM main.sqlite_schema WHERE type IN ('table','index','trigger','view') AND sql IS NOT NULL ORDER BY type,name",
    )?;
    let mut actual_statement =
        connection.prepare("SELECT sql FROM main.sqlite_schema WHERE type=?1 AND name=?2")?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let kind = row.get_ref(0)?.as_str()?;
        let name = row.get_ref(1)?.as_str()?;
        let expected = row.get_ref(2)?.as_str()?;
        let mut actual_rows = actual_statement.query(params![kind, name])?;
        let matches = match actual_rows.next()? {
            Some(actual) => actual.get_ref(0)?.as_str()?.trim() == expected.trim(),
            None => false,
        };
        ensure!(
            matches,
            "Canonical interaction {kind} is missing or changed: {name}"
        );
    }
    validate_headers(connection)?;
    foreign_key_check(connection)
}

fn foreign_key_check(connection: &Connection) -> Result<()> {
    let mut statement = connection.prepare("PRAGMA main.foreign_key_check")?;
    let mut rows = statement.query([])?;
    if let Some(row) = rows.next()? {
        let table: String = row.get(0)?;
        let rowid: Option<i64> = row.get(1)?;
        let parent: String = row.get(2)?;
        bail!("Interaction foreign-key violation in {table} at {rowid:?}, referencing {parent}");
    }
    Ok(())
}

fn validate_headers(connection: &Connection) -> Result<()> {
    let mut statement = connection
        .prepare("SELECT sequence,payload_json FROM main.interaction_events ORDER BY sequence")?;
    let mut rows = statement.query([])?;
    let mut first = true;
    while let Some(row) = rows.next()? {
        let sequence: i64 = row.get(0)?;
        let payload = row.get_ref(1)?.as_str()?;
        ensure!(
            sequence > 0,
            "Invalid interaction event sequence {sequence}"
        );
        let event = decode_event(payload)
            .with_context(|| format!("Decoding interaction event {sequence}"))?;
        let initial = matches!(event, Event::Initialized { .. });
        if first {
            ensure!(
                sequence == 1 && initial,
                "Interaction journal lacks initial event at sequence 1"
            );
            first = false;
        } else {
            ensure!(
                !initial,
                "Interaction journal contains multiple initial events"
            );
        }
    }
    ensure!(!first, "Interaction journal lacks initial event");
    Ok(())
}

/// Apply each supported event before inserting its untouched original envelope.
fn replay_journal(source: &Connection, target: &Connection) -> Result<()> {
    let mut statement = source.prepare(
        "SELECT sequence,event_id,occurred_at,request_json,payload_json,result_json FROM main.interaction_events ORDER BY sequence",
    )?;
    let mut rows = statement.query([])?;
    let mut insert = target.prepare(
        "INSERT INTO main.interaction_events(sequence,event_id,occurred_at,request_json,payload_json,result_json) VALUES(?1,?2,?3,?4,?5,?6)",
    )?;
    while let Some(row) = rows.next()? {
        let sequence: i64 = row.get(0)?;
        let event_id = row.get_ref(1)?.as_str()?;
        let timestamp: f64 = row.get(2)?;
        let request = row.get_ref(3)?.as_str()?;
        let payload = row.get_ref(4)?.as_str()?;
        let result = row.get_ref(5)?.as_str()?;
        let event = decode_event(payload)?;
        let expected: Value = serde_json::from_str(result)
            .with_context(|| format!("Decoding saved result for interaction event {sequence}"))?;
        let actual = facts::apply(target, event_id, sequence, timestamp, &event)
            .with_context(|| format!("Replaying interaction event {sequence} ({event_id})"))?;
        ensure!(
            actual == expected,
            "Saved interaction result mismatch at sequence {sequence} ({event_id})"
        );
        insert.execute(params![
            sequence, event_id, timestamp, request, payload, result
        ])?;
    }
    Ok(())
}

fn compare_rows(source: &Connection, replay: &Connection, table: &FactTable) -> Result<()> {
    let mut left_statement = source.prepare(table.ordered_rows)?;
    let mut right_statement = replay.prepare(table.ordered_rows)?;
    let columns = left_statement.column_count();
    ensure!(
        columns == right_statement.column_count(),
        "Interaction fact column count mismatch in {}",
        table.name
    );
    let mut left = left_statement.query([])?;
    let mut right = right_statement.query([])?;
    loop {
        match (left.next()?, right.next()?) {
            (None, None) => break,
            (Some(a), Some(b)) => {
                for column in 0..columns {
                    ensure!(
                        a.get_ref(column)? == b.get_ref(column)?,
                        "Interaction fact mismatch in {} column {column}",
                        table.name
                    );
                }
            }
            _ => bail!("Interaction fact row count mismatch in {}", table.name),
        }
    }
    Ok(())
}

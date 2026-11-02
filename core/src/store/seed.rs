//! Exact native detail rows carried by the first canonical journal event.
use anyhow::{Context, Result, ensure};
use rusqlite::types::{ToSql, ToSqlOutput, ValueRef};
use rusqlite::{Connection, params_from_iter};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct InitialFacts {
    pub tables: Vec<InitialTable>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct InitialTable {
    pub table: FactTable,
    pub rows: Vec<Vec<SqlValue>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum FactTable {
    SelectionFacts,
    OfferFacts,
    RatingFacts,
    ReviewFacts,
    PresentationFacts,
    ActivationFacts,
    PreferenceFacts,
    LegacyDeliveryFacts,
}

impl FactTable {
    pub(super) const ALL: [Self; 8] = [
        Self::SelectionFacts,
        Self::OfferFacts,
        Self::RatingFacts,
        Self::ReviewFacts,
        Self::PresentationFacts,
        Self::ActivationFacts,
        Self::PreferenceFacts,
        Self::LegacyDeliveryFacts,
    ];

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::SelectionFacts => "selection_facts",
            Self::OfferFacts => "offer_facts",
            Self::RatingFacts => "rating_facts",
            Self::ReviewFacts => "review_facts",
            Self::PresentationFacts => "presentation_facts",
            Self::ActivationFacts => "activation_facts",
            Self::PreferenceFacts => "preference_facts",
            Self::LegacyDeliveryFacts => "legacy_delivery_facts",
        }
    }
}

// Encode REALs as bits rather than decimal JSON numbers. Opaque TEXT and BLOB
// values are bound by reference, without parsing or re-encoding their contents.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum SqlValue {
    Null,
    Integer(i64),
    RealBits(u64),
    Text(String),
    Blob(Vec<u8>),
}

impl ToSql for SqlValue {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Borrowed(match self {
            Self::Null => ValueRef::Null,
            Self::Integer(value) => ValueRef::Integer(*value),
            Self::RealBits(bits) => ValueRef::Real(f64::from_bits(*bits)),
            Self::Text(value) => ValueRef::Text(value.as_bytes()),
            Self::Blob(value) => ValueRef::Blob(value),
        }))
    }
}

pub(super) fn apply(connection: &Connection, sequence: i64, facts: &InitialFacts) -> Result<()> {
    ensure!(sequence == 1, "Initial facts must have sequence 1");
    let mut seen = [false; FactTable::ALL.len()];
    for table in &facts.tables {
        let name = table.table.name();
        let already_seen = &mut seen[table.table as usize];
        ensure!(!*already_seen, "Duplicate initial fact family: {name}");
        *already_seen = true;
        let column_count: i64 = connection.query_row(
            "SELECT COUNT(*) FROM pragma_table_info(?1)",
            [name],
            |row| row.get(0),
        )?;
        let column_count = usize::try_from(column_count)?;
        ensure!(column_count > 0, "Missing native fact table: {name}");
        for row in &table.rows {
            ensure!(
                row.len() == column_count,
                "Invalid initial {name} row: expected {column_count} columns, got {}",
                row.len()
            );
            ensure!(
                matches!(row.first(), Some(SqlValue::Integer(value)) if *value == sequence),
                "Initial {name} row must have sequence {sequence}"
            );
        }
        if table.rows.is_empty() {
            continue;
        }
        // VALUES follows the current table's canonical column order. Counts
        // come from SQLite rather than a second catalog of native fields.
        let mut sql = format!("INSERT INTO {name} VALUES(");
        for index in 0..column_count {
            if index != 0 {
                sql.push(',');
            }
            sql.push('?');
        }
        sql.push(')');
        let mut statement = connection.prepare(&sql)?;
        for row in &table.rows {
            statement
                .execute(params_from_iter(row.iter()))
                .with_context(|| format!("Inserting initial {name} row"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn capture(connection: &Connection) -> Result<InitialFacts> {
    let mut facts = InitialFacts::default();
    for table in FactTable::ALL {
        let name = table.name();
        let mut statement =
            connection.prepare(&format!("SELECT * FROM {name} ORDER BY sequence,ordinal"))?;
        let column_count = statement.column_count();
        let mut rows = statement.query([])?;
        let mut captured = Vec::new();
        while let Some(row) = rows.next()? {
            let mut values = Vec::with_capacity(column_count);
            for index in 0..column_count {
                values.push(match row.get_ref(index)? {
                    ValueRef::Null => SqlValue::Null,
                    ValueRef::Integer(value) => SqlValue::Integer(value),
                    ValueRef::Real(value) => SqlValue::RealBits(value.to_bits()),
                    ValueRef::Text(value) => SqlValue::Text(
                        std::str::from_utf8(value)
                            .with_context(|| format!("Reading initial {name} TEXT"))?
                            .to_owned(),
                    ),
                    ValueRef::Blob(value) => SqlValue::Blob(value.to_vec()),
                });
            }
            captured.push(values);
        }
        if !captured.is_empty() {
            facts.tables.push(InitialTable {
                table,
                rows: captured,
            });
        }
    }
    Ok(facts)
}

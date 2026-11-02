use super::TrainingLabel;
use crate::corpus::Corpus;
use anyhow::{Result, ensure};
use std::collections::HashSet;

/// Aligned, validated input owned once for the duration of a training run.
pub(super) struct PreparedTraining {
    rows: Vec<PreparedRow>,
}
pub(super) struct PreparedRow {
    pub label: TrainingLabel,
    pub row: usize,
}
#[derive(Clone, Copy)]
pub(super) struct RowView<'a> {
    rows: &'a [PreparedRow],
    indices: Option<&'a [usize]>,
}
impl PreparedTraining {
    pub fn new(corpus: &Corpus, source: &[TrainingLabel]) -> Result<Self> {
        let mut seen = HashSet::with_capacity(source.len());
        let mut rows = Vec::with_capacity(source.len());
        for label in source {
            // Invalid/duplicate input is rejected even when absent from the corpus.
            ensure!(
                label.grade <= 3 && label.updated_at.is_finite(),
                "Invalid current grade or timestamp"
            );
            ensure!(
                seen.insert(label.post_id),
                "Duplicate current training label {}",
                label.post_id
            );
            if let Some(row) = corpus.find_post(label.post_id) {
                rows.push(PreparedRow {
                    label: label.clone(),
                    row,
                });
            }
        }
        rows.sort_by(|a, b| {
            a.label
                .updated_at
                .total_cmp(&b.label.updated_at)
                .then(a.label.post_id.cmp(&b.label.post_id))
        });
        Ok(Self { rows })
    }
    pub fn all(&self) -> RowView<'_> {
        RowView {
            rows: &self.rows,
            indices: None,
        }
    }
    pub fn select<'a>(&'a self, indices: &'a [usize]) -> RowView<'a> {
        RowView {
            rows: &self.rows,
            indices: Some(indices),
        }
    }
}
impl<'a> RowView<'a> {
    pub fn len(self) -> usize {
        self.indices.map_or(self.rows.len(), <[usize]>::len)
    }
    pub fn get(self, index: usize) -> &'a PreparedRow {
        &self.rows[self.indices.map_or(index, |indices| indices[index])]
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = &'a PreparedRow> + Clone {
        (0..self.len()).map(move |index| self.get(index))
    }
    pub fn labels(self) -> impl ExactSizeIterator<Item = &'a TrainingLabel> + Clone {
        self.iter().map(|row| &row.label)
    }
}

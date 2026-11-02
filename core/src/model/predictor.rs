use super::decoder::RawLogits;
use super::prepared::RowView;
use super::recipe::{AdaptivePrior, PredictorLoss, SupportRecipe};
use super::sparse::{Objective, SparseRows, minimize};
use super::{FeaturePrior, ModelRecipe};
use crate::corpus::Corpus;
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopularityPredictorArtifact {
    pub mean: f64,
    pub scale: f64,
    pub coefficients: [f64; 3],
}
impl PopularityPredictorArtifact {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.mean.is_finite()
                && self.scale.is_finite()
                && self.scale > 0.0
                && self.coefficients.iter().all(|c| c.is_finite()),
            "Invalid popularity normalization or coefficients"
        );
        Ok(())
    }
    pub(super) fn standardized(&self, value: f64) -> f64 {
        (value - self.mean) / self.scale
    }
    pub(super) fn fit(values: impl ExactSizeIterator<Item = f64>) -> Result<Self> {
        let count = values.len();
        ensure!(count > 0, "Empty popularity training partition");
        // Unweighted population variance, in training-row order, using Welford.
        let (mut mean, mut m2) = (0.0, 0.0);
        for (n, value) in values.enumerate() {
            let delta = value - mean;
            mean += delta / (n + 1) as f64;
            m2 += delta * (value - mean);
        }
        let variance = m2 / count as f64;
        ensure!(
            variance.is_finite() && variance >= 0.0,
            "Invalid popularity training variance"
        );
        let result = Self {
            mean,
            scale: if variance == 0.0 {
                1.0
            } else {
                variance.sqrt()
            },
            coefficients: [0.0; 3],
        };
        result.validate()?;
        Ok(result)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawHeadArtifact {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawPredictorArtifact {
    pub selected_features: Vec<usize>,
    pub heads: [RawHeadArtifact; 3],
    pub popularity: PopularityPredictorArtifact,
}
impl RawPredictorArtifact {
    pub fn validate(&self, tag_count: usize) -> Result<()> {
        self.popularity.validate()?;
        ensure!(
            !self.selected_features.is_empty(),
            "Predictor has no selected features"
        );
        let mut seen = vec![false; tag_count];
        for &tag in &self.selected_features {
            ensure!(
                tag < tag_count && !seen[tag],
                "Invalid or repeated selected predictor feature"
            );
            seen[tag] = true;
        }
        for h in &self.heads {
            ensure!(
                h.coefficients.len() == tag_count
                    && h.coefficients.iter().all(|x| x.is_finite())
                    && h.intercept.is_finite(),
                "Invalid predictor dimensions or nonfinite parameters"
            );
            ensure!(
                h.coefficients
                    .iter()
                    .zip(&seen)
                    .all(|(&value, &selected)| selected || value == 0.0),
                "Nonzero coefficient outside selected predictor features"
            );
        }
        Ok(())
    }
}

pub(super) struct RawHead {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
}
pub(super) struct RawPredictor {
    selected_features: Vec<usize>,
    pub heads: [RawHead; 3],
    pub popularity: PopularityPredictorArtifact,
}
impl RawPredictor {
    // The enclosing import validates the complete DTO once.
    pub fn from_artifact(dto: RawPredictorArtifact) -> Self {
        Self {
            selected_features: dto.selected_features,
            heads: dto.heads.map(|h| RawHead {
                coefficients: h.coefficients,
                intercept: h.intercept,
            }),
            popularity: dto.popularity,
        }
    }
    pub fn export(&self) -> RawPredictorArtifact {
        RawPredictorArtifact {
            selected_features: self.selected_features.clone(),
            heads: self.heads.each_ref().map(|h| RawHeadArtifact {
                coefficients: h.coefficients.clone(),
                intercept: h.intercept,
            }),
            popularity: self.popularity,
        }
    }
    pub fn raw(&self, tags: &[i32], popularity: f64) -> RawLogits {
        let mut raw = self.heads.each_ref().map(|h| h.intercept);
        for &tag in tags {
            for h in 0..3 {
                raw[h] += self.heads[h].coefficients[tag as usize];
            }
        }
        let numeric = self.popularity.standardized(popularity);
        for h in 0..3 {
            raw[h] += self.popularity.coefficients[h] * numeric;
        }
        raw
    }
}
fn balanced_weights(labels: &[u8]) -> Vec<f64> {
    let positives = labels.iter().filter(|&&y| y == 1).count();
    let negatives = labels.len() - positives;
    if positives == 0 || negatives == 0 {
        vec![1.0; labels.len()]
    } else {
        labels
            .iter()
            .map(|&y| {
                labels.len() as f64 / (2.0 * (if y == 1 { positives } else { negatives }) as f64)
            })
            .collect()
    }
}
fn scales_for(
    rows: &SparseRows,
    indices: &[usize],
    labels: &[u8],
    feature_count: usize,
    prior: FeaturePrior,
) -> Vec<f64> {
    if prior == FeaturePrior::Uniform {
        return vec![1.0; feature_count];
    }
    let mut positive = vec![AdaptivePrior::ALPHA; feature_count];
    let mut negative = vec![AdaptivePrior::ALPHA; feature_count];
    for (&row, &label) in indices.iter().zip(labels) {
        let counts = if label == 1 {
            &mut positive
        } else {
            &mut negative
        };
        for &column in rows.row(row) {
            counts[column] += 1.0;
        }
    }
    let positive_total = positive.iter().sum::<f64>();
    let negative_total = negative.iter().sum::<f64>();
    (0..feature_count)
        .map(|c| {
            ((positive[c] / positive_total) / (negative[c] / negative_total))
                .ln()
                .abs()
                .clamp(AdaptivePrior::FLOOR, AdaptivePrior::CEILING)
        })
        .collect()
}
fn fit_head(
    rows: &SparseRows,
    indices: &[usize],
    labels: &[u8],
    scales: &[f64],
    numeric: &[f64],
    recipe: &ModelRecipe,
) -> Result<Vec<f64>> {
    let (samples, penalize_intercept) = match recipe.statistical.loss {
        PredictorLoss::BalancedInterceptRidge => (balanced_weights(labels), true),
    };
    let objective = Objective {
        rows,
        indices,
        labels,
        scales,
        sample_weights: &samples,
        c: recipe.statistical.c,
        penalize_intercept,
        numeric,
    };
    minimize(
        vec![0.0; scales.len() + 2],
        recipe.numerical.predictor.tolerance,
        recipe.numerical.predictor.iterations,
        |w, g| objective.evaluate(w, g),
    )
}

pub(super) fn supported_features(
    corpus: &Corpus,
    view: RowView<'_>,
) -> impl Iterator<Item = usize> {
    let mut support = vec![0usize; corpus.tag_count];
    let mut is_artist = vec![false; corpus.tag_count];
    for &tag in corpus.artist_vocab_indices() {
        is_artist[tag as usize] = true;
    }
    for item in view.iter() {
        for &tag in corpus.post_tags(item.row) {
            support[tag as usize] += 1;
        }
    }
    (0..corpus.tag_count).filter(move |&tag| {
        support[tag]
            >= if is_artist[tag] {
                SupportRecipe::ARTIST
            } else {
                SupportRecipe::CONTENT
            }
    })
}

struct PredictorRows<'a> {
    view: RowView<'a>,
    selected_features: Vec<usize>,
    rows: SparseRows,
    all: Vec<usize>,
}
impl<'a> PredictorRows<'a> {
    fn prepare(corpus: &Corpus, view: RowView<'a>) -> Option<Self> {
        if view.len() < SupportRecipe::RATINGS {
            return None;
        }
        let selected_features = supported_features(corpus, view).collect::<Vec<_>>();
        if selected_features.is_empty() {
            return None;
        }
        let mut local_by_global = vec![usize::MAX; corpus.tag_count];
        for (local, &tag) in selected_features.iter().enumerate() {
            local_by_global[tag] = local;
        }
        let mut rows = SparseRows {
            offsets: Vec::with_capacity(view.len() + 1),
            columns: Vec::new(),
        };
        rows.offsets.push(0);
        for item in view.iter() {
            for &tag in corpus.post_tags(item.row) {
                let local = local_by_global[tag as usize];
                if local != usize::MAX {
                    rows.columns.push(local);
                }
            }
            rows.offsets.push(rows.columns.len());
        }
        Some(Self {
            view,
            selected_features,
            rows,
            all: (0..view.len()).collect(),
        })
    }
    fn binary(&self, threshold: u8) -> Vec<u8> {
        self.view
            .labels()
            .map(|label| u8::from(label.grade >= threshold))
            .collect()
    }
    fn fit_full_head(
        &self,
        corpus: &Corpus,
        threshold: u8,
        binary: &[u8],
        recipe: &ModelRecipe,
        numeric: &[f64],
    ) -> Result<(RawHead, f64)> {
        let scales = scales_for(
            &self.rows,
            &self.all,
            binary,
            self.selected_features.len(),
            recipe.statistical.feature_prior,
        );
        let weights = fit_head(&self.rows, &self.all, binary, &scales, numeric, recipe)
            .with_context(|| format!("Ordinal head {threshold} failed"))?;
        let mut coefficients = vec![0.0; corpus.tag_count];
        for (local, &global) in self.selected_features.iter().enumerate() {
            coefficients[global] = weights[local] * scales[local];
        }
        Ok((
            RawHead {
                coefficients,
                intercept: weights[weights.len() - 1],
            },
            weights[self.selected_features.len()],
        ))
    }
}
pub(super) fn fit_raw_predictor(
    corpus: &Corpus,
    view: RowView<'_>,
    recipe: &ModelRecipe,
) -> Result<Option<RawPredictor>> {
    let Some(data) = PredictorRows::prepare(corpus, view) else {
        return Ok(None);
    };
    let mut popularity =
        PopularityPredictorArtifact::fit(view.iter().map(|item| corpus.popularity(item.row)))?;
    let numeric = view
        .iter()
        .map(|item| popularity.standardized(corpus.popularity(item.row)))
        .collect::<Vec<_>>();
    let heads = (1..=3u8)
        .into_par_iter()
        .map(|threshold| {
            data.fit_full_head(corpus, threshold, &data.binary(threshold), recipe, &numeric)
        })
        .collect::<Result<Vec<_>>>()?;
    let heads: [(RawHead, f64); 3] = heads
        .try_into()
        .map_err(|_| anyhow::anyhow!("Expected three ordinal heads"))?;
    popularity.coefficients = heads.each_ref().map(|(_, c)| *c);
    Ok(Some(RawPredictor {
        selected_features: data.selected_features,
        heads: heads.map(|(head, _)| head),
        popularity,
    }))
}

#[cfg(test)]
#[path = "../../tests/unit/model/predictor.rs"]
mod tests;

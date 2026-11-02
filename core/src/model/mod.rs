mod decoder;
mod predictor;
mod prepared;
mod recipe;
mod sparse;
#[cfg(test)]
#[path = "../../tests/unit/model/mod.rs"]
mod tests;

use crate::corpus::Corpus;
use anyhow::{Context, Result, ensure};
pub use decoder::{DecoderArtifact, Forecast, RawLogits, StageDiagnostics, mixture};
pub use predictor::{PopularityPredictorArtifact, RawHeadArtifact, RawPredictorArtifact};
use predictor::{RawPredictor, fit_raw_predictor};
use prepared::{PreparedTraining, RowView};
use rayon::prelude::*;
pub use recipe::{FeaturePrior, ModelRecipe};
use recipe::{PartitionRecipe, SupportRecipe};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingLabel {
    pub post_id: i64,
    pub grade: u8,
    pub session_id: String,
    pub updated_at: f64,
}
#[derive(Clone, Debug)]
pub struct ScoreBuffers {
    pub not_dislike: Vec<f32>,
    pub like_or_love: Vec<f32>,
    pub love: Vec<f32>,
    pub expected_grade: Vec<f32>,
}
impl ScoreBuffers {
    pub fn zeros(count: usize) -> Self {
        Self {
            not_dislike: vec![0.0; count],
            like_or_love: vec![0.0; count],
            love: vec![0.0; count],
            expected_grade: vec![0.0; count],
        }
    }
}
pub struct FitResult {
    pub model_id: String,
    pub scores: Option<ScoreBuffers>,
    pub status: ModelStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemberArtifact {
    pub member: usize,
    pub predictor: RawPredictorArtifact,
    pub decoder: DecoderArtifact,
    pub training_rows: usize,
    pub withheld_rows: usize,
    pub calibration_rows: usize,
    pub stages: [StageDiagnostics; 3],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnsembleArtifact {
    pub members: [MemberArtifact; PartitionRecipe::MEMBERS],
}
impl EnsembleArtifact {
    pub fn validate(&self, corpus: &Corpus) -> Result<()> {
        let tag_count = corpus.tag_count;
        for (index, member) in self.members.iter().enumerate() {
            ensure!(
                member.member == index
                    && member.training_rows >= SupportRecipe::RATINGS
                    && member.calibration_rows >= SupportRecipe::CALIBRATION
                    && member.calibration_rows <= member.withheld_rows,
                "Invalid ensemble member support"
            );
            member.predictor.validate(tag_count)?;
            member.decoder.validate()?;
            for (h, stage) in member.stages.iter().enumerate() {
                ensure!(
                    stage.negative_outcomes >= SupportRecipe::OUTCOMES
                        && stage.positive_outcomes >= SupportRecipe::OUTCOMES
                        && stage.negative_outcomes.checked_add(stage.positive_outcomes)
                            == Some(stage.risk_rows)
                        && stage.iterations <= recipe::NUMERICAL.decoder.iterations
                        && stage.objective.is_finite()
                        && stage.objective >= 0.0
                        && stage.projected_gradient_maximum.is_finite()
                        && (0.0..=recipe::NUMERICAL.decoder.certificate)
                            .contains(&stage.projected_gradient_maximum),
                    "Invalid decoder convergence/support certificate"
                );
                ensure!(
                    stage.risk_rows
                        == if h == 0 {
                            member.calibration_rows
                        } else {
                            member.stages[h - 1].positive_outcomes
                        },
                    "Inconsistent conditional risk counts"
                );
            }
        }
        let total = self.members[0]
            .training_rows
            .checked_add(self.members[0].withheld_rows)
            .context("Ensemble source count overflow")?;
        ensure!(
            self.members
                .iter()
                .all(|m| m.training_rows.checked_add(m.withheld_rows) == Some(total)),
            "Inconsistent ensemble source counts"
        );
        ensure!(
            self.members
                .iter()
                .try_fold(0usize, |sum, m| sum.checked_add(m.withheld_rows))
                == Some(total),
            "Inconsistent ensemble withheld source counts"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "artifact",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ModelArtifact {
    Ensemble(EnsembleArtifact),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainedModelArtifact {
    pub version: u32,
    pub model_id: String,
    pub corpus_id: String,
    pub corpus_manifest_sha256: String,
    pub tag_count: usize,
    pub artifact: ModelArtifact,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelRoute {
    Starter,
    Ensemble,
}
impl ModelRoute {
    fn identity_name(self) -> &'static str {
        match self {
            Self::Starter => "starter",
            Self::Ensemble => "ensemble",
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelStatus {
    pub route: ModelRoute,
    /// None for supported ensemble fits; imported artifacts do not record source
    /// support evidence, so None there means not recorded.
    pub fallback_reason: Option<SupportReason>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SupportReason {
    InsufficientLabels {
        actual: usize,
        minimum: usize,
    },
    InsufficientSessions {
        actual: usize,
        minimum: usize,
    },
    MemberTrainingRows {
        member: usize,
        actual: usize,
        minimum: usize,
    },
    MemberCalibrationRows {
        member: usize,
        actual: usize,
        minimum: usize,
    },
    /// Conditional stages are numbered 1..=3; member indices are 0..=4.
    ConditionalOutcomes {
        member: usize,
        stage: usize,
        negative: usize,
        positive: usize,
        minimum: usize,
    },
}

/// Immutable serving state, with no personal predictor until Ensemble support.
pub struct TrainedModel<'a> {
    corpus: &'a Corpus,
    model_id: String,
    runtime: ModelRuntime,
    status: ModelStatus,
}
enum ModelRuntime {
    Starter,
    Ensemble([Member; PartitionRecipe::MEMBERS]),
}
struct Member {
    member: usize,
    predictor: RawPredictor,
    decoder: decoder::Decoder,
    training_rows: usize,
    withheld_rows: usize,
    calibration_rows: usize,
    stages: [StageDiagnostics; 3],
}
impl Member {
    fn from_artifact(dto: MemberArtifact) -> Self {
        Self {
            member: dto.member,
            predictor: RawPredictor::from_artifact(dto.predictor),
            decoder: decoder::Decoder::from_artifact(dto.decoder),
            training_rows: dto.training_rows,
            withheld_rows: dto.withheld_rows,
            calibration_rows: dto.calibration_rows,
            stages: dto.stages,
        }
    }
    fn export(&self) -> MemberArtifact {
        MemberArtifact {
            member: self.member,
            predictor: self.predictor.export(),
            decoder: self.decoder.export(),
            training_rows: self.training_rows,
            withheld_rows: self.withheld_rows,
            calibration_rows: self.calibration_rows,
            stages: self.stages.clone(),
        }
    }
}
impl<'a> TrainedModel<'a> {
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
    pub fn route(&self) -> ModelRoute {
        self.status.route
    }
    /// Export only a supported, genuinely trained ensemble.
    pub fn export(&self) -> Result<TrainedModelArtifact> {
        let corpus = self.corpus;
        let ModelRuntime::Ensemble(members) = &self.runtime else {
            anyhow::bail!("Starter has no trained model artifact");
        };
        Ok(TrainedModelArtifact {
            version: 4,
            model_id: self.model_id.clone(),
            corpus_id: corpus.corpus_id.clone(),
            corpus_manifest_sha256: corpus.manifest_sha256.clone(),
            tag_count: corpus.tag_count,
            artifact: ModelArtifact::Ensemble(EnsembleArtifact {
                members: members.each_ref().map(Member::export),
            }),
        })
    }
    pub fn from_artifact(corpus: &'a Corpus, dto: TrainedModelArtifact) -> Result<Self> {
        ensure!(
            dto.version == 4,
            "Unsupported native model artifact version"
        );
        ensure!(
            dto.corpus_id == corpus.corpus_id
                && dto.corpus_manifest_sha256 == corpus.manifest_sha256
                && dto.tag_count == corpus.tag_count,
            "Model corpus identity mismatch"
        );
        let (runtime, route) = match dto.artifact {
            ModelArtifact::Ensemble(e) => {
                e.validate(corpus)?;
                (
                    ModelRuntime::Ensemble(e.members.map(Member::from_artifact)),
                    ModelRoute::Ensemble,
                )
            }
        };
        Ok(Self {
            corpus,
            model_id: dto.model_id,
            runtime,
            status: ModelStatus {
                route,
                fallback_reason: None,
            },
        })
    }
    pub fn score_all(&self) -> Result<Option<ScoreBuffers>> {
        match &self.runtime {
            ModelRuntime::Starter => Ok(None),
            ModelRuntime::Ensemble(members) => {
                let predictors = members.each_ref().map(|m| &m.predictor);
                let decoders = members.each_ref().map(|m| &m.decoder);
                score_packed(self.corpus, &predictors, &decoders).map(Some)
            }
        }
    }
    /// Reference ensemble forecasts retain f64 stable log masses.
    pub fn predict_rows(&self, rows: &[usize]) -> Result<Vec<Forecast>> {
        let ModelRuntime::Ensemble(members) = &self.runtime else {
            anyhow::bail!("Starter has no personal forecasts");
        };
        ensure!(
            rows.iter().all(|&r| r < self.corpus.post_count),
            "Prediction row outside corpus"
        );
        rows.iter()
            .map(|&row| {
                let [a, b, c, d, e] = members.each_ref().map(|m| {
                    m.decoder.forecast(
                        m.predictor
                            .raw(self.corpus.post_tags(row), self.corpus.popularity(row)),
                    )
                });
                Ok(mixture(&[a?, b?, c?, d?, e?]))
            })
            .collect()
    }
}

fn session_key(label: &TrainingLabel) -> String {
    if label.session_id.trim().is_empty() {
        format!("post:{}", label.post_id)
    } else {
        format!("session:{}", label.session_id)
    }
}
pub fn member_for(label: &TrainingLabel) -> usize {
    let mut hash = Sha256::new();
    hash.update(PartitionRecipe::SEED.as_bytes());
    hash.update(b"|member|");
    hash.update(session_key(label).as_bytes());
    hash.finalize().iter().fold(0usize, |remainder, &value| {
        (remainder * 256 + value as usize) % PartitionRecipe::MEMBERS
    })
}
/// Accumulate newest whole groups toward ceil(20% of rows), retaining the oldest
/// even when that prevents reaching the target. Ties use Rust String ordering;
/// results preserve input order. A single group remains entirely in training.
pub(super) fn partition_recent<'a>(
    labels: impl ExactSizeIterator<Item = &'a TrainingLabel>,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let count = labels.len();
    let mut groups: BTreeMap<String, (f64, Vec<usize>)> = BTreeMap::new();
    for (index, label) in labels.enumerate() {
        ensure!(label.updated_at.is_finite(), "Nonfinite label chronology");
        let key = session_key(label);
        let entry = groups.entry(key).or_insert((label.updated_at, Vec::new()));
        entry.0 = entry.0.max(label.updated_at);
        entry.1.push(index);
    }
    if groups.len() < PartitionRecipe::MIN_SESSIONS {
        return Ok(((0..count).collect(), Vec::new()));
    }
    let mut ordered = groups.into_iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.1.0.total_cmp(&b.1.0).then_with(|| a.0.cmp(&b.0)));
    let target = ((count as f64 * PartitionRecipe::RECENT).ceil() as usize).max(1);
    let mut boundary = ordered.len();
    let mut held_count = 0;
    while boundary > 1 && held_count < target {
        boundary -= 1;
        held_count += ordered[boundary].1.1.len();
    }
    let mut held_mask = vec![false; count];
    for (_, (_, indices)) in &ordered[boundary..] {
        for &i in indices {
            held_mask[i] = true;
        }
    }
    Ok((
        (0..count).filter(|&i| !held_mask[i]).collect(),
        (0..count).filter(|&i| held_mask[i]).collect(),
    ))
}
struct Partition {
    train: Vec<usize>,
    withheld: Vec<usize>,
    calibration: Vec<usize>,
}
fn partitions<'a>(
    labels: impl ExactSizeIterator<Item = &'a TrainingLabel> + Clone,
) -> Result<[Partition; PartitionRecipe::MEMBERS]> {
    let (_, recent) = partition_recent(labels.clone())?;
    let count = labels.len();
    let mut is_recent = vec![false; count];
    for i in recent {
        is_recent[i] = true;
    }
    let assignments = labels.map(member_for).collect::<Vec<_>>();
    Ok(std::array::from_fn(|member| Partition {
        train: (0..count).filter(|&i| assignments[i] != member).collect(),
        withheld: (0..count).filter(|&i| assignments[i] == member).collect(),
        calibration: (0..count)
            .filter(|&i| assignments[i] == member && is_recent[i])
            .collect(),
    }))
}
fn hash_field(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}
fn identity<'a>(
    corpus: &Corpus,
    labels: impl Iterator<Item = &'a TrainingLabel>,
    route: &str,
    recipe: &ModelRecipe,
) -> String {
    let mut hash = Sha256::new();
    hash_field(&mut hash, recipe.identity().as_bytes());
    hash_field(&mut hash, corpus.corpus_id.as_bytes());
    hash_field(&mut hash, corpus.manifest_sha256.as_bytes());
    hash_field(&mut hash, route.as_bytes());
    hash.update((corpus.tag_count as u64).to_le_bytes());
    let mut ordered = labels.collect::<Vec<_>>();
    ordered.sort_by_key(|l| l.post_id);
    for label in ordered {
        hash.update(label.post_id.to_le_bytes());
        hash.update([label.grade]);
        hash_field(&mut hash, label.session_id.as_bytes());
        hash.update(label.updated_at.to_bits().to_le_bytes());
    }
    format!("rust-{route}-v5-{}", hex::encode(hash.finalize()))
}

/// Train immutable executable state without scoring the complete corpus.
pub fn train<'a>(
    corpus: &'a Corpus,
    source_labels: &[TrainingLabel],
    recipe: &ModelRecipe,
) -> Result<TrainedModel<'a>> {
    let prepared = PreparedTraining::new(corpus, source_labels)?;
    train_prepared(corpus, &prepared, recipe, None)
}

fn train_prepared<'a>(
    corpus: &'a Corpus,
    prepared: &PreparedTraining,
    recipe: &ModelRecipe,
    candidate: Option<(ModelRoute, String)>,
) -> Result<TrainedModel<'a>> {
    let labels = prepared.all();
    let independent = labels
        .labels()
        .map(session_key)
        .collect::<HashSet<_>>()
        .len();
    let (parts, fallback_reason) = if labels.len() < SupportRecipe::RATINGS {
        (
            None,
            Some(SupportReason::InsufficientLabels {
                actual: labels.len(),
                minimum: SupportRecipe::RATINGS,
            }),
        )
    } else if independent < PartitionRecipe::MIN_SESSIONS {
        (
            None,
            Some(SupportReason::InsufficientSessions {
                actual: independent,
                minimum: PartitionRecipe::MIN_SESSIONS,
            }),
        )
    } else {
        let parts = partitions(labels.labels())?;
        let failure = support_failure(labels, &parts);
        (Some(parts), failure)
    };
    let runtime = if let Some(parts) = parts.filter(|_| fallback_reason.is_none()) {
        let degree = (rayon::current_num_threads() / 3).clamp(1, PartitionRecipe::MEMBERS);
        let batches = (0..degree)
            .into_par_iter()
            .map(|worker| -> Result<Vec<Member>> {
                let mut result = Vec::new();
                for member in (worker..PartitionRecipe::MEMBERS).step_by(degree) {
                    result.push(fit_member(
                        corpus,
                        prepared,
                        recipe,
                        member,
                        &parts[member],
                    )?);
                }
                Ok(result)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut members = batches.into_iter().flatten().collect::<Vec<_>>();
        members.sort_by_key(|m| m.member);
        let members = members
            .try_into()
            .map_err(|_| anyhow::anyhow!("Expected five ensemble members"))?;
        ModelRuntime::Ensemble(members)
    } else {
        ModelRuntime::Starter
    };
    let route = match &runtime {
        ModelRuntime::Starter => ModelRoute::Starter,
        ModelRuntime::Ensemble(_) => ModelRoute::Ensemble,
    };
    Ok(TrainedModel {
        model_id: match candidate {
            Some((candidate_route, id)) if candidate_route == route => id,
            _ => identity(corpus, labels.labels(), route.identity_name(), recipe),
        },
        corpus,
        runtime,
        status: ModelStatus {
            route,
            fallback_reason,
        },
    })
}
pub fn fit(corpus: &Corpus, labels: &[TrainingLabel]) -> Result<FitResult> {
    let recipe = ModelRecipe::production();
    let model = train(corpus, labels, &recipe)?;
    score_model(model)
}

/// Reuse only a successful serving identity, after validating every source label.
pub(crate) fn fit_if_changed(
    corpus: &Corpus,
    labels: &[TrainingLabel],
    serving_model_id: &str,
    serving_route: ModelRoute,
) -> Result<Option<FitResult>> {
    fit_changed_with_recipe(
        corpus,
        labels,
        &ModelRecipe::production(),
        serving_model_id,
        serving_route,
    )
}

fn fit_changed_with_recipe(
    corpus: &Corpus,
    labels: &[TrainingLabel],
    recipe: &ModelRecipe,
    serving_model_id: &str,
    serving_route: ModelRoute,
) -> Result<Option<FitResult>> {
    let prepared = PreparedTraining::new(corpus, labels)?;
    let candidate = identity(
        corpus,
        prepared.all().labels(),
        serving_route.identity_name(),
        recipe,
    );
    if candidate == serving_model_id {
        return Ok(None);
    }
    let model = train_prepared(corpus, &prepared, recipe, Some((serving_route, candidate)))?;
    score_model(model).map(Some)
}

fn score_model(model: TrainedModel<'_>) -> Result<FitResult> {
    let scores = model.score_all()?;
    Ok(FitResult {
        model_id: model.model_id,
        scores,
        status: model.status,
    })
}

/// The sole member fitting owner.
fn fit_member(
    corpus: &Corpus,
    prepared: &PreparedTraining,
    recipe: &ModelRecipe,
    member: usize,
    p: &Partition,
) -> Result<Member> {
    let predictor = fit_raw_predictor(corpus, prepared.select(&p.train), recipe)?
        .context("Supported ensemble member has no selectable predictor features")?;
    let raw = prepared
        .select(&p.calibration)
        .iter()
        .map(|item| predictor.raw(corpus.post_tags(item.row), corpus.popularity(item.row)))
        .collect::<Vec<_>>();
    let grades = prepared
        .select(&p.calibration)
        .labels()
        .map(|label| label.grade)
        .collect::<Vec<_>>();
    let (decoder, stages) = DecoderArtifact::fit_with_numerical(&raw, &grades, recipe.numerical)
        .with_context(|| format!("Conditional member {member} failed"))?;
    Ok(Member {
        member,
        predictor,
        decoder: decoder::Decoder::from_artifact(decoder),
        training_rows: p.train.len(),
        withheld_rows: p.withheld.len(),
        calibration_rows: p.calibration.len(),
        stages,
    })
}

fn support_failure(
    labels: RowView<'_>,
    parts: &[Partition; PartitionRecipe::MEMBERS],
) -> Option<SupportReason> {
    if labels.len() < SupportRecipe::RATINGS {
        return Some(SupportReason::InsufficientLabels {
            actual: labels.len(),
            minimum: SupportRecipe::RATINGS,
        });
    }
    for (member, p) in parts.iter().enumerate() {
        if p.train.len() < SupportRecipe::RATINGS {
            return Some(SupportReason::MemberTrainingRows {
                member,
                actual: p.train.len(),
                minimum: SupportRecipe::RATINGS,
            });
        }
        if p.calibration.len() < SupportRecipe::CALIBRATION {
            return Some(SupportReason::MemberCalibrationRows {
                member,
                actual: p.calibration.len(),
                minimum: SupportRecipe::CALIBRATION,
            });
        }
        for stage in 0..3 {
            let negative = p
                .calibration
                .iter()
                .filter(|&&i| labels.get(i).label.grade == stage)
                .count();
            let positive = p
                .calibration
                .iter()
                .filter(|&&i| labels.get(i).label.grade > stage)
                .count();
            if negative.min(positive) < SupportRecipe::OUTCOMES {
                return Some(SupportReason::ConditionalOutcomes {
                    member,
                    stage: stage as usize + 1,
                    negative,
                    positive,
                    minimum: SupportRecipe::OUTCOMES,
                });
            }
        }
    }
    None
}
fn score_packed(
    corpus: &Corpus,
    predictors: &[&RawPredictor; PartitionRecipe::MEMBERS],
    decoders: &[&decoder::Decoder; PartitionRecipe::MEMBERS],
) -> Result<ScoreBuffers> {
    let stride = 16;
    let mut coefficients = vec![0.0; corpus.tag_count * stride];
    let mut intercepts = [0.0; 16];
    for (m, predictor) in predictors.iter().enumerate() {
        for h in 0..3 {
            let lane = m * 3 + h;
            intercepts[lane] = predictor.heads[h].intercept;
            for tag in 0..corpus.tag_count {
                coefficients[tag * stride + lane] = predictor.heads[h].coefficients[tag];
            }
        }
    }
    let kernel = packed_kernel();
    let mut scores = ScoreBuffers::zeros(corpus.post_count);
    scores
        .not_dislike
        .par_chunks_mut(8192)
        .zip(scores.like_or_love.par_chunks_mut(8192))
        .zip(scores.love.par_chunks_mut(8192))
        .zip(scores.expected_grade.par_chunks_mut(8192))
        .enumerate()
        .try_for_each(|(chunk, (((a, b), c), expected))| -> Result<()> {
            for i in 0..a.len() {
                // The chosen AVX kernel requires runtime feature detection;
                // corpus validation and packed construction prove its bounds.
                let mut logits = unsafe {
                    kernel(
                        corpus.post_tags(chunk * 8192 + i),
                        &coefficients,
                        &intercepts,
                        stride,
                    )
                };
                // Append after tags, in the scalar reference's operation order.
                let value = corpus.popularity(chunk * 8192 + i);
                for (m, predictor) in predictors.iter().enumerate() {
                    let numeric = predictor.popularity.standardized(value);
                    for h in 0..3 {
                        logits[m * 3 + h] += predictor.popularity.coefficients[h] * numeric;
                    }
                }
                ensure!(
                    logits[..predictors.len() * 3].iter().all(|x| x.is_finite()),
                    "Nonfinite packed raw logits at corpus row {}",
                    chunk * 8192 + i
                );
                let mut sum = [0.0; 3];
                for m in 0..PartitionRecipe::MEMBERS {
                    let q = decoders[m]
                        .cumulative([logits[m * 3], logits[m * 3 + 1], logits[m * 3 + 2]])
                        .with_context(|| {
                            format!("Packed decoder member {m}, corpus row {}", chunk * 8192 + i)
                        })?;
                    for h in 0..3 {
                        sum[h] += q[h];
                    }
                }
                let q = sum.map(|v| v / PartitionRecipe::MEMBERS as f64);
                a[i] = q[0] as f32;
                b[i] = q[1] as f32;
                c[i] = q[2] as f32;
                expected[i] = (a[i] + b[i]) + c[i];
            }
            Ok(())
        })?;
    Ok(scores)
}

type PackedKernel = unsafe fn(&[i32], &[f64], &[f64; 16], usize) -> [f64; 16];
fn packed_kernel() -> PackedKernel {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx") {
        return packed_avx;
    }
    packed_scalar
}
fn packed_scalar(
    tags: &[i32],
    coefficients: &[f64],
    intercepts: &[f64; 16],
    stride: usize,
) -> [f64; 16] {
    let mut logits = *intercepts;
    for &tag in tags {
        let base = tag as usize * stride;
        for lane in 0..stride {
            logits[lane] += coefficients[base + lane];
        }
    }
    logits
}
/// Same per-lane tag-addition order as the scalar reference, no FMA/reduction.
/// Safety: AVX available; stride is 16; tags are nonnegative vocabulary
/// indices; coefficients has tag_count * stride entries.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn packed_avx(
    tags: &[i32],
    coefficients: &[f64],
    intercepts: &[f64; 16],
    stride: usize,
) -> [f64; 16] {
    use std::arch::x86_64::{_mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};
    unsafe {
        let mut logits = *intercepts;
        let mut v0 = _mm256_loadu_pd(intercepts.as_ptr());
        let mut v1 = _mm256_loadu_pd(intercepts.as_ptr().add(4));
        let mut v2 = _mm256_loadu_pd(intercepts.as_ptr().add(8));
        let mut v3 = _mm256_loadu_pd(intercepts.as_ptr().add(12));
        for &tag in tags {
            let p = coefficients.as_ptr().add(tag as usize * stride);
            v0 = _mm256_add_pd(v0, _mm256_loadu_pd(p));
            v1 = _mm256_add_pd(v1, _mm256_loadu_pd(p.add(4)));
            v2 = _mm256_add_pd(v2, _mm256_loadu_pd(p.add(8)));
            v3 = _mm256_add_pd(v3, _mm256_loadu_pd(p.add(12)));
        }
        _mm256_storeu_pd(logits.as_mut_ptr(), v0);
        _mm256_storeu_pd(logits.as_mut_ptr().add(4), v1);
        _mm256_storeu_pd(logits.as_mut_ptr().add(8), v2);
        _mm256_storeu_pd(logits.as_mut_ptr().add(12), v3);
        logits
    }
}

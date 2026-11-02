use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use anyhow::{Result, bail, ensure};
use sha2::{Digest, Sha256};

use crate::catalog::TagCatalog;
use crate::corpus::Corpus;
use crate::model::ScoreBuffers;
use crate::types::{Facet, Grade, StoredRating};

/// Native Gallery ranking policy recorded on newly offered slates.
pub const GALLERY_POLICY: &str = "gallery-rust-v7";
// Delivery/cohort policy revisions must not reseed Train Broadly priorities.
const TRAIN_BROADLY_SAMPLER_DOMAIN: &str = "gallery-rust-v2";
/// Native Auto continuation/admission policy recorded on newly offered cards.
pub const AUTO_POLICY: &str = "auto-rust-v6";

pub struct Familiarity {
    pub rated_count: Vec<i32>,
    pub weighted_sentiment: Vec<f32>,
    pub post_rated_count: Vec<i32>,
    pub post_weighted_sentiment: Vec<f32>,
    pub post_has_artist: Vec<bool>,
    sentiment_total: Vec<f32>,
}

impl Familiarity {
    pub fn build(corpus: &Corpus, ratings: &[StoredRating]) -> Self {
        let mut rated_count = vec![0; corpus.artist_count];
        let mut sentiment_total = vec![0.0; corpus.artist_count];
        for rating in ratings {
            if let Some(row) = corpus.find_post(rating.post_id) {
                let sentiment = Self::sentiment(rating.grade);
                for &artist in corpus.post_artists(row) {
                    rated_count[artist as usize] += 1;
                    sentiment_total[artist as usize] += sentiment;
                }
            }
        }
        let weighted_sentiment = sentiment_total
            .iter()
            .zip(&rated_count)
            .map(|(&total, &count)| {
                if count == 0 {
                    0.0
                } else {
                    total / count as f32
                }
            })
            .collect();
        let mut result = Self {
            rated_count,
            weighted_sentiment,
            sentiment_total,
            post_rated_count: vec![0; corpus.post_count],
            post_weighted_sentiment: vec![0.0; corpus.post_count],
            post_has_artist: vec![false; corpus.post_count],
        };
        for row in 0..corpus.post_count {
            result.recompute_post(corpus, row);
        }
        result
    }

    fn sentiment(grade: Grade) -> f32 {
        match grade {
            Grade::Love => 2.0,
            Grade::Like => 1.0,
            Grade::Meh => 0.0,
            Grade::Dislike => -1.0,
        }
    }

    fn recompute_post(&mut self, corpus: &Corpus, row: usize) {
        let artists = corpus.post_artists(row);
        self.post_has_artist[row] = !artists.is_empty();
        let mut maximum = 0;
        let mut total = 0.0;
        for &artist in artists {
            maximum = maximum.max(self.rated_count[artist as usize]);
            total += self.weighted_sentiment[artist as usize];
        }
        self.post_rated_count[row] = maximum;
        self.post_weighted_sentiment[row] = if artists.is_empty() {
            0.0
        } else {
            total / artists.len() as f32
        };
    }

    pub fn update(&mut self, corpus: &Corpus, row: usize, previous: Option<Grade>, next: Grade) {
        if previous == Some(next) {
            return;
        }
        let artists = corpus.post_artists(row);
        let delta = Self::sentiment(next) - previous.map_or(0.0, Self::sentiment);
        for &artist in artists {
            let artist = artist as usize;
            if previous.is_none() {
                self.rated_count[artist] += 1;
            }
            self.sentiment_total[artist] += delta;
            self.weighted_sentiment[artist] =
                self.sentiment_total[artist] / self.rated_count[artist] as f32;
        }
        // Merge the touched artists' already-sorted inverted postings. Every
        // affected post is recomputed once, with O(coartists) auxiliary storage,
        // rather than allocating/clearing a corpus-sized visitation mask.
        let mut postings = BinaryHeap::with_capacity(artists.len());
        for (index, &artist) in artists.iter().enumerate() {
            let tag = corpus.artist_vocab_indices()[artist as usize] as usize;
            if let Some(&first) = corpus.tag_posts(tag).first() {
                postings.push(Reverse((first as usize, index, 0_usize)));
            }
        }
        let mut previous_row = None;
        while let Some(Reverse((affected, index, position))) = postings.pop() {
            if previous_row != Some(affected) {
                self.recompute_post(corpus, affected);
                previous_row = Some(affected);
            }
            let tag = corpus.artist_vocab_indices()[artists[index] as usize] as usize;
            if let Some(&next) = corpus.tag_posts(tag).get(position + 1) {
                postings.push(Reverse((next as usize, index, position + 1)));
            }
        }
    }
}

/// Presentation-only exclusion, compiled once from canonical tag postings.
#[derive(Default)]
pub(crate) struct Blacklist {
    pub names: Vec<String>,
    rows: Vec<u64>,
}
impl Blacklist {
    pub fn compile(catalog: &TagCatalog, corpus: &Corpus, names: &[String]) -> Result<Self> {
        let mut tags = HashSet::new();
        let mut invalid = Vec::new();
        for raw in names {
            let name = raw.trim();
            if name.is_empty() {
                continue;
            }
            if name.starts_with(['+', '-']) || name.chars().any(char::is_whitespace) {
                invalid.push(name.to_owned());
            } else if let Some(tag) = catalog.resolve(name) {
                tags.insert(tag);
            } else {
                invalid.push(name.to_owned());
            }
        }
        ensure!(
            invalid.is_empty(),
            "Invalid blacklist tag(s): {}. Enter one known tag per line, without filter operators.",
            invalid.join(", ")
        );
        let mut tags: Vec<_> = tags.into_iter().collect();
        tags.sort_unstable_by(|&a, &b| catalog.tag(a).name.cmp(&catalog.tag(b).name));
        let mut rows = if tags.is_empty() {
            Vec::new()
        } else {
            vec![0; corpus.post_count.div_ceil(64)]
        };
        for &tag in &tags {
            for &row in corpus.tag_posts(tag) {
                let row = row as usize;
                rows[row / 64] |= 1u64 << (row % 64);
            }
        }
        Ok(Self {
            names: tags
                .into_iter()
                .map(|tag| catalog.tag(tag).name.clone())
                .collect(),
            rows,
        })
    }
    pub fn blocks(&self, row: usize) -> bool {
        self.rows
            .get(row / 64)
            .is_some_and(|word| word & (1u64 << (row % 64)) != 0)
    }
}

#[derive(Clone, Debug)]
pub struct TagFilter {
    pub canonical: String,
    pub required: Vec<usize>,
    pub excluded: Vec<usize>,
}

impl TagFilter {
    pub fn parse(catalog: &TagCatalog, text: &str) -> Result<Self> {
        let mut required = HashSet::new();
        let mut excluded = HashSet::new();
        let mut unknown = Vec::new();
        for raw in text
            .split([' ', '\t', '\r', '\n'])
            .filter(|token| !token.is_empty())
        {
            let normalized = raw.to_lowercase();
            let negative = normalized.starts_with('-');
            let name = normalized.strip_prefix(['+', '-']).unwrap_or(&normalized);
            match catalog.resolve(name).filter(|_| !name.is_empty()) {
                Some(tag) => {
                    if negative {
                        excluded.insert(tag);
                    } else {
                        required.insert(tag);
                    }
                }
                None => {
                    let name = if name.is_empty() { raw } else { name };
                    if !unknown.iter().any(|other| other == name) {
                        unknown.push(name.to_owned());
                    }
                }
            }
        }
        ensure!(
            unknown.is_empty(),
            "Unknown tag(s): {}.",
            unknown.join(", ")
        );
        let mut contradictions: Vec<_> = required
            .intersection(&excluded)
            .map(|&tag| catalog.tag(tag).name.as_str())
            .collect();
        contradictions.sort_unstable();
        ensure!(
            contradictions.is_empty(),
            "Filter both requires and excludes: {}.",
            contradictions.join(", ")
        );
        // Preserve the original filter's 254-required-tag limit without a byte counter.
        ensure!(
            required.len() < 255,
            "A filter cannot require more than 254 tags."
        );
        let mut required: Vec<_> = required.into_iter().collect();
        let mut excluded: Vec<_> = excluded.into_iter().collect();
        required.sort_unstable_by(|&a, &b| catalog.tag(a).name.cmp(&catalog.tag(b).name));
        excluded.sort_unstable_by(|&a, &b| catalog.tag(a).name.cmp(&catalog.tag(b).name));
        let canonical = required
            .iter()
            .map(|&tag| format!("+{}", catalog.tag(tag).name))
            .chain(
                excluded
                    .iter()
                    .map(|&tag| format!("-{}", catalog.tag(tag).name)),
            )
            .collect::<Vec<_>>()
            .join(" ");
        Ok(Self {
            canonical,
            required,
            excluded,
        })
    }

    pub fn matches(&self, corpus: &Corpus, row: usize) -> bool {
        let tags = corpus.post_tags(row);
        self.required
            .iter()
            .all(|&tag| tags.binary_search(&(tag as i32)).is_ok())
            && self
                .excluded
                .iter()
                .all(|&tag| tags.binary_search(&(tag as i32)).is_err())
    }
}

pub fn facets() -> Vec<Facet> {
    vec![
        Facet {
            slug: "discover",
            label: "Discover",
            cohort_size: 30,
            requires_model: false,
        },
        Facet {
            slug: "develop",
            label: "Develop",
            cohort_size: 30,
            requires_model: false,
        },
        Facet {
            slug: "train_broadly",
            label: "Something New",
            cohort_size: 20,
            requires_model: false,
        },
        Facet {
            slug: "q3_probe",
            label: "Love or Like?",
            cohort_size: 20,
            requires_model: true,
        },
        Facet {
            slug: "rating_review",
            label: "Rating Review",
            cohort_size: 100,
            requires_model: true,
        },
    ]
}

pub fn median_grade(scores: &ScoreBuffers, row: usize) -> u8 {
    if scores.love[row] >= 0.5 {
        3
    } else if scores.like_or_love[row] >= 0.5 {
        2
    } else if scores.not_dislike[row] >= 0.5 {
        1
    } else {
        0
    }
}

fn disagreement(scores: &ScoreBuffers, row: usize, grade: Grade) -> f32 {
    let grade = grade as i32;
    (1.0 - scores.not_dislike[row]) * grade.abs() as f32
        + (scores.not_dislike[row] - scores.like_or_love[row]) * (1 - grade).abs() as f32
        + (scores.like_or_love[row] - scores.love[row]) * (2 - grade).abs() as f32
        + scores.love[row] * (3 - grade).abs() as f32
}

fn check_alignment(
    corpus: &Corpus,
    scores: Option<&ScoreBuffers>,
    familiarity: &Familiarity,
    grades: &[Option<Grade>],
) -> Result<()> {
    let n = corpus.post_count;
    ensure!(
        grades.len() == n
            && scores.is_none_or(|scores| {
                scores.not_dislike.len() == n
                    && scores.like_or_love.len() == n
                    && scores.love.len() == n
                    && scores.expected_grade.len() == n
            })
            && familiarity.post_rated_count.len() == n
            && familiarity.post_has_artist.len() == n,
        "Selection arrays are misaligned with the corpus."
    );
    Ok(())
}

#[derive(Clone, Copy)]
struct Ranked {
    row: usize,
    offset: usize,
    priority: f64,
    value: f32,
}
impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| other.row.cmp(&self.row))
            .then_with(|| other.offset.cmp(&self.offset))
    }
}

fn offer(heap: &mut BinaryHeap<Reverse<Ranked>>, item: Ranked, limit: usize) {
    if limit == 0 {
        return;
    }
    if heap.len() < limit {
        heap.push(Reverse(item));
    } else if heap.peek().is_some_and(|worst| item > worst.0) {
        *heap.peek_mut().unwrap() = Reverse(item);
    }
}

fn ignored_tags(corpus: &Corpus) -> Vec<bool> {
    let mut ignored = vec![false; corpus.tag_count];
    for &tag in corpus
        .artist_vocab_indices()
        .iter()
        .chain(corpus.ordinal_metadata_vocab_indices())
    {
        ignored[tag as usize] = true;
    }
    ignored
}

fn similarity(ignored: &[bool], left: &[i32], right: &[i32]) -> f64 {
    let left_count = left.iter().filter(|&&tag| !ignored[tag as usize]).count();
    let right_count = right.iter().filter(|&&tag| !ignored[tag as usize]).count();
    if left_count == 0 || right_count == 0 {
        return 0.0;
    }
    let (mut i, mut j, mut intersection) = (0, 0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                if !ignored[left[i] as usize] {
                    intersection += 1;
                }
                i += 1;
                j += 1;
            }
        }
    }
    intersection as f64 / ((left_count as f64) * (right_count as f64)).sqrt()
}

fn diverse_slate(
    corpus: &Corpus,
    pool: BinaryHeap<Reverse<Ranked>>,
    size: usize,
    issued_rows: &[usize],
    cancelled: &dyn Fn() -> bool,
) -> Vec<usize> {
    let mut pool: Vec<_> = pool.into_iter().map(|item| item.0).collect();
    let ignored = ignored_tags(corpus);
    let mut selected = Vec::with_capacity(size.min(pool.len()));
    // Cache each remaining row's maximum similarity: only the newly chosen row
    // can change it. This avoids rescanning every earlier pair on every pick.
    let mut maximum_similarity = vec![0.0_f64; pool.len()];
    for &row in issued_rows {
        if cancelled() {
            return Vec::new();
        }
        for (item, maximum) in pool.iter().zip(&mut maximum_similarity) {
            *maximum = maximum.max(similarity(
                &ignored,
                corpus.post_tags(item.row),
                corpus.post_tags(row),
            ));
        }
    }
    while selected.len() < size && !pool.is_empty() {
        if cancelled() {
            return Vec::new();
        }
        let mut best = 0;
        let mut best_value = f64::NEG_INFINITY;
        for (index, item) in pool.iter().enumerate() {
            let value = 0.3 * item.priority - 0.7 * maximum_similarity[index];
            if value > best_value || (value == best_value && item.row < pool[best].row) {
                best = index;
                best_value = value;
            }
        }
        let chosen = pool.swap_remove(best).row;
        maximum_similarity.swap_remove(best);
        selected.push(chosen);
        for (item, maximum) in pool.iter().zip(&mut maximum_similarity) {
            *maximum = maximum.max(similarity(
                &ignored,
                corpus.post_tags(item.row),
                corpus.post_tags(chosen),
            ));
        }
    }
    selected
}

fn starter_sample(
    corpus: &Corpus,
    candidates: &[usize],
    excluded: &HashSet<usize>,
    accept: &dyn Fn(usize) -> bool,
    seed: u64,
    limit: usize,
    cancelled: &dyn Fn() -> bool,
) -> Result<(Vec<(usize, f32)>, usize, usize)> {
    let mut pool = BinaryHeap::with_capacity(limit.min(candidates.len()));
    let mut seen = vec![0_u64; corpus.post_count.div_ceil(64)];
    let (mut eligible_count, mut admissible_count) = (0, 0);
    for (offset, &row) in candidates.iter().enumerate() {
        if offset % 4096 == 0 {
            ensure!(!cancelled(), "Selection cancelled");
        }
        let bit = 1 << (row % 64);
        if seen[row / 64] & bit != 0 {
            continue;
        }
        seen[row / 64] |= bit;
        if !accept(row) {
            continue;
        }
        admissible_count += 1;
        if excluded.contains(&row) {
            continue;
        }
        eligible_count += 1;
        // Corpus signed-log score is log(weight): exp(x) recovers 1 + score
        // for nonnegative scores, or 1 / (1 + abs(score)) for negatives.
        let log_weight = corpus.popularity(row);
        let bits = mix64(seed ^ (corpus.post_ids()[row] as u64).wrapping_mul(0xA0761D6478BD642F));
        // 52-bit midpoints stay strictly inside (0, 1), including both extremes.
        let uniform = ((bits >> 12) as f64 + 0.5) / 4503599627370496.0;
        offer(
            &mut pool,
            Ranked {
                row,
                offset: 0,
                // Log-space exponential race: exact weighted sampling without
                // replacement, without materializing potentially extreme weights.
                priority: log_weight - (-uniform.ln()).ln(),
                // Display utility is materialized only for the final sampled rows.
                value: 0.0,
            },
            limit,
        );
    }
    ensure!(!cancelled(), "Selection cancelled");
    let rows = pool
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse(item)| {
            let log_weight = corpus.popularity(item.row);
            let inverse = (-log_weight.abs()).exp();
            let utility = if log_weight >= 0.0 {
                1.0 / (1.0 + inverse)
            } else {
                inverse / (1.0 + inverse)
            };
            (item.row, utility as f32)
        })
        .collect();
    Ok((rows, eligible_count, admissible_count))
}

fn percentile_90(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return f32::INFINITY;
    }
    let position = 0.9 * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let (_, value, greater) = values.select_nth_unstable_by(lower, f32::total_cmp);
    let low = *value;
    if lower == upper {
        low
    } else {
        let high = greater.iter().copied().min_by(f32::total_cmp).unwrap();
        let fraction = (position - lower as f64) as f32;
        // Difference interpolation preserves equal-score ties exactly.
        low + (high - low) * fraction
    }
}

/// Rank the entire unpublished remainder of a Gallery cohort. Issued rows
/// consume capacity and remain diversity context even after leaving eligibility.
/// With no scores, the identity argument is the cohort's stable session seed and
/// returned values are popularity utilities, not personal grade probabilities.
pub fn gallery(
    corpus: &Corpus,
    scores: Option<&ScoreBuffers>,
    familiarity: &Familiarity,
    grades: &[Option<Grade>],
    facet: &str,
    cohort_size: usize,
    issued_rows: &[usize],
    candidates: &[usize],
    generation_or_starter_seed: &str,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<(usize, f32)>> {
    check_alignment(corpus, scores, familiarity, grades)?;
    ensure!(
        matches!(
            facet,
            "discover" | "develop" | "train_broadly" | "q3_probe" | "rating_review"
        ),
        "Unknown facet '{facet}'."
    );
    ensure!(
        candidates.iter().all(|&row| row < corpus.post_count),
        "Gallery candidate outside corpus."
    );
    ensure!(
        issued_rows.iter().all(|&row| row < corpus.post_count),
        "Gallery issued row outside corpus."
    );
    ensure!(
        scores.is_some() || !matches!(facet, "q3_probe" | "rating_review"),
        "This facet requires the supported Ensemble."
    );
    let size = cohort_size.saturating_sub(issued_rows.len());
    if size == 0 || candidates.is_empty() {
        return Ok(Vec::new());
    }
    let issued: HashSet<_> = issued_rows.iter().copied().collect();
    let Some(scores) = scores else {
        let mut identity = Sha256::new();
        identity.update(GALLERY_POLICY.as_bytes());
        identity.update(generation_or_starter_seed.as_bytes());
        let digest = identity.finalize();
        let seed = u64::from_le_bytes(digest[..8].try_into().unwrap());
        let (rows, _, _) = starter_sample(
            corpus,
            candidates,
            &issued,
            &|row| {
                facet != "develop"
                    || (familiarity.post_has_artist[row]
                        && (1..=2).contains(&familiarity.post_rated_count[row]))
            },
            seed,
            size,
            cancelled,
        )?;
        return Ok(rows);
    };
    let mut heap = BinaryHeap::new();
    let rows = match facet {
        "discover" | "develop" => {
            let mut used_artists: HashSet<_> = issued_rows
                .iter()
                .flat_map(|&row| corpus.post_artists(row).iter().copied())
                .collect();
            // An inferior row with exactly the same credits can never become
            // eligible after its better representative conflicts with a pick.
            // Borrow the CSR slices: no artist-set allocation per candidate.
            let mut best_by_artists: HashMap<&[i32], Ranked> = HashMap::new();
            for (offset, &row) in candidates.iter().enumerate() {
                if offset % 4096 == 0 {
                    ensure!(!cancelled(), "Selection cancelled");
                }
                if issued.contains(&row) {
                    continue;
                }
                let count = familiarity.post_rated_count[row];
                let eligible = familiarity.post_has_artist[row]
                    && if facet == "discover" {
                        count == 0
                    } else {
                        (1..=2).contains(&count)
                    };
                let artists = corpus.post_artists(row);
                if !eligible || artists.iter().any(|artist| used_artists.contains(artist)) {
                    continue;
                }
                let item = Ranked {
                    row,
                    offset: 0,
                    priority: scores.expected_grade[row] as f64,
                    value: 0.0,
                };
                let best = best_by_artists.entry(artists).or_insert(item);
                if item > *best {
                    *best = item;
                }
            }
            // Heap every distinct credit set, not a truncated global pool:
            // overlapping high-ranked rows must not hide available artists.
            let mut ranked: BinaryHeap<_> = best_by_artists.into_values().collect();
            let mut rows = Vec::with_capacity(size.min(ranked.len()));
            while rows.len() < size {
                let Some(item) = ranked.pop() else {
                    break;
                };
                ensure!(!cancelled(), "Selection cancelled");
                let artists = corpus.post_artists(item.row);
                if artists.iter().any(|artist| used_artists.contains(artist)) {
                    continue;
                }
                used_artists.extend(artists.iter().copied());
                rows.push(item.row);
            }
            rows
        }
        "train_broadly" => {
            let mut values: Vec<_> = candidates
                .iter()
                .filter(|&&row| !issued.contains(&row))
                .map(|&row| scores.expected_grade[row])
                .collect();
            let quality = percentile_90(&mut values);
            drop(values);
            let mut seen = vec![0_u64; corpus.post_count.div_ceil(64)];
            // Hash the fixed policy domain and model once. Each post then gets
            // an independent native draw using all 64 bits of its positive ID;
            // candidate ordering, duplicates and exclusions cannot advance it.
            let mut identity = Sha256::new();
            identity.update(TRAIN_BROADLY_SAMPLER_DOMAIN.as_bytes());
            identity.update(generation_or_starter_seed.as_bytes());
            let digest = identity.finalize();
            let base_seed = u64::from_le_bytes(digest[..8].try_into().unwrap());
            for (offset, &row) in candidates.iter().enumerate() {
                if offset % 4096 == 0 {
                    ensure!(!cancelled(), "Selection cancelled");
                }
                if issued.contains(&row) {
                    continue;
                }
                let bit = 1 << (row % 64);
                if scores.expected_grade[row] >= quality && seen[row / 64] & bit == 0 {
                    seen[row / 64] |= bit;
                    let priority =
                        fastrand::Rng::with_seed(base_seed ^ corpus.post_ids()[row] as u64).f64();
                    offer(
                        &mut heap,
                        Ranked {
                            row,
                            offset: 0,
                            priority,
                            value: 0.0,
                        },
                        5000,
                    );
                }
            }
            diverse_slate(corpus, heap, size, issued_rows, cancelled)
        }
        _ => {
            for (offset, &row) in candidates.iter().enumerate() {
                if offset % 4096 == 0 {
                    ensure!(!cancelled(), "Selection cancelled");
                }
                if issued.contains(&row) {
                    continue;
                }
                let priority = if facet == "q3_probe" {
                    1.0 - 2.0 * (scores.love[row] - 0.5).abs()
                } else {
                    grades[row].map_or(0.0, |grade| disagreement(scores, row, grade))
                };
                offer(
                    &mut heap,
                    Ranked {
                        row,
                        offset: 0,
                        priority: priority as f64,
                        value: 0.0,
                    },
                    5000,
                );
            }
            diverse_slate(corpus, heap, size, issued_rows, cancelled)
        }
    };
    Ok(rows
        .into_iter()
        .map(|row| {
            let value = if facet == "rating_review" {
                grades[row].map_or(0.0, |grade| disagreement(scores, row, grade))
            } else {
                scores.expected_grade[row]
            };
            (row, value)
        })
        .collect())
}

/// Without an Ensemble forecast, admission is based only on observed grades.
pub fn auto_eligible(preset: &str, grade: Option<Grade>, p: Option<f32>) -> bool {
    match grade {
        Some(Grade::Love | Grade::Like) => true,
        Some(Grade::Meh | Grade::Dislike) => false,
        None => match preset {
            "favorites" => p.is_none_or(|p| p >= 0.8),
            "balanced" => p.is_none_or(|p| p >= 0.7),
            "fresh" => p.is_none_or(|p| p >= 0.6),
            _ => false,
        },
    }
}

pub struct AutoPlan {
    pub rows: Vec<usize>,
    /// Ensemble ranking utility, or bounded popularity utility in Starter.
    pub scores: Vec<f32>,
    pub exhausted: bool,
}

fn mix64(value: u64) -> u64 {
    let mut mixed = value.wrapping_add(0x9E3779B97F4A7C15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D049BB133111EB);
    mixed ^ (mixed >> 31)
}

fn gumbel(seed: i32, epoch: u32, row: usize) -> f64 {
    let key = (seed as u32 as u64)
        ^ (epoch as u64).wrapping_mul(0xD6E8FEB86659FD93)
        ^ (row as u64).wrapping_mul(0xA0761D6478BD642F);
    let bits = mix64(key);
    let uniform = ((bits >> 11) as f64 + 0.5) / 9007199254740992.0;
    -(-uniform.ln()).ln()
}

fn base_quality(scores: &ScoreBuffers, grade: Option<Grade>, row: usize) -> f32 {
    let expected = (scores.expected_grade[row] / 3.0).clamp(0.0, 1.0);
    match grade {
        Some(Grade::Love) => 0.9 + 0.1 * expected,
        Some(Grade::Like) => 0.8 + 0.1 * expected,
        _ => expected,
    }
}

#[derive(Default)]
struct Portfolio {
    top: [f32; 3],
    count: u8,
}
impl Portfolio {
    fn add(&mut self, value: f32) {
        if self.count == 0 {
            self.top = [f32::NEG_INFINITY; 3];
        }
        self.count = (self.count + 1).min(3);
        if value >= self.top[0] {
            self.top[2] = self.top[1];
            self.top[1] = self.top[0];
            self.top[0] = value;
        } else if value >= self.top[1] {
            self.top[2] = self.top[1];
            self.top[1] = value;
        } else if value > self.top[2] {
            self.top[2] = value;
        }
    }
    fn mean(&self) -> f32 {
        match self.count {
            0 => 0.0,
            1 => self.top[0],
            2 => (self.top[0] + self.top[1]) / 2.0,
            _ => (self.top[0] + self.top[1] + self.top[2]) / 3.0,
        }
    }
}

pub fn auto_page(
    corpus: &Corpus,
    scores: Option<&ScoreBuffers>,
    familiarity: &Familiarity,
    grades: &[Option<Grade>],
    candidates: &[usize],
    emitted: &HashSet<usize>,
    recent_rows: &[usize],
    preset: &str,
    size: usize,
    seed: i32,
    epoch: u32,
    accept: &dyn Fn(usize) -> bool,
    cancelled: &dyn Fn() -> bool,
) -> Result<AutoPlan> {
    check_alignment(corpus, scores, familiarity, grades)?;
    ensure!(size > 0, "Auto page size must be positive.");
    ensure!(
        candidates
            .iter()
            .chain(recent_rows)
            .all(|&row| row < corpus.post_count),
        "Auto row outside corpus."
    );
    let (temperature, artist_quota, recent_artist_penalty) = match preset {
        "favorites" => (0.01, usize::MAX, 0.0),
        "balanced" => (0.025, 1, 0.012),
        "fresh" => (0.04, 1, 0.028),
        _ => bail!("Unknown Auto selection '{preset}'."),
    };
    let Some(scores) = scores else {
        let seed = (seed as u32 as u64) ^ (epoch as u64).wrapping_mul(0xD6E8FEB86659FD93);
        let (selected, eligible_count, admissible_count) = starter_sample(
            corpus,
            candidates,
            emitted,
            &|row| {
                accept(row)
                    && auto_eligible(preset, grades[row], None)
                    && (preset != "fresh"
                        || (grades[row].is_none()
                            && familiarity.post_has_artist[row]
                            && familiarity.post_rated_count[row] == 0))
            },
            seed,
            size,
            cancelled,
        )?;
        let requested = size.min(eligible_count);
        let (rows, scores) = selected.into_iter().unzip();
        return Ok(AutoPlan {
            rows,
            scores,
            exhausted: admissible_count == requested,
        });
    };
    let has_artist = |row| !corpus.post_artists(row).is_empty();
    let admissible = |row: usize| {
        accept(row)
            && auto_eligible(preset, grades[row], Some(scores.like_or_love[row]))
            && (preset != "fresh"
                || (grades[row].is_none()
                    && has_artist(row)
                    && familiarity.post_rated_count[row] == 0))
    };
    let mut eligible_count = 0;
    let mut admissible_count = 0;
    let mut portfolios = if preset == "fresh" {
        (0..corpus.artist_count)
            .map(|_| Portfolio::default())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    for (offset, &row) in candidates.iter().enumerate() {
        if offset % 4096 == 0 {
            ensure!(!cancelled(), "Selection cancelled");
        }
        if admissible(row) {
            admissible_count += 1;
        } else {
            continue;
        }
        if !emitted.contains(&row) {
            eligible_count += 1;
            if preset == "fresh" {
                for &artist in corpus.post_artists(row) {
                    portfolios[artist as usize].add(scores.expected_grade[row]);
                }
            }
        }
    }
    let requested = size.min(eligible_count);
    let mut plan = AutoPlan {
        rows: Vec::with_capacity(requested),
        scores: Vec::with_capacity(requested),
        // Previously emitted rows can replay in a later epoch unless retained.
        exhausted: admissible_count == requested,
    };
    if requested == 0 {
        return Ok(plan);
    }
    let pool_size = eligible_count.min(512.max(requested.saturating_mul(20)));
    let mut heap = BinaryHeap::with_capacity(pool_size);
    for (offset, &row) in candidates.iter().enumerate() {
        if offset % 4096 == 0 {
            ensure!(!cancelled(), "Selection cancelled");
        }
        if emitted.contains(&row) || !admissible(row) {
            continue;
        }
        let value = if preset == "fresh" {
            let artists = corpus.post_artists(row);
            let portfolio = artists
                .iter()
                .map(|&artist| portfolios[artist as usize].mean() as f64)
                .sum::<f64>()
                / artists.len() as f64;
            ((0.5 * scores.expected_grade[row] as f64 + 0.5 * portfolio) / 3.0).clamp(0.0, 1.0)
                as f32
        } else {
            base_quality(scores, grades[row], row)
        };
        offer(
            &mut heap,
            Ranked {
                row,
                offset,
                value,
                priority: value as f64 + temperature * gumbel(seed, epoch, row),
            },
            pool_size,
        );
    }
    drop(portfolios);
    let mut pool: Vec<_> = heap.into_iter().map(|item| item.0).collect();
    let mut artist_counts: HashMap<i32, usize> = HashMap::new();
    let mut recent_artist_counts: HashMap<i32, usize> = HashMap::new();
    let ignored = if preset != "favorites" {
        ignored_tags(corpus)
    } else {
        Vec::new()
    };
    let mut tag_counts = if preset != "favorites" {
        vec![0.0_f32; corpus.tag_count]
    } else {
        Vec::new()
    };
    if preset != "favorites" {
        for &row in recent_rows {
            for &artist in corpus.post_artists(row) {
                *recent_artist_counts.entry(artist).or_default() += 1;
            }
        }
        for &row in recent_rows.iter().take(200) {
            for &tag in corpus.post_tags(row) {
                if !ignored[tag as usize] {
                    tag_counts[tag as usize] += 0.05;
                }
            }
        }
    }
    while plan.rows.len() < requested {
        ensure!(!cancelled(), "Selection cancelled");
        let mut best = None;
        let mut best_value = f64::NEG_INFINITY;
        for enforce_quota in [true, false] {
            for (index, item) in pool.iter().enumerate() {
                let artists = corpus.post_artists(item.row);
                if enforce_quota
                    && preset != "favorites"
                    && artists.iter().any(|artist| {
                        artist_counts.get(artist).copied().unwrap_or(0) >= artist_quota
                    })
                {
                    continue;
                }
                let mut adjusted = item.priority;
                if preset != "favorites" {
                    if !artists.is_empty() {
                        let count: usize = artists
                            .iter()
                            .map(|artist| artist_counts.get(artist).copied().unwrap_or(0))
                            .sum();
                        let recent: usize = artists
                            .iter()
                            .map(|artist| recent_artist_counts.get(artist).copied().unwrap_or(0))
                            .sum();
                        let adjacent = plan.rows.last().map_or(0, |&row| {
                            artists
                                .iter()
                                .filter(|artist| {
                                    corpus.post_artists(row).binary_search(artist).is_ok()
                                })
                                .count()
                        });
                        adjusted -= (0.06 * count as f64
                            + recent_artist_penalty * recent as f64
                            + 0.24 * adjacent as f64)
                            / artists.len() as f64;
                    }
                    let (mut repetition, mut content_count) = (0.0_f32, 0);
                    for &tag in corpus.post_tags(item.row) {
                        if !ignored[tag as usize] {
                            repetition += tag_counts[tag as usize];
                            content_count += 1;
                        }
                    }
                    if content_count > 0 {
                        adjusted -= 0.02 * repetition as f64 / content_count as f64;
                    }
                }
                if adjusted > best_value
                    || (adjusted == best_value
                        && best.is_none_or(|i: usize| item.row < pool[i].row))
                {
                    best = Some(index);
                    best_value = adjusted;
                }
            }
            if best.is_some() {
                break;
            }
        }
        let index =
            best.ok_or_else(|| anyhow::anyhow!("Auto candidate pool was exhausted early."))?;
        let item = pool.swap_remove(index);
        plan.rows.push(item.row);
        plan.scores.push(item.value);
        if preset != "favorites" {
            for &artist in corpus.post_artists(item.row) {
                *artist_counts.entry(artist).or_default() += 1;
            }
            for &tag in corpus.post_tags(item.row) {
                if !ignored[tag as usize] {
                    tag_counts[tag as usize] += 1.0;
                }
            }
        }
    }
    Ok(plan)
}

#[cfg(test)]
#[path = "../tests/unit/selection.rs"]
mod tests;

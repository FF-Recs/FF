use super::decoder::{logistic_loss, sigmoid};
use super::recipe::{Lbfgs, NumericalRecipe};
use anyhow::{Result, bail, ensure};

/// Only labeled rows are compacted; the immutable full-corpus CSR remains mmap-backed.
#[derive(Debug)]
pub(crate) struct SparseRows {
    pub offsets: Vec<usize>,
    pub columns: Vec<usize>,
}
impl SparseRows {
    pub fn row(&self, index: usize) -> &[usize] {
        &self.columns[self.offsets[index]..self.offsets[index + 1]]
    }
}

pub(crate) struct Objective<'a> {
    pub rows: &'a SparseRows,
    pub indices: &'a [usize],
    pub labels: &'a [u8],
    pub scales: &'a [f64],
    pub sample_weights: &'a [f64],
    pub c: f64,
    pub penalize_intercept: bool,
    /// Standardized popularity for labeled rows, separate from tag incidence.
    pub numeric: &'a [f64],
}
impl Objective<'_> {
    pub fn evaluate(&self, weights: &[f64], gradient: &mut [f64]) -> f64 {
        let dimensions = weights.len();
        let regularized = dimensions - usize::from(!self.penalize_intercept);
        gradient.fill(0.0);
        gradient[..regularized].copy_from_slice(&weights[..regularized]);
        let mut loss = weights[..regularized]
            .iter()
            .map(|w| 0.5 * w * w)
            .sum::<f64>();
        for (i, &row) in self.indices.iter().enumerate() {
            let mut score = weights[dimensions - 1];
            for &column in self.rows.row(row) {
                score += weights[column] * self.scales[column];
            }
            score += weights[dimensions - 2] * self.numeric[row];
            let multiplier = self.c * self.sample_weights[i];
            let positive = self.labels[i] == 1;
            loss += multiplier * logistic_loss(positive, score);
            // This algebraically identical residual avoids 1 - sigmoid rounding
            // to zero for well-predicted positive saturated rows.
            let residual = multiplier
                * if positive {
                    -sigmoid(-score)
                } else {
                    sigmoid(score)
                };
            for &column in self.rows.row(row) {
                gradient[column] += residual * self.scales[column];
            }
            gradient[dimensions - 2] += residual * self.numeric[row];
            gradient[dimensions - 1] += residual;
        }
        loss
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x.abs()).fold(0.0, f64::max)
}
fn finite(loss: f64, gradient: &[f64], weights: &[f64]) -> bool {
    loss.is_finite() && gradient.iter().chain(weights).all(|x| x.is_finite())
}
struct History {
    s: Vec<f64>,
    y: Vec<f64>,
    rho: f64,
}

/// L-BFGS with ten reusable correction pairs and two reusable evaluation
/// gradients. Every exit is certified against the same objective; a supported
/// fit cannot silently fall back after failed optimization.
pub(crate) fn minimize(
    mut weights: Vec<f64>,
    tolerance: f64,
    max_iterations: usize,
    mut evaluate: impl FnMut(&[f64], &mut [f64]) -> f64,
) -> Result<Vec<f64>> {
    let dimensions = weights.len();
    let mut gradient = vec![0.0; dimensions];
    let mut work_gradient = vec![0.0; dimensions];
    let mut direction = vec![0.0; dimensions];
    let mut candidate = vec![0.0; dimensions];
    let mut history = (0..Lbfgs::HISTORY)
        .map(|_| History {
            s: vec![0.0; dimensions],
            y: vec![0.0; dimensions],
            rho: 0.0,
        })
        .collect::<Vec<_>>();
    let mut history_start = 0;
    let mut history_len = 0;
    let mut loss = evaluate(&weights, &mut gradient);
    ensure!(
        finite(loss, &gradient, &weights),
        "Nonfinite initial logistic objective"
    );
    for iteration in 0..max_iterations {
        if norm(&gradient) <= tolerance * loss.abs().max(1.0) {
            return Ok(weights);
        }
        direction.copy_from_slice(&gradient);
        let mut alphas = [0.0; Lbfgs::HISTORY];
        for i in (0..history_len).rev() {
            let h = &history[(history_start + i) % Lbfgs::HISTORY];
            alphas[i] = h.rho * dot(&h.s, &direction);
            for c in 0..dimensions {
                direction[c] -= alphas[i] * h.y[c];
            }
        }
        if history_len > 0 {
            let h = &history[(history_start + history_len - 1) % Lbfgs::HISTORY];
            let denominator = dot(&h.y, &h.y);
            let gamma = if denominator > 0.0 {
                dot(&h.s, &h.y) / denominator
            } else {
                1.0
            };
            for value in &mut direction {
                *value *= gamma;
            }
        }
        for i in 0..history_len {
            let h = &history[(history_start + i) % Lbfgs::HISTORY];
            let factor = alphas[i] - h.rho * dot(&h.y, &direction);
            for c in 0..dimensions {
                direction[c] += factor * h.s[c];
            }
        }
        for value in &mut direction {
            *value = -*value;
        }
        let mut slope = dot(&gradient, &direction);
        if !slope.is_finite() || slope >= 0.0 {
            for c in 0..dimensions {
                direction[c] = -gradient[c];
            }
            slope = -dot(&gradient, &gradient);
        }
        let mut next_loss = loss;
        let mut accepted = false;
        // Retry a failed quasi-Newton direction with steepest descent, retaining
        // the objective and Armijo condition, not a solver-failure fallback model.
        for attempt in 0..2 {
            if attempt == 1 {
                for c in 0..dimensions {
                    direction[c] = -gradient[c];
                }
                slope = -dot(&gradient, &gradient);
                history_len = 0;
                history_start = 0;
            }
            let mut step = 1.0;
            for _ in 0..NumericalRecipe::BACKTRACKS {
                for c in 0..dimensions {
                    candidate[c] = weights[c] + step * direction[c];
                }
                next_loss = evaluate(&candidate, &mut work_gradient);
                if finite(next_loss, &work_gradient, &candidate)
                    && slope < 0.0
                    && next_loss <= loss + NumericalRecipe::ARMIJO * step * slope
                {
                    accepted = true;
                    break;
                }
                step *= NumericalRecipe::SHRINK;
            }
            if accepted {
                break;
            }
        }
        if !accepted {
            // Re-evaluate rather than trusting stale scratch or an ftol exit.
            loss = evaluate(&weights, &mut gradient);
            ensure!(
                finite(loss, &gradient, &weights),
                "Nonfinite failed-search logistic objective"
            );
            if norm(&gradient) <= tolerance * loss.abs().max(1.0) {
                return Ok(weights);
            }
            bail!(
                "L-BFGS line search failed at iteration {iteration}: objective {loss}, maximum gradient {}, certificate {}",
                norm(&gradient),
                tolerance * loss.abs().max(1.0)
            );
        }
        let slot = if history_len == Lbfgs::HISTORY {
            history_start
        } else {
            (history_start + history_len) % Lbfgs::HISTORY
        };
        let h = &mut history[slot];
        for c in 0..dimensions {
            h.s[c] = candidate[c] - weights[c];
            h.y[c] = work_gradient[c] - gradient[c];
        }
        let curvature = dot(&h.s, &h.y);
        if curvature > NumericalRecipe::MIN_CURVATURE && curvature.is_finite() {
            h.rho = 1.0 / curvature;
            if history_len == Lbfgs::HISTORY {
                history_start = (history_start + 1) % Lbfgs::HISTORY;
            } else {
                history_len += 1;
            }
        } else if history_len == Lbfgs::HISTORY {
            // The oldest pair was overwritten, so exclude it when curvature is
            // unusable. Never reuse a pair with a stale inverse curvature.
            history_start = (history_start + 1) % Lbfgs::HISTORY;
            history_len -= 1;
        }
        std::mem::swap(&mut weights, &mut candidate);
        std::mem::swap(&mut gradient, &mut work_gradient);
        loss = next_loss;
    }
    loss = evaluate(&weights, &mut gradient);
    ensure!(
        finite(loss, &gradient, &weights) && norm(&gradient) <= tolerance * loss.abs().max(1.0),
        "L-BFGS iteration limit without convergence: objective {loss}, maximum gradient {}, certificate {}",
        norm(&gradient),
        tolerance * loss.abs().max(1.0)
    );
    Ok(weights)
}

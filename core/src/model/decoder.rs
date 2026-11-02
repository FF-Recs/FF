use super::recipe::{ConditionalDecoder, NUMERICAL, NumericalRecipe};
use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

pub type RawLogits = [f64; 3];

pub(crate) fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}
pub(crate) fn log_sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        -(-z).exp().ln_1p()
    } else {
        z - z.exp().ln_1p()
    }
}
pub(crate) fn logistic_loss(positive: bool, z: f64) -> f64 {
    -log_sigmoid(if positive { z } else { -z })
}
fn affine(theta: &[f64; 4], raw: RawLogits) -> f64 {
    theta[0] + theta[1] * raw[0] + theta[2] * raw[1] + theta[3] * raw[2]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderArtifact {
    pub version: u32,
    pub parameters: [[f64; 4]; 3],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageDiagnostics {
    pub risk_rows: usize,
    pub negative_outcomes: usize,
    pub positive_outcomes: usize,
    pub iterations: usize,
    pub objective: f64,
    pub projected_gradient_maximum: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Forecast {
    pub cumulative: [f64; 3],
    pub masses: [f64; 4],
    pub log_masses: [f64; 4],
}

/// Decoder executable state is constructed only after import validation or fit.
pub(super) struct Decoder {
    parameters: [[f64; 4]; 3],
}
impl Decoder {
    pub fn from_artifact(dto: DecoderArtifact) -> Self {
        Self {
            parameters: dto.parameters,
        }
    }
    pub fn export(&self) -> DecoderArtifact {
        DecoderArtifact {
            version: 1,
            parameters: self.parameters,
        }
    }
    pub fn forecast(&self, raw: RawLogits) -> Result<Forecast> {
        forecast(&self.parameters, raw)
    }
    pub fn cumulative(&self, raw: RawLogits) -> Result<[f64; 3]> {
        Ok(cumulative_from_logs(
            checked_affine(&self.parameters, raw)?.map(log_sigmoid),
        ))
    }
}
fn checked_affine(parameters: &[[f64; 4]; 3], raw: RawLogits) -> Result<[f64; 3]> {
    ensure!(raw.iter().all(|x| x.is_finite()), "Nonfinite raw logits");
    let z = parameters.map(|p| affine(&p, raw));
    ensure!(
        z.iter().all(|x| x.is_finite()),
        "Nonfinite decoder affine score"
    );
    Ok(z)
}
fn forecast(parameters: &[[f64; 4]; 3], raw: RawLogits) -> Result<Forecast> {
    let z = checked_affine(parameters, raw)?;
    let positive = z.map(log_sigmoid);
    let negative = z.map(|x| log_sigmoid(-x));
    let log_masses = [
        negative[0],
        positive[0] + negative[1],
        positive[0] + positive[1] + negative[2],
        positive[0] + positive[1] + positive[2],
    ];
    Ok(Forecast {
        cumulative: cumulative_from_logs(positive),
        masses: log_masses.map(f64::exp),
        log_masses,
    })
}

impl DecoderArtifact {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.version == 1, "Unsupported decoder artifact version");
        for (head, theta) in self.parameters.iter().enumerate() {
            ensure!(
                theta.iter().all(|x| x.is_finite()) && theta[head + 1] >= 0.0,
                "Invalid conditional decoder parameters"
            );
        }
        Ok(())
    }
    pub fn forecast(&self, raw: RawLogits) -> Result<Forecast> {
        self.validate()?;
        forecast(&self.parameters, raw)
    }
    pub fn fit(raw: &[RawLogits], grades: &[u8]) -> Result<(Self, [StageDiagnostics; 3])> {
        Self::fit_with_numerical(raw, grades, &NUMERICAL)
    }
    pub(super) fn fit_with_numerical(
        raw: &[RawLogits],
        grades: &[u8],
        numerical: &NumericalRecipe,
    ) -> Result<(Self, [StageDiagnostics; 3])> {
        ensure!(
            raw.len() == grades.len()
                && grades.iter().all(|&g| g <= 3)
                && raw.iter().flatten().all(|v| v.is_finite()),
            "Expected aligned finite logits and grades in [0,3]"
        );
        let stages = (0..3)
            .map(|head| fit_stage(head, raw, grades, numerical))
            .collect::<Result<Vec<_>>>()?;
        let parameters = std::array::from_fn(|h| stages[h].0);
        let diagnostics = std::array::from_fn(|h| stages[h].1.clone());
        Ok((
            Self {
                version: 1,
                parameters,
            },
            diagnostics,
        ))
    }
}

fn cumulative_from_logs([a, b, c]: [f64; 3]) -> [f64; 3] {
    [a.exp(), (a + b).exp(), (a + b + c).exp()]
}

pub fn mixture(components: &[Forecast; super::recipe::PartitionRecipe::MEMBERS]) -> Forecast {
    let count = super::recipe::PartitionRecipe::MEMBERS as f64;
    let cumulative =
        std::array::from_fn(|h| components.iter().map(|f| f.cumulative[h]).sum::<f64>() / count);
    let log_masses = std::array::from_fn(|g| {
        let maximum = components
            .iter()
            .map(|f| f.log_masses[g])
            .fold(f64::NEG_INFINITY, f64::max);
        if maximum == f64::NEG_INFINITY {
            maximum
        } else {
            maximum
                + components
                    .iter()
                    .map(|f| (f.log_masses[g] - maximum).exp())
                    .sum::<f64>()
                    .ln()
                - count.ln()
        }
    });
    Forecast {
        cumulative,
        masses: log_masses.map(f64::exp),
        log_masses,
    }
}

pub(crate) fn evaluate(
    head: usize,
    raw: &[RawLogits],
    grades: &[u8],
    theta: &[f64; 4],
) -> Result<(f64, [f64; 4], [[f64; 4]; 4])> {
    ensure!(
        head < 3 && raw.len() == grades.len() && theta.iter().all(|x| x.is_finite()),
        "Invalid conditional objective dimensions"
    );
    let mut objective = 0.0;
    let mut gradient = [0.0; 4];
    let mut hessian = [[0.0; 4]; 4];
    for column in 0..4 {
        let own = column == head + 1;
        let ridge = if column == 0 || own {
            ConditionalDecoder::OWN_RIDGE
        } else {
            ConditionalDecoder::CROSS_RIDGE
        };
        let delta = theta[column]
            - if own {
                ConditionalDecoder::OWN_CENTER
            } else {
                0.0
            };
        objective += 0.5 * ridge * delta * delta;
        gradient[column] = ridge * delta;
        hessian[column][column] = ridge;
    }
    for (&value, &grade) in raw.iter().zip(grades) {
        if grade as usize >= head {
            let x = [1.0, value[0], value[1], value[2]];
            let z = affine(theta, value);
            let positive = grade as usize > head;
            objective += logistic_loss(positive, z);
            let residual = if positive { -sigmoid(-z) } else { sigmoid(z) };
            let curvature = sigmoid(z) * sigmoid(-z);
            for a in 0..4 {
                gradient[a] += residual * x[a];
                for b in 0..4 {
                    hessian[a][b] += curvature * x[a] * x[b];
                }
            }
        }
    }
    ensure!(
        objective.is_finite()
            && gradient.iter().all(|x| x.is_finite())
            && hessian.iter().flatten().all(|x| x.is_finite()),
        "Nonfinite decoder objective"
    );
    Ok((objective, gradient, hessian))
}

pub(crate) fn objective_change(
    head: usize,
    raw: &[RawLogits],
    grades: &[u8],
    theta: &[f64; 4],
    candidate: &[f64; 4],
) -> f64 {
    let delta = std::array::from_fn(|c| candidate[c] - theta[c]);
    let mut change = 0.0;
    for c in 0..4 {
        let own = c == head + 1;
        let ridge = if c == 0 || own {
            ConditionalDecoder::OWN_RIDGE
        } else {
            ConditionalDecoder::CROSS_RIDGE
        };
        let prior = if own {
            ConditionalDecoder::OWN_CENTER
        } else {
            0.0
        };
        change += 0.5 * ridge * delta[c] * ((candidate[c] - prior) + (theta[c] - prior));
    }
    for (&value, &grade) in raw.iter().zip(grades) {
        if grade as usize >= head {
            let sign = if grade as usize > head { 1.0 } else { -1.0 };
            let margin = sign * affine(theta, value);
            let d = sign * affine(&delta, value);
            change += if d.abs() <= NumericalRecipe::STABLE_DELTA {
                (sigmoid(-margin) * (-d).exp_m1()).ln_1p()
            } else {
                log_sigmoid(margin) - log_sigmoid(sign * affine(candidate, value))
            };
        }
    }
    change
}
pub(crate) fn projected_maximum(own: usize, theta: &[f64; 4], gradient: &[f64; 4]) -> f64 {
    (0..4)
        .map(|c| {
            if c == own && theta[c] == 0.0 {
                gradient[c].min(0.0).abs()
            } else {
                gradient[c].abs()
            }
        })
        .fold(0.0, f64::max)
}
fn solve(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Result<[f64; 4]> {
    for c in 0..4 {
        let pivot = (c..4)
            .max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs()))
            .unwrap();
        ensure!(
            a[pivot][c].is_finite() && a[pivot][c].abs() >= NumericalRecipe::MIN_PIVOT,
            "Singular decoder Hessian"
        );
        a.swap(c, pivot);
        b.swap(c, pivot);
        for row in c + 1..4 {
            let multiplier = a[row][c] / a[c][c];
            for k in c..4 {
                a[row][k] -= multiplier * a[c][k];
            }
            b[row] -= multiplier * b[c];
        }
    }
    let mut result = [0.0; 4];
    for row in (0..4).rev() {
        let mut value = b[row];
        for c in row + 1..4 {
            value -= a[row][c] * result[c];
        }
        result[row] = value / a[row][row];
    }
    ensure!(
        result.iter().all(|x| x.is_finite()),
        "Nonfinite Newton direction"
    );
    Ok(result)
}
fn fit_stage(
    head: usize,
    raw: &[RawLogits],
    grades: &[u8],
    numerical: &NumericalRecipe,
) -> Result<([f64; 4], StageDiagnostics)> {
    let own = head + 1;
    let mut theta = [0.0; 4];
    theta[own] = ConditionalDecoder::OWN_CENTER;
    let mut iterations = 0;
    let mut converged = false;
    while !converged && iterations < numerical.decoder.iterations {
        let (_, gradient, mut hessian) = evaluate(head, raw, grades, &theta)?;
        if projected_maximum(own, &theta, &gradient) <= numerical.decoder.tolerance {
            converged = true;
            break;
        }
        let mut rhs = gradient;
        if theta[own] == 0.0 && gradient[own] >= 0.0 {
            for c in 0..4 {
                hessian[own][c] = 0.0;
                hessian[c][own] = 0.0;
            }
            hessian[own][own] = 1.0;
            rhs[own] = 0.0;
        }
        let direction = solve(hessian, rhs)?;
        let candidate_at = |step: f64| {
            std::array::from_fn(|c| {
                let updated = theta[c] - step * direction[c];
                if c == own { updated.max(0.0) } else { updated }
            })
        };
        let mut step = 1.0;
        let mut accepted = None;
        while step >= NumericalRecipe::MIN_STEP {
            let candidate = candidate_at(step);
            let slope: f64 = (0..4)
                .map(|c| gradient[c] * (candidate[c] - theta[c]))
                .sum();
            // Keep the existing finite objective/derivative checks on each trial,
            // but do not subtract rounded totals to decide a near-optimum step.
            evaluate(head, raw, grades, &candidate)?;
            let change = objective_change(head, raw, grades, &theta, &candidate);
            if slope < 0.0 && change.is_finite() && change <= NumericalRecipe::ARMIJO * slope {
                accepted = Some(candidate);
                break;
            }
            step *= NumericalRecipe::SHRINK;
        }
        if let Some(candidate) = accepted {
            theta = candidate;
        } else if projected_maximum(own, &theta, &gradient) <= numerical.decoder.certificate {
            converged = true;
        } else {
            let candidate = candidate_at(1.0);
            let (_, candidate_gradient, _) = evaluate(head, raw, grades, &candidate)?;
            let slope: f64 = (0..4)
                .map(|c| gradient[c] * (candidate[c] - theta[c]))
                .sum();
            let change = objective_change(head, raw, grades, &theta, &candidate);
            if slope < 0.0
                && change.is_finite()
                && change <= NumericalRecipe::ARMIJO * slope
                && projected_maximum(own, &candidate, &candidate_gradient)
                    <= numerical.decoder.tolerance
            {
                theta = candidate;
                converged = true;
            } else {
                bail!(
                    "Conditional decoder line search failed (head {head}, projected gradient {})",
                    projected_maximum(own, &theta, &gradient)
                );
            }
        }
        iterations += 1;
    }
    let (objective, gradient, _) = evaluate(head, raw, grades, &theta)?;
    let norm = projected_maximum(own, &theta, &gradient);
    ensure!(
        converged && norm <= numerical.decoder.certificate,
        "Conditional decoder failed convergence certificate (head {head}, gradient {norm})"
    );
    let negative_outcomes = grades.iter().filter(|&&g| g as usize == head).count();
    let positive_outcomes = grades.iter().filter(|&&g| g as usize > head).count();
    Ok((
        theta,
        StageDiagnostics {
            risk_rows: negative_outcomes + positive_outcomes,
            negative_outcomes,
            positive_outcomes,
            iterations,
            objective,
            projected_gradient_maximum: norm,
        },
    ))
}

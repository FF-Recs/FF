use super::decoder::{evaluate, objective_change, projected_maximum};
use super::sparse::{Objective, SparseRows, minimize};
use super::*;
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("../../fixtures/conditional-decoder.json")).unwrap()
}
fn raw(value: &Value) -> Vec<RawLogits> {
    serde_json::from_value(value.clone()).unwrap()
}
fn grades(value: &Value) -> Vec<u8> {
    serde_json::from_value(value.clone()).unwrap()
}
fn near(tolerance: f64, expected: f64, actual: f64) {
    assert!(
        (expected - actual).abs() <= tolerance,
        "expected {expected:.17e}, actual {actual:.17e}, tolerance {tolerance}"
    );
}
#[test]
fn independent_all_objectives_gradients_and_hessians() {
    let f = fixture();
    for head in 0..3 {
        let mut total = 0.0;
        for member in 0..5 {
            let x = raw(&f["raw"][member]);
            let y = grades(&f["grades"][member]);
            let theta: [f64; 4] =
                serde_json::from_value(f["oracle_theta"][member].clone()).unwrap();
            let (objective, gradient, hessian) = evaluate(head, &x, &y, &theta).unwrap();
            total += objective;
            for c in 0..4 {
                near(
                    1e-10,
                    f["objective"][head]["gradient"][member * 4 + c]
                        .as_f64()
                        .unwrap(),
                    gradient[c],
                );
                let mut plus = theta;
                let mut minus = theta;
                plus[c] += 1e-5;
                minus[c] -= 1e-5;
                let (a, ga, _) = evaluate(head, &x, &y, &plus).unwrap();
                let (b, gb, _) = evaluate(head, &x, &y, &minus).unwrap();
                near(1e-7, gradient[c], (a - b) / 2e-5);
                for row in 0..4 {
                    near(1e-7, hessian[row][c], (ga[row] - gb[row]) / 2e-5);
                }
            }
        }
        near(
            1e-10,
            f["objective"][head]["value"].as_f64().unwrap(),
            total,
        );
    }
}
#[test]
fn independent_all_constrained_fit_matches_scipy_reference() {
    let f = fixture();
    for member in 0..5 {
        let (model, diagnostics) =
            DecoderArtifact::fit(&raw(&f["raw"][member]), &grades(&f["grades"][member])).unwrap();
        for head in 0..3 {
            assert!(diagnostics[head].projected_gradient_maximum <= 1e-7);
            assert!(model.parameters[head][head + 1] >= 0.0);
            for c in 0..4 {
                near(
                    2e-5,
                    f["parameters"][member][head][c].as_f64().unwrap(),
                    model.parameters[head][c],
                );
            }
        }
    }
}
#[test]
fn independent_forecasts_and_log_probability_mixtures_survive_serialization() {
    let f = fixture();
    let predictions = (0..5)
        .map(|member| {
            let model = DecoderArtifact {
                version: 1,
                parameters: serde_json::from_value(f["parameters"][member].clone()).unwrap(),
            };
            let model: DecoderArtifact =
                serde_json::from_slice(&serde_json::to_vec(&model).unwrap()).unwrap();
            raw(&f["predict_raw"][member])
                .into_iter()
                .enumerate()
                .map(|(row, x)| {
                    let p = model.forecast(x).unwrap();
                    let expected = &f["forecasts"][member];
                    for h in 0..3 {
                        let q = expected["q"][row][h].as_f64().unwrap();
                        near(1e-12, q, p.cumulative[h]);
                        assert_eq!(q as f32, p.cumulative[h] as f32);
                    }
                    for g in 0..4 {
                        near(1e-12, expected["p"][row][g].as_f64().unwrap(), p.masses[g]);
                        near(
                            1e-9,
                            expected["log_p"][row][g].as_f64().unwrap(),
                            p.log_masses[g],
                        );
                    }
                    near(1e-12, 1.0, p.masses.iter().sum());
                    p
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for row in 0..predictions[0].len() {
        let p = mixture(&std::array::from_fn(|m| predictions[m][row].clone()));
        for h in 0..3 {
            near(
                1e-12,
                f["mixture"]["q"][row][h].as_f64().unwrap(),
                p.cumulative[h],
            );
        }
        for g in 0..4 {
            near(
                1e-9,
                f["mixture"]["log_p"][row][g].as_f64().unwrap(),
                p.log_masses[g],
            );
        }
    }
}
#[test]
fn decoder_cancellation_regressions_retain_strict_gradient_certificate() {
    for (count, scale) in [(400, 2.0), (800, 0.5), (800, 2.0)] {
        let y = (0..count).map(|i| (i % 4) as u8).collect::<Vec<_>>();
        let x = y
            .iter()
            .map(|&g| {
                [
                    scale * (g as f64 - 0.5),
                    scale * (g as f64 - 1.5),
                    scale * (g as f64 - 2.5),
                ]
            })
            .collect::<Vec<_>>();
        let (model, diagnostics) = DecoderArtifact::fit(&x, &y).unwrap();
        for h in 0..3 {
            assert!(diagnostics[h].projected_gradient_maximum <= 1e-7);
            let (_, gradient, _) = evaluate(h, &x, &y, &model.parameters[h]).unwrap();
            assert!(projected_maximum(h + 1, &model.parameters[h], &gradient) <= 1e-7);
        }
        for &row in &x {
            let forecast = model.forecast(row).unwrap();
            assert!(forecast.log_masses.iter().all(|x| x.is_finite()));
            near(1e-12, 1.0, forecast.masses.iter().sum());
        }
    }
}
#[test]
fn stable_objective_changes_match_taylor_expansion_and_extreme_tails() {
    let f = fixture();
    let x = raw(&f["raw"][0]);
    let y = grades(&f["grades"][0]);
    let theta: [f64; 4] = serde_json::from_value(f["oracle_theta"][0].clone()).unwrap();
    for h in 0..3 {
        let (value, gradient, hessian) = evaluate(h, &x, &y, &theta).unwrap();
        let candidate = std::array::from_fn(|c| theta[c] + (c as f64 - 2.0) * 0.7);
        let (next, _, _) = evaluate(h, &x, &y, &candidate).unwrap();
        near(
            1e-10,
            next - value,
            objective_change(h, &x, &y, &theta, &candidate),
        );
        let tiny = std::array::from_fn(|c| theta[c] + (c as f64 - 2.0) * 1e-10);
        let delta: [f64; 4] = std::array::from_fn(|c| tiny[c] - theta[c]);
        let first = (0..4).map(|c| gradient[c] * delta[c]).sum::<f64>();
        let mut second = 0.0;
        for a in 0..4 {
            for b in 0..4 {
                second += 0.5 * delta[a] * hessian[a][b] * delta[b];
            }
        }
        let change = objective_change(h, &x, &y, &theta, &tiny);
        near(1e-18, first + second, change);
        near(1e-18, -change, objective_change(h, &x, &y, &tiny, &theta));
    }
    for intercept in [-1000.0, 1000.0] {
        for grade in [0, 1] {
            for step in [-1e-8, 1e-8] {
                let theta = [intercept, 1.0, 0.0, 0.0];
                let candidate = [intercept + step, 1.0, 0.0, 0.0];
                let delta = candidate[0] - theta[0];
                let residual = if grade == 1 {
                    -decoder::sigmoid(-intercept)
                } else {
                    decoder::sigmoid(intercept)
                };
                let expected = delta * (0.5 * (candidate[0] + theta[0]) + residual);
                near(
                    1e-18,
                    expected,
                    objective_change(0, &[[0.0; 3]], &[grade], &theta, &candidate),
                );
            }
        }
    }
}
#[test]
fn constrained_stage_hits_boundary_and_empty_risk_retains_prior() {
    let grades = (0..100).map(|i| (i % 4) as u8).collect::<Vec<_>>();
    let raw = grades
        .iter()
        .map(|&g| [-(g as f64), -(g as f64), -(g as f64)])
        .collect::<Vec<_>>();
    let (model, diagnostics) = DecoderArtifact::fit(&raw, &grades).unwrap();
    for h in 0..3 {
        assert_eq!(model.parameters[h][h + 1], 0.0);
        assert!(diagnostics[h].projected_gradient_maximum <= 1e-7);
    }
    let (empty, diagnostics) = DecoderArtifact::fit(&[], &[]).unwrap();
    for h in 0..3 {
        assert_eq!(empty.parameters[h][h + 1], 1.0);
        assert_eq!(diagnostics[h].risk_rows, 0);
    }
    assert!(DecoderArtifact::fit(&[[f64::NAN, 0.0, 0.0]], &[0]).is_err());
    assert!(DecoderArtifact::fit(&[[0.0; 3]], &[4]).is_err());
}
#[test]
fn no_probability_floor_and_finite_saturated_log_masses() {
    let model = DecoderArtifact {
        version: 1,
        parameters: [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };
    let neutral = model.forecast([0.0; 3]).unwrap();
    for (expected, actual) in [0.5, 0.25, 0.125, 0.125].into_iter().zip(neutral.masses) {
        near(1e-15, expected, actual);
    }
    let tail = model.forecast([20.0; 3]).unwrap();
    assert!(tail.masses.iter().all(|&p| p > 0.0));
    assert_eq!(
        (tail.cumulative[0] as f32) - (tail.cumulative[1] as f32),
        0.0
    );
    let extreme = model.forecast([-1000.0, 1000.0, -1000.0]).unwrap();
    assert!(extreme.log_masses.iter().all(|x| x.is_finite()));
    assert!(extreme.masses.contains(&0.0));
    assert!(model.forecast([f64::NAN, 0.0, 0.0]).is_err());
}
#[test]
fn sparse_objective_reuses_gradient_and_matches_finite_differences() {
    let rows = SparseRows {
        offsets: vec![0, 2, 2, 3, 6],
        columns: vec![0, 2, 1, 2, 0, 2],
    };
    let labels = [1, 0, 1, 1];
    let scales = [0.25, 2.0, 0.75];
    let samples = [1.0, 2.0, 0.5, 1.5];
    let numeric = [-2.0, 0.0, 1.5, 3.0];
    let weights = [0.3, -0.2, 0.1, -0.7, 0.5];
    for penalize_intercept in [true, false] {
        let objective = Objective {
            rows: &rows,
            indices: &[0, 1, 2, 3],
            labels: &labels,
            scales: &scales,
            sample_weights: &samples,
            c: 0.1,
            penalize_intercept,
            numeric: &numeric,
        };
        let mut expected = [0.0; 5];
        let mut loss = 0.0;
        for i in 0..if penalize_intercept { 5 } else { 4 } {
            loss += 0.5 * weights[i] * weights[i];
            expected[i] = weights[i];
        }
        for row in 0..4 {
            let mut z = weights[4];
            for &c in rows.row(row) {
                z += weights[c] * scales[c];
            }
            z += weights[3] * numeric[row];
            let multiplier = 0.1 * samples[row];
            loss += multiplier * decoder::logistic_loss(labels[row] == 1, z);
            let residual = multiplier
                * if labels[row] == 1 {
                    -decoder::sigmoid(-z)
                } else {
                    decoder::sigmoid(z)
                };
            for &c in rows.row(row) {
                expected[c] += residual * scales[c];
            }
            expected[3] += residual * numeric[row];
            expected[4] += residual;
        }
        let mut gradient = [999.0; 5];
        for _ in 0..3 {
            near(1e-15, loss, objective.evaluate(&weights, &mut gradient));
            for c in 0..5 {
                near(1e-15, expected[c], gradient[c]);
            }
        }
        for c in 0..5 {
            let mut plus = weights;
            let mut minus = weights;
            plus[c] += 1e-6;
            minus[c] -= 1e-6;
            let difference = (objective.evaluate(&plus, &mut [0.0; 5])
                - objective.evaluate(&minus, &mut [0.0; 5]))
                / 2e-6;
            near(1e-8, gradient[c], difference);
        }
    }
}
#[test]
fn lbfgs_minimizes_original_objective_and_rejects_uncertified_exits() {
    let evaluate = |w: &[f64], g: &mut [f64]| {
        g[0] = 2.0 * (w[0] - 3.0);
        g[1] = 100.0 * (w[1] + 2.0);
        (w[0] - 3.0).powi(2) + 50.0 * (w[1] + 2.0).powi(2)
    };
    let fitted = minimize(vec![0.0; 2], 1e-8, 500, evaluate).unwrap();
    near(1e-7, 3.0, fitted[0]);
    near(1e-7, -2.0, fitted[1]);
    assert!(minimize(vec![0.0; 2], 1e-8, 0, evaluate).is_err());
    assert!(
        minimize(vec![0.0], 1e-8, 50, |_, g| {
            g[0] = 1.0;
            1.0
        })
        .is_err()
    );
    assert!(
        minimize(vec![0.0], 1e-8, 50, |_, g| {
            g[0] = f64::NAN;
            0.0
        })
        .is_err()
    );
}
#[test]
fn whole_session_partition_and_member_withholding_are_exact() {
    let labels = (0..1000)
        .map(|i| TrainingLabel {
            post_id: i + 1,
            grade: (i % 4) as u8,
            session_id: format!("session-{}", i / 10),
            updated_at: i as f64,
        })
        .collect::<Vec<_>>();
    let parts = partitions(labels.iter()).unwrap();
    let (_, recent) = partition_recent(labels.iter()).unwrap();
    for i in 0..labels.len() {
        assert_eq!(parts.iter().filter(|p| p.train.contains(&i)).count(), 4);
        assert_eq!(parts.iter().filter(|p| p.withheld.contains(&i)).count(), 1);
        assert_eq!(
            parts.iter().filter(|p| p.calibration.contains(&i)).count(),
            usize::from(recent.contains(&i))
        );
    }
    for p in &parts {
        let training = p
            .train
            .iter()
            .map(|&i| &labels[i].session_id)
            .collect::<HashSet<_>>();
        assert!(
            p.withheld
                .iter()
                .all(|&i| !training.contains(&labels[i].session_id))
        );
        assert!(p.calibration.iter().all(|i| p.withheld.contains(i)));
    }
    let one_session = (0..40)
        .map(|i| TrainingLabel {
            post_id: i,
            grade: 0,
            session_id: "one".into(),
            updated_at: i as f64,
        })
        .collect::<Vec<_>>();
    let (train, calibration) = partition_recent(one_session.iter()).unwrap();
    assert_eq!(train.len(), 40);
    assert!(calibration.is_empty());
}
#[test]
fn support_requires_every_conditional_outcome_in_every_member() {
    let mut labels = (0..5000)
        .map(|i| TrainingLabel {
            post_id: i + 1,
            grade: (i % 4) as u8,
            session_id: format!("session-{i}"),
            updated_at: i as f64,
        })
        .collect::<Vec<_>>();
    let parts = partitions(labels.iter()).unwrap();
    let (corpus, _) = tiny_corpus(labels.len());
    let prepared = PreparedTraining::new(&corpus, &labels).unwrap();
    assert!(support_failure(prepared.all(), &parts).is_none());
    for &index in &parts[0].calibration {
        if labels[index].grade == 2 {
            labels[index].grade = 3;
        }
    }
    let prepared = PreparedTraining::new(&corpus, &labels).unwrap();
    assert!(support_failure(prepared.all(), &parts).is_some());
    let starter = fit(&corpus, &labels).unwrap();
    assert_eq!(starter.status.route, ModelRoute::Starter);
    assert_eq!(
        starter.status.fallback_reason,
        support_failure(prepared.all(), &parts)
    );
    assert!(starter.scores.is_none());
    let missing_sessions = (0..30)
        .map(|i| TrainingLabel {
            post_id: i,
            grade: 0,
            session_id: String::new(),
            updated_at: i as f64,
        })
        .collect::<Vec<_>>();
    let (_, held) = partition_recent(missing_sessions.iter()).unwrap();
    assert_eq!(held.len(), 6);
}

fn tiny_corpus(count: usize) -> (Corpus, Vec<TrainingLabel>) {
    let ids = (1..=count as i64).collect::<Vec<_>>();
    let tags = (0..count)
        .map(|i| vec![0, (i % 4 + 1) as i32, 5, 6])
        .collect::<Vec<_>>();
    let corpus = Corpus::fixture(&ids, &tags, 7, &[5], &[6]);
    let labels = (0..count)
        .map(|i| TrainingLabel {
            post_id: ids[i],
            grade: (i % 4) as u8,
            session_id: format!("session-{i}"),
            updated_at: i as f64,
        })
        .collect();
    (corpus, labels)
}

#[test]
fn metadata_classification_does_not_downweight_predictions() {
    let (ordinary, labels) = tiny_corpus(5000);
    let ids = labels.iter().map(|label| label.post_id).collect::<Vec<_>>();
    let tags = (0..ordinary.post_count)
        .map(|row| ordinary.post_tags(row).to_vec())
        .collect::<Vec<_>>();
    // Feature 1 predicts grade 0. Only its metadata classification changes;
    // incidence, artist roles and all training observations remain identical.
    let metadata = Corpus::fixture(&ids, &tags, 7, &[5], &[1, 6]);
    let a = train(&ordinary, &labels, &ModelRecipe::production()).unwrap();
    let b = train(&metadata, &labels, &ModelRecipe::production()).unwrap();
    assert_eq!(a.route(), ModelRoute::Ensemble);
    assert_eq!(b.route(), ModelRoute::Ensemble);
    let rows = (0..ordinary.post_count).collect::<Vec<_>>();
    let expected = a.predict_rows(&rows).unwrap();
    let actual = b.predict_rows(&rows).unwrap();
    for (left, right) in expected.iter().zip(&actual) {
        assert_eq!(left.cumulative, right.cumulative);
        assert_eq!(left.masses, right.masses);
        assert_eq!(left.log_masses, right.log_masses);
    }
}

#[test]
fn prefit_reuse_matches_canonical_prepared_inputs_and_preserves_ordinary_fit_ids() {
    let (mut corpus, labels) = tiny_corpus(5000);
    let serving = fit(&corpus, &labels).unwrap();
    assert_eq!(serving.status.route, ModelRoute::Ensemble);
    let reuse = |corpus: &Corpus, labels: &[TrainingLabel]| {
        fit_if_changed(corpus, labels, &serving.model_id, serving.status.route)
    };
    assert!(reuse(&corpus, &labels).unwrap().is_none());
    let mut reordered = labels.clone();
    reordered.reverse();
    assert!(reuse(&corpus, &reordered).unwrap().is_none());

    for field in 0..3 {
        let mut changed = labels.clone();
        match field {
            0 => changed[0].grade = (changed[0].grade + 1) % 4,
            1 => changed[0].session_id.push_str("-changed"),
            _ => changed[0].updated_at = f64::from_bits(changed[0].updated_at.to_bits() + 1),
        }
        let refreshed = reuse(&corpus, &changed)
            .unwrap()
            .expect("Changed input must fit");
        let ordinary = fit(&corpus, &changed).unwrap();
        assert_ne!(refreshed.model_id, serving.model_id);
        assert_eq!(refreshed.model_id, ordinary.model_id);
        assert_eq!(refreshed.status, ordinary.status);
        assert_eq!(
            refreshed.scores.unwrap().expected_grade,
            ordinary.scores.unwrap().expected_grade
        );
    }
    let changed_route = reuse(&corpus, &labels[..29]).unwrap().unwrap();
    assert_eq!(changed_route.status.route, ModelRoute::Starter);
    assert_eq!(
        changed_route.model_id,
        fit(&corpus, &labels[..29]).unwrap().model_id
    );

    let recipe = ModelRecipe::production().with_c(0.4).unwrap();
    let changed_recipe = fit_changed_with_recipe(
        &corpus,
        &labels,
        &recipe,
        &serving.model_id,
        serving.status.route,
    )
    .unwrap()
    .expect("Recipe changes must fit");
    assert_ne!(changed_recipe.model_id, serving.model_id);
    assert_eq!(
        changed_recipe.model_id,
        train(&corpus, &labels, &recipe).unwrap().model_id()
    );
    assert!(
        fit_changed_with_recipe(
            &corpus,
            &labels,
            &recipe,
            &changed_recipe.model_id,
            changed_recipe.status.route,
        )
        .unwrap()
        .is_none()
    );

    corpus.corpus_id.push_str("-changed");
    assert!(reuse(&corpus, &labels).unwrap().is_some());
    corpus
        .corpus_id
        .truncate(corpus.corpus_id.len() - "-changed".len());
    corpus.manifest_sha256.push('x');
    assert!(reuse(&corpus, &labels).unwrap().is_some());
}

#[test]
fn failed_supported_fit_is_never_an_authority_for_prefit_reuse() {
    static FAILING_NUMERICAL: recipe::NumericalRecipe = recipe::NumericalRecipe {
        decoder: recipe::DecoderNewton {
            iterations: 0,
            tolerance: 1e-8,
            certificate: 1e-7,
        },
        ..recipe::NUMERICAL
    };
    let (corpus, labels) = tiny_corpus(5000);
    let serving = fit(&corpus, &labels).unwrap();
    let mut failing_recipe = ModelRecipe::production();
    failing_recipe.numerical = &FAILING_NUMERICAL;
    for _ in 0..2 {
        assert!(
            fit_changed_with_recipe(
                &corpus,
                &labels,
                &failing_recipe,
                &serving.model_id,
                serving.status.route,
            )
            .is_err()
        );
    }
    assert!(
        fit_if_changed(&corpus, &labels, &serving.model_id, serving.status.route,)
            .unwrap()
            .is_none()
    );
}

#[test]
fn prefit_reuse_validates_even_omitted_labels_before_comparing_serving_identity() {
    let corpus = Corpus::fixture(&[1], &[vec![0]], 1, &[], &[]);
    let base = TrainingLabel {
        post_id: 1,
        grade: 2,
        session_id: "base".into(),
        updated_at: 1.0,
    };
    let serving = fit(&corpus, &[base.clone()]).unwrap();
    let reuse = |labels: &[TrainingLabel]| {
        fit_if_changed(&corpus, labels, &serving.model_id, serving.status.route)
    };
    for omitted_id in [2, 99] {
        let omitted = TrainingLabel {
            post_id: omitted_id,
            ..base.clone()
        };
        assert!(reuse(&[base.clone(), omitted.clone()]).unwrap().is_none());
        for invalid in [
            TrainingLabel {
                grade: 4,
                ..omitted.clone()
            },
            TrainingLabel {
                updated_at: f64::NAN,
                ..omitted.clone()
            },
            TrainingLabel {
                updated_at: f64::INFINITY,
                ..omitted.clone()
            },
        ] {
            // Failed attempts never become a reuse authority.
            assert!(reuse(&[base.clone(), invalid.clone()]).is_err());
            assert!(reuse(&[base.clone(), invalid]).is_err());
        }
        assert!(reuse(&[base.clone(), omitted.clone(), omitted]).is_err());
    }
    assert!(reuse(&[base.clone(), base.clone()]).is_err());
    assert!(reuse(&[base]).unwrap().is_none());
}
#[test]
fn native_artifact_import_requires_schema_and_complete_ensemble_parameters() {
    let (corpus, labels) = tiny_corpus(5000);
    let fitted = train(&corpus, &labels, &ModelRecipe::production()).unwrap();
    let dto = fitted.export().unwrap();
    assert_eq!(dto.version, 4);
    assert!(dto.model_id.starts_with("rust-ensemble-v5-"));
    for version in [1, 2, 3, 5] {
        let mut unsupported = dto.clone();
        unsupported.version = version;
        assert!(TrainedModel::from_artifact(&corpus, unsupported).is_err());
    }

    let native = serde_json::to_value(&dto).unwrap();
    assert_eq!(native["artifact"]["kind"], "ensemble");
    let mut missing_version = native.clone();
    missing_version.as_object_mut().unwrap().remove("version");
    assert!(serde_json::from_value::<TrainedModelArtifact>(missing_version).is_err());
    for kind in ["cold", "bootstrap", "Ensemble", "starter"] {
        let mut unsupported = native.clone();
        unsupported["artifact"]["kind"] = kind.into();
        assert!(serde_json::from_value::<TrainedModelArtifact>(unsupported).is_err());
    }
    let mut old_head = native.clone();
    old_head["artifact"]["artifact"]["members"][0]["predictor"]["heads"][0]["calibration_coefficient"] =
        1.0.into();
    assert!(serde_json::from_value::<TrainedModelArtifact>(old_head).is_err());
    for field in ["popularity", "mean", "scale", "coefficients"] {
        let mut missing = native.clone();
        let predictor = &mut missing["artifact"]["artifact"]["members"][0]["predictor"];
        if field == "popularity" {
            predictor.as_object_mut().unwrap().remove(field);
        } else {
            predictor["popularity"]
                .as_object_mut()
                .unwrap()
                .remove(field);
        }
        assert!(serde_json::from_value::<TrainedModelArtifact>(missing).is_err());
    }
    let mut old_name = native;
    let coefficients = old_name["artifact"]["artifact"]["members"][0]["predictor"]["heads"][0]
        .as_object_mut()
        .unwrap()
        .remove("coefficients")
        .unwrap();
    old_name["artifact"]["artifact"]["members"][0]["predictor"]["heads"][0]["Coefficients"] =
        coefficients;
    assert!(serde_json::from_value::<TrainedModelArtifact>(old_name).is_err());

    for defect in 0..10 {
        let mut invalid = dto.clone();
        let ModelArtifact::Ensemble(ensemble) = &mut invalid.artifact;
        let predictor = &mut ensemble.members[0].predictor;
        match defect {
            0 => {
                predictor.heads[0].coefficients.pop();
            }
            1 => predictor.selected_features.clear(),
            2 => predictor.selected_features.push(0),
            3 => predictor.selected_features.push(corpus.tag_count),
            4 => predictor.popularity.mean = f64::NAN,
            5 => predictor.popularity.scale = 0.0,
            6 => predictor.popularity.scale = -1.0,
            7 => predictor.popularity.scale = f64::INFINITY,
            8 => predictor.popularity.coefficients[1] = f64::NAN,
            9 => predictor.popularity.coefficients[2] = f64::INFINITY,
            _ => unreachable!(),
        }
        assert!(TrainedModel::from_artifact(&corpus, invalid).is_err());
    }

    // Each mutation starts from the same genuinely fitted/admitted artifact.
    TrainedModel::from_artifact(&corpus, dto.clone()).unwrap();
    for defect in 0..14 {
        let mut invalid = dto.clone();
        let ModelArtifact::Ensemble(ensemble) = &mut invalid.artifact;
        match defect {
            0 => ensemble.members[0].predictor.heads[0].coefficients[0] = f64::NAN,
            1 => ensemble.members[0].predictor.heads[1].coefficients[0] = f64::INFINITY,
            2 => ensemble.members[0].predictor.heads[0].intercept = f64::NAN,
            3 => ensemble.members[0].predictor.heads[2].intercept = f64::NEG_INFINITY,
            4 => {
                let predictor = &mut ensemble.members[0].predictor;
                let tag = predictor.selected_features.remove(0);
                // Keep shape, finiteness and all remaining feature indices valid.
                predictor.heads[0].coefficients[tag] = 1.0;
            }
            5 => ensemble.members[0].decoder.version = 2,
            6 => {
                for member in &mut ensemble.members {
                    member.training_rows = usize::MAX - member.withheld_rows + 1;
                }
            }
            7 => ensemble.members[1].training_rows += 1,
            8 => {
                // Equal member source totals, but their withheld sum no longer
                // accounts for the same source population.
                for member in &mut ensemble.members {
                    member.training_rows += 1;
                }
            }
            9..=11 => {
                let stage = &mut ensemble.members[0].stages[defect - 9];
                // Preserve the outcome sum and positive count: only the link
                // to calibration / the preceding conditional stage is broken.
                stage.negative_outcomes += 1;
                stage.risk_rows += 1;
            }
            12 => {
                // Preserve every conditional chain and member source count.
                let member = &mut ensemble.members[0];
                member.stages[2].negative_outcomes = 4;
                member.stages[2].positive_outcomes = member.stages[2].risk_rows - 4;
            }
            13 => {
                // Each member's total is finite and identical. Only the sum
                // of withheld counts overflows; wrapping would equal that
                // member total and incorrectly admit this artifact.
                let third = usize::MAX / 3;
                for member in &mut ensemble.members {
                    member.training_rows = third - 1;
                    member.withheld_rows = third;
                }
            }
            _ => unreachable!(),
        }
        assert!(
            TrainedModel::from_artifact(&corpus, invalid).is_err(),
            "admitted artifact defect {defect}"
        );
    }
    for head in 0..3 {
        let mut invalid = dto.clone();
        let ModelArtifact::Ensemble(ensemble) = &mut invalid.artifact;
        ensemble.members[0].decoder.parameters[head][head + 1] = -0.01;
        assert!(TrainedModel::from_artifact(&corpus, invalid).is_err());
    }
}

#[test]
fn paired_raw_ensemble_packed_scoring_matches_exported_reference_forecasts() {
    let (corpus, labels) = tiny_corpus(5000);
    let parts = partitions(labels.iter()).unwrap();
    let prepared = PreparedTraining::new(&corpus, &labels).unwrap();
    assert!(support_failure(prepared.all(), &parts).is_none());
    let model = train(&corpus, &labels, &ModelRecipe::production()).unwrap();
    let ModelArtifact::Ensemble(artifact) = model.export().unwrap().artifact;
    for member in &artifact.members {
        assert_eq!(member.training_rows + member.withheld_rows, labels.len());
    }
    let rows = [0, 3, 100, 4001, 4999];
    let reference = model.predict_rows(&rows).unwrap();
    let scores = model.score_all().unwrap().unwrap();
    for (index, &row) in rows.iter().enumerate() {
        assert_eq!(
            scores.not_dislike[row],
            reference[index].cumulative[0] as f32
        );
        assert_eq!(
            scores.like_or_love[row],
            reference[index].cumulative[1] as f32
        );
        assert_eq!(scores.love[row], reference[index].cumulative[2] as f32);
        near(1e-12, 1.0, reference[index].masses.iter().sum());
    }
    let mut invalid = model.export().unwrap();
    let ModelArtifact::Ensemble(artifact) = &mut invalid.artifact;
    artifact.members[0].stages[0].projected_gradient_maximum = 1.0;
    assert!(TrainedModel::from_artifact(&corpus, invalid).is_err());

    // Reuse the admitted baseline's support/certificates, not another learning
    // fit. Deliberately set finite executable parameters for this transport
    // fixture so every row's numeric coordinate and tag set affect scoring.
    let count = 8211;
    let ids = (1..=count as i64).collect::<Vec<_>>();
    let tags = (0..count)
        .map(|row| {
            (0..7)
                .filter(|tag| row & (1 << tag) != 0)
                .collect::<Vec<i32>>()
        })
        .collect::<Vec<_>>();
    let popularity = (0..count)
        .map(|row| row as f64 / count as f64 - 0.5)
        .collect::<Vec<_>>();
    let expanded = Corpus::fixture(&ids, &tags, 7, &[5], &[6]).with_popularities(&popularity);
    let mut dto = model.export().unwrap();
    let ModelArtifact::Ensemble(ensemble) = &mut dto.artifact;
    for (m, member) in ensemble.members.iter_mut().enumerate() {
        member.predictor.selected_features = (0..7).collect();
        member.predictor.popularity = PopularityPredictorArtifact {
            mean: 0.1,
            scale: 0.75,
            coefficients: [0.8, -0.6, 0.4],
        };
        for h in 0..3 {
            member.predictor.heads[h].intercept = -0.2 * (m + h + 1) as f64;
            member.predictor.heads[h].coefficients = (0..7)
                .map(|tag| 0.01 * (tag + 1) as f64 * (m + h + 1) as f64)
                .collect();
            member.decoder.parameters[h] = [0.0; 4];
            member.decoder.parameters[h][h + 1] = 1.0;
        }
    }
    let admitted = TrainedModel::from_artifact(&expanded, dto).unwrap();
    let rows = (0..count).collect::<Vec<_>>();
    let reference = admitted.predict_rows(&rows).unwrap();
    let scores = admitted.score_all().unwrap().unwrap();
    for (row, expected) in reference.iter().enumerate() {
        let q = expected.cumulative.map(|value| value as f32);
        assert_eq!(scores.not_dislike[row], q[0], "row {row}");
        assert_eq!(scores.like_or_love[row], q[1], "row {row}");
        assert_eq!(scores.love[row], q[2], "row {row}");
        assert_eq!(
            scores.expected_grade[row],
            (q[0] + q[1]) + q[2],
            "row {row}"
        );
        near(1e-12, 1.0, expected.masses.iter().sum());
    }
    // Explicitly discriminate a reset-to-row-zero chunk offset, including the
    // final partial chunk; periodic tags alone would not distinguish 0 / 8192.
    assert_ne!(reference[8191].cumulative, reference[8190].cumulative);
    for row in [8192, 8193, count - 1] {
        assert_ne!(reference[row].cumulative, reference[row % 8192].cumulative);
    }
}
#[test]
fn starter_has_no_personal_scores_forecasts_or_artifacts_and_does_not_fit() {
    static NO_FITTING: recipe::NumericalRecipe = recipe::NumericalRecipe {
        predictor: recipe::Lbfgs {
            iterations: 0,
            tolerance: 1e-5,
        },
        decoder: recipe::DecoderNewton {
            iterations: 0,
            tolerance: 1e-8,
            certificate: 1e-7,
        },
    };
    let recipe = ModelRecipe {
        numerical: &NO_FITTING,
        ..ModelRecipe::production()
    };
    for count in [0, 29, 120] {
        let (corpus, labels) = tiny_corpus(count);
        let starter = train(&corpus, &labels, &recipe).unwrap();
        assert_eq!(starter.route(), ModelRoute::Starter);
        assert!(starter.model_id().starts_with("rust-starter-v5-"));
        assert!(starter.score_all().unwrap().is_none());
        assert!(starter.predict_rows(&[]).is_err());
        assert!(starter.export().is_err());
        assert!(fit(&corpus, &labels).unwrap().scores.is_none());
    }
    let (corpus, labels) = tiny_corpus(29);
    let mut duplicate = labels.clone();
    duplicate.push(labels[0].clone());
    assert!(train(&corpus, &duplicate, &ModelRecipe::production()).is_err());
    let (supported, labels) = tiny_corpus(5000);
    let ids = labels.iter().map(|label| label.post_id).collect::<Vec<_>>();
    let empty = Corpus::fixture(&ids, &vec![vec![]; supported.post_count], 1, &[], &[]);
    assert!(train(&empty, &labels, &ModelRecipe::production()).is_err());
}

#[test]
fn packed_scalar_and_selected_simd_kernel_preserve_each_head_addition_order() {
    let intercepts = std::array::from_fn(|i| (i as f64 - 7.0) * 0.125);
    let stride = 16;
    let coefficients = (0..3 * stride)
        .map(|i| (i as f64 * 0.37).sin())
        .collect::<Vec<_>>();
    for tags in [&[][..], &[2, 0, 2, 1][..], &[0, 1, 2][..]] {
        let scalar = packed_scalar(tags, &coefficients, &intercepts, stride);
        // All tags and dimensions meet the kernel's safety contract, and
        // selection performs runtime CPU detection.
        let selected = unsafe { packed_kernel()(tags, &coefficients, &intercepts, stride) };
        assert_eq!(scalar.map(f64::to_bits), selected.map(f64::to_bits));
    }
}

#[test]
fn finite_overflow_artifacts_rejected_by_reference_and_packed_serving() {
    let (corpus, labels) = tiny_corpus(5000);
    let fitted = train(&corpus, &labels, &ModelRecipe::production()).unwrap();
    for raw_overflow in [true, false] {
        let mut model = fitted.export().unwrap();
        let ModelArtifact::Ensemble(artifact) = &mut model.artifact;
        let member = &mut artifact.members[0];
        let head = &mut member.predictor.heads[0];
        head.coefficients.fill(0.0);
        if raw_overflow {
            head.intercept = f64::MAX;
            head.coefficients[0] = f64::MAX;
        } else {
            head.intercept = 2.0;
            member.decoder.parameters[0] = [f64::MAX, f64::MAX, 0.0, 0.0];
        }
        // Every stored parameter is finite; failure must come from evaluating
        // the actual row, before any score buffers can be returned/published.
        let model = TrainedModel::from_artifact(&corpus, model).unwrap();
        assert!(model.predict_rows(&[0]).is_err());
        assert!(model.score_all().is_err());
    }
}

#[test]
fn production_decoder_rounded_loss_stall_regression_preserves_packed_consumers() {
    let (corpus, labels) = tiny_corpus(5000);
    // This first-4800-label fit previously stalled in the decoder when Armijo
    // compared rounded total losses instead of stable objective changes.
    let training_labels = &labels[..4800];
    let model = train(&corpus, training_labels, &ModelRecipe::production()).unwrap();
    let ModelArtifact::Ensemble(artifact) = model.export().unwrap().artifact;
    for member in &artifact.members {
        assert_eq!(
            member.training_rows + member.withheld_rows,
            training_labels.len()
        );
        for stage in &member.stages {
            assert!(stage.projected_gradient_maximum <= 1e-7);
        }
    }
    let restored: TrainedModelArtifact =
        serde_json::from_slice(&serde_json::to_vec(&model.export().unwrap()).unwrap()).unwrap();
    let restored = TrainedModel::from_artifact(&corpus, restored).unwrap();
    let rows = (4800..5000).collect::<Vec<_>>();
    let reference = restored.predict_rows(&rows).unwrap();
    let scores = restored.score_all().unwrap().unwrap();
    for (&row, forecast) in rows.iter().zip(&reference) {
        let q = [
            scores.not_dislike[row],
            scores.like_or_love[row],
            scores.love[row],
        ];
        assert_eq!(q, forecast.cumulative.map(|value| value as f32));
        assert!(q[0] >= q[1] && q[1] >= q[2] && q[2] >= 0.0 && q[0] <= 1.0);
        assert_eq!(scores.expected_grade[row], (q[0] + q[1]) + q[2]);
        near(1e-12, 1.0, forecast.masses.iter().sum());
        assert!(forecast.log_masses.iter().all(|value| value.is_finite()));
    }
    assert!(scores.expected_grade[4803] > scores.expected_grade[4800]);
}

#[test]
fn serialized_labels_preserve_adjacent_timestamp_calibration_order() {
    let later = 1_784_872_757.285_959_7_f64;
    let earlier = f64::from_bits(later.to_bits() - 1);
    let labels = [
        TrainingLabel {
            post_id: 1,
            grade: 0,
            session_id: "z-earlier".into(),
            updated_at: earlier,
        },
        TrainingLabel {
            post_id: 2,
            grade: 3,
            session_id: "a-later".into(),
            updated_at: later,
        },
    ];
    let bytes = serde_json::to_vec(&labels).unwrap();
    let restored: Vec<TrainingLabel> = serde_json::from_slice(&bytes).unwrap();
    let (training, calibration) = partition_recent(restored.iter()).unwrap();
    assert_eq!(
        training
            .iter()
            .map(|&i| restored[i].post_id)
            .collect::<Vec<_>>(),
        vec![1]
    );
    assert_eq!(
        calibration
            .iter()
            .map(|&i| restored[i].post_id)
            .collect::<Vec<_>>(),
        vec![2]
    );
}

#[test]
fn preparation_validates_all_inputs_before_omitting_absent_posts_and_aligns_rows_once() {
    let corpus = Corpus::fixture(&[10, 20, 30], &[vec![0], vec![0], vec![0]], 1, &[], &[]);
    let labels = vec![
        TrainingLabel {
            post_id: 30,
            grade: 3,
            session_id: "late".into(),
            updated_at: 2.0,
        },
        TrainingLabel {
            post_id: 20,
            grade: 2,
            session_id: "earliest".into(),
            updated_at: 0.0,
        },
        TrainingLabel {
            post_id: 10,
            grade: 1,
            session_id: "early".into(),
            updated_at: 1.0,
        },
        TrainingLabel {
            post_id: 99,
            grade: 0,
            session_id: "absent".into(),
            updated_at: 0.0,
        },
    ];
    let prepared = PreparedTraining::new(&corpus, &labels).unwrap();
    assert_eq!(
        prepared
            .all()
            .iter()
            .map(|item| (item.label.post_id, item.row))
            .collect::<Vec<_>>(),
        [(20, 1), (10, 0), (30, 2)]
    );
    for index in [1, 3] {
        let mut duplicate = labels.clone();
        duplicate.push(labels[index].clone());
        assert!(PreparedTraining::new(&corpus, &duplicate).is_err());
        let mut invalid = labels.clone();
        invalid[index].grade = 4;
        assert!(PreparedTraining::new(&corpus, &invalid).is_err());
        invalid[index].grade = 0;
        invalid[index].updated_at = f64::NAN;
        assert!(PreparedTraining::new(&corpus, &invalid).is_err());
    }
}

#[test]
fn member_view_support_and_feature_priors_exclude_withheld_rows() {
    let ids = (1..=60).collect::<Vec<_>>();
    let tags = (0..60)
        .map(|i| {
            if i < 40 {
                vec![0, (i % 2 + 2) as i32]
            } else {
                vec![0, 1, 2]
            }
        })
        .collect::<Vec<_>>();
    let corpus = Corpus::fixture(&ids, &tags, 4, &[], &[]);
    let labels = ids
        .iter()
        .enumerate()
        .map(|(i, &post_id)| TrainingLabel {
            post_id,
            grade: (i % 4) as u8,
            session_id: format!("s{i}"),
            updated_at: i as f64,
        })
        .collect::<Vec<_>>();
    let prepared = PreparedTraining::new(&corpus, &labels).unwrap();
    let indices = (0..40).collect::<Vec<_>>();
    let independent = PreparedTraining::new(&corpus, &labels[..40]).unwrap();
    let mut changed = labels.clone();
    for label in &mut changed[40..] {
        label.grade = 3 - label.grade;
    }
    let changed = PreparedTraining::new(&corpus, &changed).unwrap();
    for prior in [FeaturePrior::Adaptive, FeaturePrior::Uniform] {
        let recipe = ModelRecipe::production()
            .with_feature_prior(prior)
            .with_c(0.4)
            .unwrap();
        let member = fit_raw_predictor(&corpus, prepared.select(&indices), &recipe)
            .unwrap()
            .unwrap()
            .export();
        let reference = fit_raw_predictor(&corpus, independent.all(), &recipe)
            .unwrap()
            .unwrap()
            .export();
        let changed = fit_raw_predictor(&corpus, changed.select(&indices), &recipe)
            .unwrap()
            .unwrap()
            .export();
        assert_eq!(member.selected_features, [0, 2, 3]);
        assert_eq!(
            serde_json::to_vec(&member).unwrap(),
            serde_json::to_vec(&reference).unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&member).unwrap(),
            serde_json::to_vec(&changed).unwrap()
        );
        let global = fit_raw_predictor(&corpus, prepared.all(), &recipe)
            .unwrap()
            .unwrap()
            .export();
        assert!(global.selected_features.contains(&1));
    }
}

#[test]
fn native_recent_partition_keeps_whole_groups_and_rust_unicode_ties() {
    let labels = [
        TrainingLabel {
            post_id: 1,
            grade: 0,
            session_id: "z".into(),
            updated_at: 0.0,
        },
        TrainingLabel {
            post_id: 2,
            grade: 1,
            session_id: "a".into(),
            updated_at: 0.0,
        },
        TrainingLabel {
            post_id: 3,
            grade: 2,
            session_id: "z".into(),
            updated_at: 0.0,
        },
        TrainingLabel {
            post_id: 4,
            grade: 3,
            session_id: "\u{10000}".into(),
            updated_at: 0.0,
        },
        TrainingLabel {
            post_id: 5,
            grade: 0,
            session_id: "\u{e000}".into(),
            updated_at: 0.0,
        },
        TrainingLabel {
            post_id: 6,
            grade: 1,
            session_id: "\u{10000}".into(),
            updated_at: 0.0,
        },
    ];
    // UTF-8 Rust ordering places U+10000 after U+E000. Both rows in the newest
    // tied group are held together, and both outputs retain input-row order.
    let (train, held) = partition_recent(labels.iter()).unwrap();
    assert_eq!(train, [0, 1, 2, 4]);
    assert_eq!(held, [3, 5]);
}

#[test]
fn native_recent_partition_uses_prefixed_keys_max_update_and_ceiling() {
    let labels = [
        TrainingLabel {
            post_id: 1,
            grade: 0,
            session_id: "".into(),
            updated_at: 1.0,
        },
        TrainingLabel {
            post_id: 2,
            grade: 0,
            session_id: "post:1".into(),
            updated_at: 2.0,
        },
        TrainingLabel {
            post_id: 3,
            grade: 0,
            session_id: "recent".into(),
            updated_at: 0.0,
        },
        TrainingLabel {
            post_id: 4,
            grade: 0,
            session_id: "middle".into(),
            updated_at: 3.0,
        },
        TrainingLabel {
            post_id: 5,
            grade: 0,
            session_id: "recent".into(),
            updated_at: 5.0,
        },
        TrainingLabel {
            post_id: 6,
            grade: 0,
            session_id: "last".into(),
            updated_at: 6.0,
        },
    ];
    // ceil(6 * .2) requires two rows, so the newest single-row group and the
    // whole group dated by its maximum update are held (three rows in total).
    let (train, held) = partition_recent(labels.iter()).unwrap();
    assert_eq!(train, [0, 1, 3]);
    assert_eq!(held, [2, 4, 5]);
    assert_ne!(session_key(&labels[0]), session_key(&labels[1]));
    let (train, held) = partition_recent(labels[..2].iter()).unwrap();
    assert_eq!(train, [0]);
    assert_eq!(held, [1]);
    let mut oldest_group = labels.clone();
    for label in &mut oldest_group[..5] {
        label.session_id = "oldest".into();
        label.updated_at = 0.0;
    }
    let (train, held) = partition_recent(oldest_group.iter()).unwrap();
    assert_eq!(train, [0, 1, 2, 3, 4]);
    assert_eq!(held, [5]);
}

#[test]
fn imported_runtime_is_isolated_from_mutable_exported_artifacts() {
    let (corpus, labels) = tiny_corpus(5000);
    let fitted = train(&corpus, &labels, &ModelRecipe::production()).unwrap();
    let rows = [0, 31, 4999];
    let reference = fitted.predict_rows(&rows).unwrap();
    let mut dto = fitted.export().unwrap();
    assert!(matches!(&dto.artifact, ModelArtifact::Ensemble(_)));
    let imported = TrainedModel::from_artifact(&corpus, dto.clone()).unwrap();
    let ModelArtifact::Ensemble(ensemble) = &mut dto.artifact;
    ensemble.members[0].predictor.heads[0]
        .coefficients
        .fill(f64::NAN);
    ensemble.members[1].predictor.popularity.mean = f64::NAN;
    ensemble.members[2].predictor.popularity.scale = 0.0;
    ensemble.members[3]
        .predictor
        .popularity
        .coefficients
        .fill(f64::NAN);
    assert!(TrainedModel::from_artifact(&corpus, dto).is_err());
    let actual = imported.predict_rows(&rows).unwrap();
    for (actual, expected) in actual.iter().zip(&reference) {
        assert_eq!(actual.cumulative, expected.cumulative);
        assert_eq!(actual.masses, expected.masses);
        assert_eq!(actual.log_masses, expected.log_masses);
    }
}

#[test]
fn support_status_reports_actual_counts_and_conditional_stage_boundaries() {
    let (corpus, labels) = tiny_corpus(29);
    let starter = fit(&corpus, &labels).unwrap();
    assert_eq!(
        starter.status,
        ModelStatus {
            route: ModelRoute::Starter,
            fallback_reason: Some(SupportReason::InsufficientLabels {
                actual: 29,
                minimum: 30
            }),
        }
    );
    let (corpus, mut labels) = tiny_corpus(40);
    for label in &mut labels {
        label.session_id = "one".into();
    }
    let starter = fit(&corpus, &labels).unwrap();
    assert_eq!(
        starter.status,
        ModelStatus {
            route: ModelRoute::Starter,
            fallback_reason: Some(SupportReason::InsufficientSessions {
                actual: 1,
                minimum: 2
            }),
        }
    );
    let empty = Corpus::fixture(
        &(1..=40).collect::<Vec<_>>(),
        &vec![vec![]; 40],
        1,
        &[],
        &[],
    );
    let unsupported = fit(&empty, &labels).unwrap();
    assert_eq!(unsupported.status, starter.status);
    assert!(unsupported.scores.is_none());
    let (corpus, mut labels) = tiny_corpus(5000);
    let parts = partitions(labels.iter()).unwrap();
    let calibration = &parts[0].calibration;
    // Exactly five outcomes at each stage remains supported; four fails.
    for &index in calibration {
        labels[index].grade = 3;
    }
    for (position, &index) in calibration.iter().take(15).enumerate() {
        labels[index].grade = (position / 5) as u8;
    }
    let prepared = PreparedTraining::new(&corpus, &labels).unwrap();
    assert!(support_failure(prepared.all(), &parts).is_none());
    labels[calibration[10]].grade = 3;
    let prepared = PreparedTraining::new(&corpus, &labels).unwrap();
    assert_eq!(
        support_failure(prepared.all(), &parts),
        Some(SupportReason::ConditionalOutcomes {
            member: 0,
            stage: 3,
            negative: 4,
            positive: calibration.len() - 14,
            minimum: 5,
        })
    );
}

#[test]
fn effective_recipes_change_learning_and_identity_without_changing_routes() {
    let (corpus, labels) = tiny_corpus(5000);
    let corpus = corpus.with_popularities(
        &(0..5000)
            .map(|i| ((i % 4) as f64 - 1.5) * 0.8 + (i % 17) as f64 / 100.0)
            .collect::<Vec<_>>(),
    );
    let production = ModelRecipe::production();
    let uniform = production.clone().with_feature_prior(FeaturePrior::Uniform);
    let stronger_c = production.clone().with_c(0.4).unwrap();
    assert_eq!(
        production.identity(),
        production
            .clone()
            .with_c(0.1)
            .unwrap()
            .with_feature_prior(FeaturePrior::Adaptive)
            .identity()
    );
    for bad in [0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(production.clone().with_c(bad).is_err());
    }
    let lower_c = production.clone().with_c(0.00625).unwrap();
    let baseline = train(&corpus, &labels, &production).unwrap();
    let reference = baseline.predict_rows(&[0, 1, 2, 3]).unwrap();
    let ModelArtifact::Ensemble(reference_artifact) = baseline.export().unwrap().artifact;
    for recipe in [&uniform, &stronger_c, &lower_c] {
        assert_ne!(recipe.identity(), production.identity());
        let changed = train(&corpus, &labels, recipe).unwrap();
        assert_eq!(changed.route(), baseline.route());
        assert_ne!(changed.model_id(), baseline.model_id());
        let ModelArtifact::Ensemble(artifact) = changed.export().unwrap().artifact;
        assert_eq!(
            artifact.members[0].predictor.selected_features,
            reference_artifact.members[0].predictor.selected_features
        );
        assert!(
            artifact.members[0]
                .predictor
                .heads
                .iter()
                .zip(&reference_artifact.members[0].predictor.heads)
                .any(|(a, b)| a.coefficients != b.coefficients || a.intercept != b.intercept)
        );
        let forecasts = changed.predict_rows(&[0, 1, 2, 3]).unwrap();
        assert!(
            forecasts
                .iter()
                .zip(&reference)
                .any(|(a, b)| a.cumulative != b.cumulative)
        );
        let restored = TrainedModel::from_artifact(&corpus, changed.export().unwrap()).unwrap();
        assert_eq!(
            serde_json::to_vec(&forecasts).unwrap(),
            serde_json::to_vec(&restored.predict_rows(&[0, 1, 2, 3]).unwrap()).unwrap()
        );
    }
    assert_ne!(uniform.identity(), stronger_c.identity());
}

#[test]
fn popularity_joint_objective_matches_accepted_numeric_certificate() {
    let rows = SparseRows {
        offsets: vec![0, 2, 3, 3, 4, 6],
        columns: vec![0, 1, 1, 0, 0, 1],
    };
    let numeric = [-2.0, 0.0, 1.5, -0.25, 3.0];
    let objective = Objective {
        rows: &rows,
        indices: &[0, 1, 2, 3, 4],
        labels: &[0, 1, 0, 1, 1],
        scales: &[0.4, 1.7],
        sample_weights: &[1.25, 5.0 / 6.0, 1.25, 5.0 / 6.0, 5.0 / 6.0],
        c: 0.1,
        penalize_intercept: true,
        numeric: &numeric,
    };
    let weights = [0.3, -0.5, -0.7, 0.2];
    let mut gradient = [0.0; 4];
    let loss = objective.evaluate(&weights, &mut gradient);
    // FF-26 objective-probe.json, evaluated by its accepted joint operator.
    near(1e-15, 0.9857747664118766, loss);
    let epsilon = 1e-5;
    for c in 0..4 {
        let mut plus = weights;
        let mut minus = weights;
        plus[c] += epsilon;
        minus[c] -= epsilon;
        let difference = (objective.evaluate(&plus, &mut [0.0; 4])
            - objective.evaluate(&minus, &mut [0.0; 4]))
            / (2.0 * epsilon);
        near(1e-8, gradient[c], difference);
    }
    let direction = [0.2, -0.3, 0.7, -0.4];
    let mut hv = direction;
    for (i, &row) in objective.indices.iter().enumerate() {
        let mut x = [0.0, 0.0, numeric[row], 1.0];
        for &c in rows.row(row) {
            x[c] = objective.scales[c];
        }
        let z = weights.iter().zip(x).map(|(w, x)| w * x).sum::<f64>();
        let dx = direction.iter().zip(x).map(|(d, x)| d * x).sum::<f64>();
        let factor = objective.c
            * objective.sample_weights[i]
            * decoder::sigmoid(z)
            * decoder::sigmoid(-z)
            * dx;
        for c in 0..4 {
            hv[c] += factor * x[c];
        }
    }
    let plus = std::array::from_fn::<_, 4, _>(|c| weights[c] + epsilon * direction[c]);
    let minus = std::array::from_fn::<_, 4, _>(|c| weights[c] - epsilon * direction[c]);
    let (mut gp, mut gm) = ([0.0; 4], [0.0; 4]);
    objective.evaluate(&plus, &mut gp);
    objective.evaluate(&minus, &mut gm);
    for c in 0..4 {
        near(1e-8, hv[c], (gp[c] - gm[c]) / (2.0 * epsilon));
    }
    let fitted = minimize(vec![0.0; 4], 1e-10, 500, |w, g| objective.evaluate(w, g)).unwrap();
    let fitted_loss = objective.evaluate(&fitted, &mut gradient);
    near(1e-15, 0.3383063060972507, fitted_loss);
    assert!(
        gradient
            .iter()
            .all(|g| g.abs() <= 1e-10 * fitted_loss.max(1.0))
    );
}

#[test]
fn popularity_normalization_and_fit_ignore_rows_outside_raw_training() {
    let (corpus, labels) = tiny_corpus(120);
    let values = (0..120).map(|i| i as f64 - 60.0).collect::<Vec<_>>();
    let corpus = corpus.with_popularities(&values);
    let prepared = PreparedTraining::new(&corpus, &labels).unwrap();
    let indices = (0..80).collect::<Vec<_>>();
    let recipe = ModelRecipe::production();
    let first = fit_raw_predictor(&corpus, prepared.select(&indices), &recipe)
        .unwrap()
        .unwrap();
    let (other, _) = tiny_corpus(120);
    let changed = values
        .iter()
        .enumerate()
        .map(|(i, &value)| if i < 80 { value } else { 1e6 + i as f64 })
        .collect::<Vec<_>>();
    let other = other.with_popularities(&changed);
    let second = fit_raw_predictor(&other, prepared.select(&indices), &recipe)
        .unwrap()
        .unwrap();
    assert_eq!(
        serde_json::to_vec(&first.export()).unwrap(),
        serde_json::to_vec(&second.export()).unwrap(),
    );
    near(1e-12, -20.5, first.popularity.mean);
    near(
        1e-12,
        ((80.0_f64 * 80.0 - 1.0) / 12.0).sqrt(),
        first.popularity.scale,
    );
}

#[test]
fn public_training_learns_unrestricted_popularity_and_constants_have_no_effect() {
    let count = 5000;
    let ids = (1..=count as i64).collect::<Vec<_>>();
    let tags = vec![vec![0]; count];
    let values = (0..count).map(|i| (i % 4) as f64 - 1.5).collect::<Vec<_>>();
    let corpus = Corpus::fixture(&ids, &tags, 1, &[], &[]).with_popularities(&values);
    let labels = ids
        .iter()
        .enumerate()
        .map(|(i, &post_id)| TrainingLabel {
            post_id,
            grade: (3 - i % 4) as u8,
            session_id: format!("session-{i}"),
            updated_at: i as f64,
        })
        .collect::<Vec<_>>();
    let model = train(&corpus, &labels, &ModelRecipe::production()).unwrap();
    let ModelArtifact::Ensemble(artifact) = model.export().unwrap().artifact;
    for member in &artifact.members {
        assert!(
            member
                .predictor
                .popularity
                .coefficients
                .iter()
                .all(|&c| c < 0.0)
        );
    }
    let forecasts = model.predict_rows(&[0, 119]).unwrap();
    assert!(forecasts[0].cumulative[0] > forecasts[1].cumulative[0]);
    assert!(forecasts[0].cumulative[2] > forecasts[1].cumulative[2]);
    let served = fit(&corpus, &labels).unwrap();
    assert_eq!(served.model_id, model.model_id());
    assert_eq!(
        served.scores.unwrap().love[0],
        forecasts[0].cumulative[2] as f32
    );
    let mut constant_forecasts = None;
    for value in [0.0, 7.0, -9.0] {
        let corpus =
            Corpus::fixture(&ids, &tags, 1, &[], &[]).with_popularities(&vec![value; count]);
        let model = train(&corpus, &labels, &ModelRecipe::production()).unwrap();
        let ModelArtifact::Ensemble(artifact) = model.export().unwrap().artifact;
        for member in &artifact.members {
            assert_eq!(member.predictor.popularity.mean, value);
            assert_eq!(member.predictor.popularity.scale, 1.0);
            assert_eq!(member.predictor.popularity.coefficients, [0.0; 3]);
        }
        let actual = serde_json::to_vec(&model.predict_rows(&[0, 119]).unwrap()).unwrap();
        if let Some(expected) = &constant_forecasts {
            assert_eq!(&actual, expected);
        } else {
            constant_forecasts = Some(actual);
        }
    }
}

#[test]
fn numeric_transport_and_packed_forecasts_cover_supported_ensemble() {
    let count = 5000;
    let (corpus, labels) = tiny_corpus(count);
    let values = (0..count)
        .map(|i| ((i % 4) as f64 - 1.5) * 0.9 + (i % 17) as f64 / 10.0)
        .collect::<Vec<_>>();
    let corpus = corpus.with_popularities(&values);
    let model = train(&corpus, &labels, &ModelRecipe::production()).unwrap();
    assert_eq!(model.route(), ModelRoute::Ensemble);
    let dto = model.export().unwrap();
    let ModelArtifact::Ensemble(ensemble) = &dto.artifact;
    for member in &ensemble.members {
        let expected = PopularityPredictorArtifact::fit(
            labels
                .iter()
                .enumerate()
                .filter(|(_, label)| member_for(label) != member.member)
                .map(|(i, _)| values[i])
                .collect::<Vec<_>>()
                .into_iter(),
        )
        .unwrap();
        assert_eq!(member.predictor.popularity.mean, expected.mean);
        assert_eq!(member.predictor.popularity.scale, expected.scale);
        assert!(
            member
                .predictor
                .popularity
                .coefficients
                .iter()
                .any(|&c| c != 0.0)
        );
    }
    let restored = TrainedModel::from_artifact(
        &corpus,
        serde_json::from_slice(&serde_json::to_vec(&dto).unwrap()).unwrap(),
    )
    .unwrap();
    let rows = (0..count).collect::<Vec<_>>();
    let reference = model.predict_rows(&rows).unwrap();
    let forecasts = restored.predict_rows(&rows).unwrap();
    let scores = restored.score_all().unwrap().unwrap();
    for (row, (actual, expected)) in forecasts.iter().zip(reference).enumerate() {
        assert_eq!(actual.cumulative, expected.cumulative);
        assert_eq!(actual.masses, expected.masses);
        assert_eq!(actual.log_masses, expected.log_masses);
        assert_eq!(scores.not_dislike[row], actual.cumulative[0] as f32);
        assert_eq!(scores.like_or_love[row], actual.cumulative[1] as f32);
        assert_eq!(scores.love[row], actual.cumulative[2] as f32);
    }
    assert!(restored.predict_rows(&[count]).is_err());
    let mut wrong_corpus = dto;
    wrong_corpus.corpus_manifest_sha256.push('x');
    assert!(TrainedModel::from_artifact(&corpus, wrong_corpus).is_err());
}

#[test]
fn numeric_affine_overflow_is_rejected_by_scalar_and_packed_serving() {
    let (corpus, labels) = tiny_corpus(5000);
    let fitted = train(&corpus, &labels, &ModelRecipe::production()).unwrap();
    let mut dto = fitted.export().unwrap();
    let ModelArtifact::Ensemble(ensemble) = &mut dto.artifact;
    let popularity = &mut ensemble.members[0].predictor.popularity;
    popularity.mean = f64::MAX;
    popularity.scale = f64::MIN_POSITIVE;
    popularity.coefficients = [1.0; 3];
    // Finite transport cannot prove finite evaluation for an arbitrary row.
    let model = TrainedModel::from_artifact(&corpus, dto).unwrap();
    assert!(model.predict_rows(&[0]).is_err());
    assert!(model.score_all().is_err());
}

#[test]
fn member_support_gates_distinguish_29_from_30_without_fitting() {
    let (corpus, labels) = tiny_corpus(60);
    let prepared = PreparedTraining::new(&corpus, &labels).unwrap();
    // Direct gate inputs isolate row-count thresholds from the whole-session
    // partitioner. They are not presented as a jointly realizable ensemble fit.
    let supported = || {
        std::array::from_fn(|_| Partition {
            train: (0..30).collect(),
            withheld: (30..60).collect(),
            calibration: (30..60).collect(),
        })
    };
    assert!(support_failure(prepared.all(), &supported()).is_none());
    for member in 0..5 {
        let mut parts = supported();
        parts[member].train.pop();
        assert_eq!(
            support_failure(prepared.all(), &parts),
            Some(SupportReason::MemberTrainingRows {
                member,
                actual: 29,
                minimum: 30,
            })
        );
        parts[member].train.push(29);
        assert!(support_failure(prepared.all(), &parts).is_none());

        parts[member].calibration.pop();
        assert_eq!(
            support_failure(prepared.all(), &parts),
            Some(SupportReason::MemberCalibrationRows {
                member,
                actual: 29,
                minimum: 30,
            })
        );
        parts[member].calibration.push(59);
        assert!(support_failure(prepared.all(), &parts).is_none());
    }
}

#[test]
fn feature_support_uses_member_view_artist_3_and_nonartist_5_boundaries() {
    let ids = (1..=10).collect::<Vec<_>>();
    let tags = (0..10)
        .map(|row| {
            [(0, 2), (1, 3), (2, 4), (3, 5), (4, 4), (5, 5)]
                .into_iter()
                .filter_map(|(tag, count)| (row < count).then_some(tag))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // Category-7 metadata follows the same nonartist threshold as content.
    let corpus = Corpus::fixture(&ids, &tags, 6, &[0, 1], &[4, 5]);
    let labels = ids
        .iter()
        .map(|&post_id| TrainingLabel {
            post_id,
            grade: (post_id % 4) as u8,
            session_id: format!("feature-{post_id}"),
            updated_at: post_id as f64,
        })
        .collect::<Vec<_>>();
    let prepared = PreparedTraining::new(&corpus, &labels).unwrap();
    assert_eq!(
        predictor::supported_features(&corpus, prepared.all()).collect::<Vec<_>>(),
        vec![1, 3, 5]
    );
    // Withholding one shared row crosses both admission thresholds. Global
    // support must not leak into a member's selected features.
    let member_rows = [0, 1, 3, 4, 5, 6, 7, 8, 9];
    assert!(
        predictor::supported_features(&corpus, prepared.select(&member_rows))
            .next()
            .is_none()
    );
}

#[test]
fn five_way_member_assignment_matches_independent_sha_byte_vectors() {
    // Python hashlib SHA-256 over these literal UTF-8 bytes, interpreted as one
    // big-endian 256-bit integer modulo five (not only a digest prefix).
    let cases = [
        (
            1,
            "alpha",
            "2d964ad7731a15a0a287d706cbda3bafd0dc117bcd467ad27ed4c3e4b2bd9e71",
            3,
        ),
        (
            2,
            "beta",
            "bd4318f2c606056af84c13a10e9553bafd103fd99a4ccb35aded3b5d47f2e053",
            0,
        ),
        (
            3,
            "\u{1f98a}",
            "1f69f90706e5cd446b39b8eafa01ebad337403980627ed67e0ebc35903082f39",
            1,
        ),
        (
            4,
            "\u{e000}",
            "a0e24eb9c14d54f9f0569fc17262ff75c864d650cc800920024d843f31df5b1f",
            4,
        ),
        (
            5,
            "\u{10000}",
            "00763c943607e566dbcd08173bb3bb72fb85f6da6e0218fefcf05dd30e8de2af",
            2,
        ),
        (
            1,
            "",
            "b4647eabee7219280e8ad0402bcf0df119f516357f46ac90a69969380ea46277",
            0,
        ),
        (
            4294967297,
            " \t",
            "1d68b62e556324d6d726a015c8b15a86742471f19e885c1a4a0255a7194e2b3b",
            1,
        ),
        (
            i64::MAX,
            "",
            "5f308cc4ce5010fa9b5a39b9ed8def93e054b634330929f1da9f9eb02928c4eb",
            4,
        ),
        (
            1,
            "post:1",
            "375ff9b88cf7cebf1b097add6eb0c89acc7b9fb91fed92143bdae30d34105887",
            4,
        ),
        (
            1,
            " alpha ",
            "9cb9686e15e2b7df94e4bb6e1df0e8504aa6f01cb24122c86c5982189ff2288a",
            2,
        ),
    ];
    for (post_id, session, digest, expected) in cases {
        let key = if session.trim().is_empty() {
            format!("post:{post_id}")
        } else {
            format!("session:{session}")
        };
        let bytes = format!("ff-retained-independent-all-v1|member|{key}");
        assert_eq!(format!("{:x}", Sha256::digest(bytes.as_bytes())), digest);
        let label = TrainingLabel {
            post_id,
            session_id: session.into(),
            grade: 0,
            updated_at: 0.0,
        };
        assert_eq!(member_for(&label), expected);
    }
}

use super::*;
fn near(expected: f64, actual: f64, tolerance: f64) {
    assert!(
        (expected - actual).abs() <= tolerance,
        "expected {expected}, actual {actual}"
    );
}
#[test]
fn class_balancing_and_priors_retain_production_scales() {
    let labels = [1, 0, 0, 0];
    assert_eq!(
        balanced_weights(&labels),
        vec![2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
    );
    assert_eq!(balanced_weights(&[1, 1]), vec![1.0, 1.0]);
    let rows = SparseRows {
        offsets: vec![0, 2, 3, 4, 5],
        columns: vec![0, 1, 1, 2, 2],
    };
    let scales = scales_for(&rows, &[0, 1, 2, 3], &labels, 3, FeaturePrior::Adaptive);
    let positive = [6.0_f64, 6.0, 5.0];
    let negative = [5.0_f64, 6.0, 7.0];
    for c in 0..3 {
        let expected = ((positive[c] / 17.0) / (negative[c] / 18.0))
            .ln()
            .abs()
            .clamp(0.25, 2.0);
        near(expected, scales[c], 1e-15);
    }
    let fit_only = scales_for(&rows, &[0], &[1], 3, FeaturePrior::Adaptive);
    let all = scales_for(&rows, &[0, 1, 2, 3], &labels, 3, FeaturePrior::Adaptive);
    assert!(
        fit_only
            .iter()
            .chain(&all)
            .all(|x| (0.25..=2.0).contains(x))
    );
}

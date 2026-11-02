use anyhow::{Result, ensure};

/// Feature-wise prior for the ordinal predictor. Both arms use training rows only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeaturePrior {
    Adaptive,
    Uniform,
}

/// Validated statistical choices paired with the fixed native numerical contract.
/// Native artifact schema 4 has exactly five ensemble members.
#[derive(Clone, Debug)]
pub struct ModelRecipe {
    pub(super) statistical: StatisticalRecipe,
    pub(super) numerical: &'static NumericalRecipe,
}
#[derive(Clone, Debug)]
pub(super) struct StatisticalRecipe {
    pub feature_prior: FeaturePrior,
    pub c: f64,
    pub loss: PredictorLoss,
}
#[derive(Clone, Copy, Debug)]
pub(super) enum PredictorLoss {
    BalancedInterceptRidge,
}
impl PredictorLoss {
    fn identity(self) -> &'static str {
        match self {
            Self::BalancedInterceptRidge => "balanced|intercept-ridge",
        }
    }
}

// These are fixed algorithm contracts, not configurable experiment knobs. Their
// constants are shared by execution, validation and canonical recipe encoding.
pub(super) struct PartitionRecipe;
impl PartitionRecipe {
    pub const MEMBERS: usize = 5;
    pub const SEED: &str = "ff-retained-independent-all-v1";
    pub const RECENT: f64 = 0.2;
    pub const MIN_SESSIONS: usize = 2;
}
pub(super) struct SupportRecipe;
impl SupportRecipe {
    pub const RATINGS: usize = 30;
    pub const CALIBRATION: usize = 30;
    pub const OUTCOMES: usize = 5;
    pub const CONTENT: usize = 5;
    pub const ARTIST: usize = 3;
}
pub(super) struct AdaptivePrior;
impl AdaptivePrior {
    pub const ALPHA: f64 = 5.0;
    pub const FLOOR: f64 = 0.25;
    pub const CEILING: f64 = 2.0;
}
pub(super) struct ConditionalDecoder;
impl ConditionalDecoder {
    pub const OWN_RIDGE: f64 = 1.0;
    pub const CROSS_RIDGE: f64 = 25.0;
    pub const OWN_CENTER: f64 = 1.0;
}
#[derive(Debug)]
pub(super) struct NumericalRecipe {
    pub predictor: Lbfgs,
    pub decoder: DecoderNewton,
}
#[derive(Debug)]
pub(super) struct Lbfgs {
    pub iterations: usize,
    pub tolerance: f64,
}
impl Lbfgs {
    pub const HISTORY: usize = 10;
}
#[derive(Debug)]
pub(super) struct DecoderNewton {
    pub iterations: usize,
    pub tolerance: f64,
    pub certificate: f64,
}
/// Shared stable-delta Newton and Armijo numerical policy.
impl NumericalRecipe {
    pub const BACKTRACKS: usize = 50;
    pub const ARMIJO: f64 = 1e-4;
    pub const SHRINK: f64 = 0.5;
    pub const STABLE_DELTA: f64 = 0.5;
    pub const MIN_STEP: f64 = 1e-12;
    pub const MIN_PIVOT: f64 = 1e-18;
    pub const MIN_CURVATURE: f64 = 1e-12;
}
pub(super) const NUMERICAL: NumericalRecipe = NumericalRecipe {
    predictor: Lbfgs {
        iterations: 500,
        tolerance: 1e-5,
    },
    decoder: DecoderNewton {
        iterations: 1000,
        tolerance: 1e-8,
        certificate: 1e-7,
    },
};
impl ModelRecipe {
    pub fn production() -> Self {
        Self {
            statistical: StatisticalRecipe {
                feature_prior: FeaturePrior::Adaptive,
                c: 0.1,
                loss: PredictorLoss::BalancedInterceptRidge,
            },
            numerical: &NUMERICAL,
        }
    }
    pub fn with_feature_prior(mut self, feature_prior: FeaturePrior) -> Self {
        self.statistical.feature_prior = feature_prior;
        self
    }
    pub fn with_c(mut self, c: f64) -> Result<Self> {
        ensure!(
            c.is_finite() && c > 0.0,
            "Predictor C must be finite and positive"
        );
        self.statistical.c = c;
        Ok(self)
    }
    /// Canonical effective recipe text, including native partition and DTO rules.
    pub fn identity(&self) -> String {
        use std::fmt::Write;
        let mut identity = String::with_capacity(512);
        write!(
            identity,
            "rust-ff-v5|artifact=4,snake_case,raw-predictor,ensemble-only|popularity=signed-log1p-score,f64-before-abs,unweighted-training-population-zscore,welford,zero-variance-scale1,unrestricted-unit-ridge,append-after-tags|partition=session-post-prefixed,max-updated,rust-string-ties,ceil-recent,whole-groups,retain-oldest,input-order,single-group-all-training|min-sessions={}|members={}|seed={}|recent={}|calrows={}|outcomes={}|minratings={}|content={}|artist={}|C={}|",
            PartitionRecipe::MIN_SESSIONS, PartitionRecipe::MEMBERS, PartitionRecipe::SEED, PartitionRecipe::RECENT,
            SupportRecipe::CALIBRATION, SupportRecipe::OUTCOMES, SupportRecipe::RATINGS,
            SupportRecipe::CONTENT, SupportRecipe::ARTIST, self.statistical.c,
        ).expect("Writing a recipe into a String is infallible");
        match self.statistical.feature_prior {
            FeaturePrior::Adaptive => write!(
                identity,
                "alpha={}|gamma=1|floor={}|ceiling={}",
                AdaptivePrior::ALPHA,
                AdaptivePrior::FLOOR,
                AdaptivePrior::CEILING,
            )
            .expect("Writing a recipe into a String is infallible"),
            FeaturePrior::Uniform => identity.push_str("feature-prior=uniform"),
        }
        write!(
            identity,
            "|{}|lbfgs={},{},{:e}|independent-all={},{},own-nonnegative|decoder={},{:e},{:e}|f64-train|f32-serve",
            self.statistical.loss.identity(),
            Lbfgs::HISTORY, self.numerical.predictor.iterations, self.numerical.predictor.tolerance,
            ConditionalDecoder::OWN_RIDGE, ConditionalDecoder::CROSS_RIDGE,
            self.numerical.decoder.iterations, self.numerical.decoder.tolerance, self.numerical.decoder.certificate,
        ).expect("Writing a recipe into a String is infallible");
        identity
    }
}

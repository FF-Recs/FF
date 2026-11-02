use super::*;
use crate::catalog::Tag;

fn gallery(
    corpus: &Corpus,
    scores: &ScoreBuffers,
    familiarity: &Familiarity,
    grades: &[Option<Grade>],
    facet: &str,
    size: usize,
    candidates: &[usize],
    model_id: &str,
) -> Result<Vec<(usize, f32)>> {
    super::gallery(
        corpus,
        Some(scores),
        familiarity,
        grades,
        facet,
        size,
        &[],
        candidates,
        model_id,
        &|| false,
    )
}

fn auto_page(
    corpus: &Corpus,
    scores: &ScoreBuffers,
    familiarity: &Familiarity,
    grades: &[Option<Grade>],
    candidates: &[usize],
    emitted: &HashSet<usize>,
    recent_rows: &[usize],
    preset: &str,
    size: usize,
    seed: i32,
    epoch: u32,
) -> Result<AutoPlan> {
    super::auto_page(
        corpus,
        Some(scores),
        familiarity,
        grades,
        candidates,
        emitted,
        recent_rows,
        preset,
        size,
        seed,
        epoch,
        &|_| true,
        &|| false,
    )
}

fn corpus(tags: &[Vec<i32>], tag_count: usize, artists: &[i32], metadata: &[i32]) -> Corpus {
    let ids: Vec<_> = (1..=tags.len() as i64).collect();
    Corpus::fixture(&ids, tags, tag_count, artists, metadata)
}

fn scores(expected: &[f32], like: &[f32], love: &[f32]) -> ScoreBuffers {
    ScoreBuffers {
        not_dislike: vec![1.0; expected.len()],
        like_or_love: like.to_vec(),
        love: love.to_vec(),
        expected_grade: expected.to_vec(),
    }
}

fn rating(post_id: i64, grade: Grade) -> StoredRating {
    StoredRating {
        post_id,
        grade,
        first_rated_at: 0.0,
        updated_at: 0.0,
        revision: 1,
        context_json: "{}".into(),
    }
}

#[test]
fn starter_sample_is_bounded_weighted_and_includes_negative_popularity() {
    let corpus = corpus(&[vec![], vec![], vec![]], 0, &[], &[]).with_popularities(&[
        -2.0_f64.ln(),
        0.0,
        2.0_f64.ln(),
    ]);
    let mut counts = [0; 3];
    for seed in 0..256 {
        let (selected, eligible, admissible) = starter_sample(
            &corpus,
            &[0, 1, 2, 2, 1, 0],
            &HashSet::new(),
            &|_| true,
            seed,
            1,
            &|| false,
        )
        .unwrap();
        assert_eq!((selected.len(), eligible, admissible), (1, 3, 3));
        let (row, utility) = selected[0];
        assert!(utility.is_finite() && (0.0..1.0).contains(&utility));
        counts[row] += 1;
    }
    assert!(counts[0] > 0);
    assert!(counts[0] < counts[1] && counts[1] < counts[2]);
}

#[test]
fn starter_discover_fills_a_narrow_familiar_artist_filter_without_uniqueness() {
    let corpus =
        corpus(&[vec![0], vec![0], vec![0]], 1, &[0], &[]).with_popularities(&[0.0, -1.0, 1.0]);
    let familiarity = Familiarity::build(&corpus, &[rating(1, Grade::Love)]);
    let selected = super::gallery(
        &corpus,
        None,
        &familiarity,
        &[Some(Grade::Love), None, None],
        "discover",
        3,
        &[0],
        &[1, 2],
        "session-a",
        &|| false,
    )
    .unwrap();
    assert_eq!(selected.len(), 2);
    let rows: HashSet<_> = selected.into_iter().map(|(row, _)| row).collect();
    assert_eq!(rows, HashSet::from([1, 2]));
}

#[test]
fn starter_gallery_is_order_independent_prefix_stable_and_session_sensitive() {
    let corpus = corpus(&vec![vec![]; 48], 0, &[], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let candidates: Vec<_> = (0..48).collect();
    let select = |rows: &[usize], prior: &[usize], seed: &str| {
        super::gallery(
            &corpus,
            None,
            &familiarity,
            &[None; 48],
            "discover",
            12,
            prior,
            rows,
            seed,
            &|| false,
        )
        .unwrap()
    };
    let baseline = select(&candidates, &[], "session-a");
    let mut reversed: Vec<_> = candidates.iter().rev().copied().collect();
    reversed.extend_from_slice(&candidates);
    assert_eq!(baseline, select(&reversed, &[], "session-a"));
    let prior: Vec<_> = baseline[..4].iter().map(|&(row, _)| row).collect();
    assert_eq!(baseline[4..], select(&reversed, &prior, "session-a"));
    assert_ne!(baseline, select(&candidates, &[], "session-b"));
}

#[test]
fn starter_uses_public_score_ratios_without_a_popularity_cutoff() {
    let corpus = corpus(&[vec![], vec![]], 0, &[], &[])
        .with_popularities(&[100.0_f64.ln_1p(), 10000.0_f64.ln_1p()]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let mut high_score_wins = 0;
    for seed in 0..1024 {
        let page = super::auto_page(
            &corpus,
            None,
            &familiarity,
            &[None; 2],
            &[0, 1],
            &HashSet::new(),
            &[],
            "balanced",
            1,
            seed,
            0,
            &|_| true,
            &|| false,
        )
        .unwrap();
        high_score_wins += usize::from(page.rows[0] == 1);
    }
    // Public weights are 101 versus 10001, not two similar log weights.
    // Fixed seeds and a wide margin distinguish that bias from flattening.
    assert!(high_score_wins >= 960, "{high_score_wins}");
    assert!(
        high_score_wins < 1024,
        "Lower-popularity posts retain a chance"
    );
}

#[test]
fn starter_sampling_ignores_content_and_artist_exposure() {
    let plain = corpus(&vec![vec![]; 17], 4, &[2, 3], &[]);
    let tags: Vec<_> = (0..17).map(|row| vec![row % 2, 2 + row % 2]).collect();
    let tagged = corpus(&tags, 4, &[2, 3], &[]);
    let plain_familiarity = Familiarity::build(&plain, &[]);
    let tagged_familiarity = Familiarity::build(&tagged, &[]);
    let candidates: Vec<_> = (1..17).collect();
    for seed in 0..32 {
        let select = |corpus: &Corpus, familiarity: &Familiarity| {
            super::auto_page(
                corpus,
                None,
                familiarity,
                &[None; 17],
                &candidates,
                &HashSet::new(),
                &[0],
                "balanced",
                5,
                seed,
                0,
                &|_| true,
                &|| false,
            )
            .unwrap()
        };
        let baseline = select(&plain, &plain_familiarity);
        let changed = select(&tagged, &tagged_familiarity);
        assert_eq!(baseline.rows, changed.rows);
        assert_eq!(baseline.scores, changed.scores);
        let session = format!("session-{seed}");
        let select_gallery = |corpus: &Corpus, familiarity: &Familiarity| {
            super::gallery(
                corpus,
                None,
                familiarity,
                &[None; 17],
                "discover",
                6,
                &[0],
                &candidates,
                &session,
                &|| false,
            )
            .unwrap()
        };
        assert_eq!(
            select_gallery(&plain, &plain_familiarity),
            select_gallery(&tagged, &tagged_familiarity),
        );
    }
}

#[test]
fn starter_model_facets_require_scores_even_when_no_new_rows_remain() {
    let corpus = corpus(&[vec![]], 0, &[], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    for facet in ["q3_probe", "rating_review"] {
        let error = super::gallery(
            &corpus,
            None,
            &familiarity,
            &[None],
            facet,
            0,
            &[],
            &[],
            "session",
            &|| false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("Ensemble"));
    }
}

#[test]
fn starter_auto_excludes_known_negatives_and_keeps_fresh_familiarity_explicit() {
    let corpus = corpus(
        &[
            vec![0],
            vec![0, 1],
            vec![1],
            vec![2],
            vec![3],
            vec![],
            vec![4],
        ],
        5,
        &[0, 1, 2, 3, 4],
        &[],
    );
    let familiarity = Familiarity::build(
        &corpus,
        &[
            rating(1, Grade::Love),
            rating(4, Grade::Meh),
            rating(5, Grade::Dislike),
        ],
    );
    let grades = [
        Some(Grade::Love),
        None,
        None,
        Some(Grade::Meh),
        Some(Grade::Dislike),
        None,
        None,
    ];
    let emitted = HashSet::from([6]);
    let select = |preset| {
        super::auto_page(
            &corpus,
            None,
            &familiarity,
            &grades,
            &[0, 1, 2, 3, 4, 5, 6],
            &emitted,
            &[],
            preset,
            10,
            17,
            0,
            &|row| row != 5,
            &|| false,
        )
        .unwrap()
    };
    let balanced = select("balanced");
    assert_eq!(
        balanced.rows.iter().copied().collect::<HashSet<_>>(),
        HashSet::from([0, 1, 2])
    );
    assert_eq!(balanced.scores, vec![0.5; 3]);
    let fresh = select("fresh");
    assert_eq!(fresh.rows, vec![2]);
    assert!(!fresh.exhausted);
    for preset in ["favorites", "balanced", "fresh"] {
        assert!(auto_eligible(preset, None, None));
        assert!(auto_eligible(preset, Some(Grade::Like), None));
        assert!(!auto_eligible(preset, Some(Grade::Meh), None));
        assert!(!auto_eligible(preset, Some(Grade::Dislike), None));
    }
    assert!(!auto_eligible("unknown", None, None));
}

#[test]
fn starter_auto_is_repeatable_deduplicated_and_varies_by_seed_and_epoch() {
    let corpus = corpus(&vec![vec![]; 48], 0, &[], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let candidates: Vec<_> = (0..48).collect();
    let emitted = HashSet::from([0]);
    let select = |rows: &[usize], seed, epoch| {
        super::auto_page(
            &corpus,
            None,
            &familiarity,
            &[None; 48],
            rows,
            &emitted,
            &[],
            "balanced",
            12,
            seed,
            epoch,
            &|_| true,
            &|| false,
        )
        .unwrap()
    };
    let baseline = select(&candidates, 17, 0);
    let mut reordered: Vec<_> = candidates.iter().rev().copied().collect();
    reordered.extend_from_slice(&candidates);
    assert_eq!(baseline.rows, select(&reordered, 17, 0).rows);
    assert!(!baseline.rows.contains(&0));
    assert_eq!(baseline.rows.iter().collect::<HashSet<_>>().len(), 12);
    assert_ne!(baseline.rows, select(&candidates, 18, 0).rows);
    assert_ne!(baseline.rows, select(&candidates, 17, 1).rows);
}

#[test]
fn median_uses_inclusive_cumulative_probability_boundaries() {
    let scores = ScoreBuffers {
        not_dislike: vec![0.49, 0.5, 0.8, 0.9],
        like_or_love: vec![0.2, 0.2, 0.5, 0.7],
        love: vec![0.1, 0.1, 0.1, 0.5],
        expected_grade: vec![0.0; 4],
    };
    assert_eq!(
        (0..4)
            .map(|row| median_grade(&scores, row))
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn aliases_canonicalize_deduplicate_match_and_detect_contradictions() {
    let catalog = TagCatalog::fixture(
        vec![
            Tag {
                name: "anthro".into(),
                category: 0,
                post_count: 10,
            },
            Tag {
                name: "red_fox".into(),
                category: 5,
                post_count: 5,
            },
            Tag {
                name: "wolf".into(),
                category: 5,
                post_count: 8,
            },
        ],
        vec![("vulpine".into(), 1)],
    )
    .unwrap();
    let filter = TagFilter::parse(&catalog, "WOLF +vulpine -anthro wolf").unwrap();
    assert_eq!(filter.canonical, "+red_fox +wolf -anthro");
    assert_eq!(filter.required, vec![1, 2]);
    assert_eq!(filter.excluded, vec![0]);
    let corpus = corpus(&[vec![1, 2], vec![0, 1, 2], vec![2], vec![]], 3, &[], &[]);
    assert!(filter.matches(&corpus, 0));
    assert!(!filter.matches(&corpus, 1));
    assert!(!filter.matches(&corpus, 2));
    assert!(!filter.matches(&corpus, 3));
    assert!(
        TagFilter::parse(&catalog, "vulpine -red_fox")
            .unwrap_err()
            .to_string()
            .contains("both requires and excludes: red_fox")
    );
    assert!(
        TagFilter::parse(&catalog, "+ - absent absent")
            .unwrap_err()
            .to_string()
            .contains("Unknown tag(s): +, -, absent.")
    );
    let empty = TagFilter::parse(&catalog, " \t\r\n").unwrap();
    assert!(empty.canonical.is_empty());
    assert!(empty.matches(&corpus, 3));
}

#[test]
fn familiarity_uses_all_coartists_max_exposure_and_mean_sentiment() {
    let corpus = corpus(
        &[vec![0, 1, 2], vec![0, 1], vec![0, 2], vec![0]],
        3,
        &[1, 2],
        &[],
    );
    let familiarity = Familiarity::build(
        &corpus,
        &[
            rating(1, Grade::Love),
            rating(2, Grade::Dislike),
            rating(900, Grade::Like),
        ],
    );
    assert_eq!(familiarity.rated_count, vec![2, 1]);
    assert_eq!(familiarity.weighted_sentiment, vec![0.5, 2.0]);
    assert_eq!(familiarity.post_rated_count, vec![2, 2, 1, 0]);
    assert_eq!(
        familiarity.post_weighted_sentiment,
        vec![1.25, 0.5, 2.0, 0.0]
    );
    assert_eq!(familiarity.post_has_artist, vec![true, true, true, false]);
}

#[test]
fn incremental_familiarity_matches_rebuild_after_new_corrected_and_repeated_votes() {
    let corpus = corpus(
        &[
            vec![0, 1, 2],
            vec![0, 1],
            vec![0, 2, 3],
            vec![0, 3],
            vec![0, 4],
            vec![0],
        ],
        5,
        &[1, 2, 3, 4],
        &[],
    );
    let mut ratings = vec![
        rating(1, Grade::Love),
        rating(2, Grade::Like),
        rating(3, Grade::Meh),
    ];
    let mut incremental = Familiarity::build(&corpus, &ratings);
    let changes = [
        (0, Grade::Dislike),
        (3, Grade::Love),
        (2, Grade::Like),
        (1, Grade::Meh),
        (0, Grade::Meh),
        (0, Grade::Meh),
        (4, Grade::Dislike),
        (5, Grade::Love),
        (0, Grade::Like),
        (0, Grade::Love),
        (3, Grade::Dislike),
        (2, Grade::Love),
    ];
    for (row, next) in changes {
        let id = corpus.post_ids()[row];
        let previous = ratings
            .iter()
            .find(|rating| rating.post_id == id)
            .map(|rating| rating.grade);
        incremental.update(&corpus, row, previous, next);
        if let Some(rating) = ratings.iter_mut().find(|rating| rating.post_id == id) {
            rating.grade = next;
        } else {
            ratings.push(rating(id, next));
        }
        let rebuilt = Familiarity::build(&corpus, &ratings);
        assert_eq!(incremental.rated_count, rebuilt.rated_count);
        assert_eq!(incremental.sentiment_total, rebuilt.sentiment_total);
        assert_eq!(incremental.weighted_sentiment, rebuilt.weighted_sentiment);
        assert_eq!(incremental.post_rated_count, rebuilt.post_rated_count);
        assert_eq!(
            incremental.post_weighted_sentiment,
            rebuilt.post_weighted_sentiment
        );
        assert_eq!(incremental.post_has_artist, rebuilt.post_has_artist);
    }
}

#[test]
fn discovery_and_development_choose_best_credit_set_and_stable_ties() {
    let corpus = corpus(
        &[
            vec![0, 1],
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![0, 4],
            vec![0],
        ],
        5,
        &[1, 2, 3, 4],
        &[],
    );
    let mut familiarity = Familiarity::build(&corpus, &[]);
    familiarity.post_rated_count = vec![0, 0, 1, 2, 3, 0];
    let scores = scores(&[2.5, 2.5, 2.0, 2.8, 3.0, 3.0], &[0.9; 6], &[0.5; 6]);
    let candidates = [5, 4, 3, 2, 1, 0];
    let discover = gallery(
        &corpus,
        &scores,
        &familiarity,
        &[None; 6],
        "discover",
        10,
        &candidates,
        "model",
    )
    .unwrap();
    assert_eq!(discover, vec![(0, 2.5)]);
    let develop = gallery(
        &corpus,
        &scores,
        &familiarity,
        &[None; 6],
        "develop",
        10,
        &candidates,
        "model",
    )
    .unwrap();
    assert_eq!(develop, vec![(3, 2.8), (2, 2.0)]);
}

#[test]
fn semantic_diversity_ignores_artist_and_ordinal_metadata_tags() {
    let corpus = corpus(
        &[vec![0, 2, 3], vec![0, 2, 3], vec![1, 2, 3]],
        4,
        &[2],
        &[3],
    );
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[2.0, 2.0, 2.0], &[0.8; 3], &[0.5, 0.49, 0.4]);
    let selected = gallery(
        &corpus,
        &scores,
        &familiarity,
        &[None; 3],
        "q3_probe",
        2,
        &[0, 1, 2],
        "model",
    )
    .unwrap();
    assert_eq!(
        selected.iter().map(|item| item.0).collect::<Vec<_>>(),
        vec![0, 2]
    );
    assert_eq!(
        similarity(&[false, false, true, true], &[2, 3], &[2, 3]),
        0.0
    );
}

#[test]
fn train_broadly_returns_only_the_interpolated_quality_decile() {
    let tags = vec![vec![0]; 12];
    let corpus = corpus(&tags, 1, &[], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let expected: Vec<_> = (0..12).map(|i| i as f32 / 4.0).collect();
    let scores = scores(&expected, &[0.8; 12], &[0.2; 12]);
    let candidates: Vec<_> = (0..12).collect();
    let mut selected = gallery(
        &corpus,
        &scores,
        &familiarity,
        &[None; 12],
        "train_broadly",
        20,
        &candidates,
        "generation",
    )
    .unwrap();
    selected.sort_unstable_by_key(|&(row, _)| row);
    assert_eq!(selected, vec![(10, 2.5), (11, 2.75)]);
}

#[test]
fn train_broadly_is_repeatable_and_uniform_score_duplicates_do_not_change_the_gate() {
    let corpus = corpus(&vec![vec![]; 48], 0, &[], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[2.0; 48], &[0.8; 48], &[0.2; 48]);
    let candidates: Vec<_> = (0..48).collect();
    let select = |candidates: &[usize], size| {
        gallery(
            &corpus,
            &scores,
            &familiarity,
            &[None; 48],
            "train_broadly",
            size,
            candidates,
            "generation",
        )
        .unwrap()
    };
    let selected = select(&candidates, 48);
    assert_eq!(selected, select(&candidates, 48));
    let mut offered_rows: Vec<_> = selected.iter().map(|&(row, _)| row).collect();
    offered_rows.sort_unstable();
    assert_eq!(offered_rows, candidates);
    let reversed_duplicates: Vec<_> = candidates
        .iter()
        .rev()
        .flat_map(|&row| [row, row, row])
        .collect();
    assert_eq!(selected, select(&reversed_duplicates, 48));

    // With an unchanged quality gate and no content-diversity penalty,
    // excluding other candidates cannot change the remaining priorities.
    let filtered: Vec<_> = candidates
        .iter()
        .copied()
        .filter(|row| row % 3 != 0)
        .collect();
    let expected: Vec<_> = selected
        .into_iter()
        .filter(|&(row, _)| row % 3 != 0)
        .take(9)
        .collect();
    assert_eq!(select(&filtered, 9), expected);
}

#[test]
fn train_broadly_matches_independent_fixed_domain_and_full_id_vectors() {
    // Independently derived with Python hashlib and integer WyRand v4.2:
    // s=(seed+0x2d358dccaa6c78a5) mod 2^64;
    // t=s*(s XOR 0x8bb84b93962eacc9); word=low64(t) XOR high64(t);
    // priority=float(word >> 1) / 2^63, matching locked fastrand 2.5.
    // Keep this domain literal independent of delivery's GALLERY_POLICY.
    let model = "rust-ensemble-v5-vector-\u{1f98a}";
    let digest = Sha256::digest(format!("gallery-rust-v2{model}").as_bytes());
    assert_eq!(
        format!("{digest:x}"),
        "a7890f679c604f13899827ead4fec5c4c671aa8f981d92a63741629fda7fa31a"
    );
    let ids = [
        1,
        2,
        17,
        4294967297,
        1099511627783,
        9007199254740993,
        4611686018427387907,
        i64::MAX,
    ];
    let priority_bits = [
        0x3feab5b78c6e139a,
        0x3fda3979377fa967,
        0x3fe232dcd5206d7b,
        0x3fdbd4c22abf9121,
        0x3fd30f9ccc132fd6,
        0x3fd78fc2da61b3e0,
        0x3fa56b9a372c74a3,
        0x3fd4cd8146e89c6f,
    ];
    for (&id, expected) in ids.iter().zip(priority_bits) {
        assert_eq!(
            fastrand::Rng::with_seed(0x134f609c670f89a7 ^ id as u64)
                .f64()
                .to_bits(),
            expected
        );
    }
    let corpus = Corpus::fixture(&ids, &vec![vec![]; ids.len()], 0, &[], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[2.0; 8], &[0.8; 8], &[0.2; 8]);
    let selected = gallery(
        &corpus,
        &scores,
        &familiarity,
        &[None; 8],
        "train_broadly",
        8,
        &[7, 1, 4, 0, 6, 2, 5, 3],
        model,
    )
    .unwrap();
    // Empty tag sets remove diversity penalties; all scores pass the
    // quality gate, so the real consumer must follow frozen priorities.
    assert_eq!(
        selected,
        vec![
            (0, 2.0),
            (2, 2.0),
            (3, 2.0),
            (1, 2.0),
            (5, 2.0),
            (7, 2.0),
            (4, 2.0),
            (6, 2.0)
        ]
    );
}

#[test]
fn train_broadly_priorities_follow_full_post_ids_and_model_not_corpus_rows() {
    let select = |ids: &[i64], candidates: &[usize], model_id: &str| {
        let corpus = Corpus::fixture(ids, &vec![vec![]; ids.len()], 0, &[], &[]);
        let familiarity = Familiarity::build(&corpus, &[]);
        let expected = vec![2.0; ids.len()];
        let scores = scores(&expected, &vec![0.8; ids.len()], &vec![0.2; ids.len()]);
        gallery(
            &corpus,
            &scores,
            &familiarity,
            &vec![None; ids.len()],
            "train_broadly",
            candidates.len(),
            candidates,
            model_id,
        )
        .unwrap()
        .into_iter()
        .map(|(row, _)| ids[row])
        .collect::<Vec<_>>()
    };
    let ids: Vec<_> = (2..=33).collect();
    let candidates: Vec<_> = (0..ids.len()).collect();
    let selected = select(&ids, &candidates, "generation");
    assert_ne!(selected, select(&ids, &candidates, "another-generation"));

    let high_bit = 1_i64 << 40;
    let high_ids: Vec<_> = ids.iter().map(|id| id + high_bit).collect();
    let high_selected: Vec<_> = select(&high_ids, &candidates, "generation")
        .into_iter()
        .map(|id| id - high_bit)
        .collect();
    assert_ne!(selected, high_selected);

    let inserted_ids: Vec<_> = (1..=33).collect();
    let shifted_candidates: Vec<_> = (1..inserted_ids.len()).collect();
    assert_eq!(
        selected,
        select(&inserted_ids, &shifted_candidates, "generation")
    );
}

#[test]
fn review_uses_expected_absolute_grade_disagreement_for_priority_and_score() {
    let corpus = corpus(&[vec![], vec![]], 0, &[], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = ScoreBuffers {
        not_dislike: vec![0.9; 2],
        like_or_love: vec![0.7; 2],
        love: vec![0.1; 2],
        expected_grade: vec![1.7; 2],
    };
    let selected = gallery(
        &corpus,
        &scores,
        &familiarity,
        &[Some(Grade::Love), Some(Grade::Like)],
        "rating_review",
        2,
        &[1, 0],
        "m",
    )
    .unwrap();
    assert_eq!(selected[0].0, 0);
    assert!((selected[0].1 - 1.3).abs() < 1e-6);
    assert!((selected[1].1 - 0.5).abs() < 1e-6);
}

#[test]
fn gallery_fixed_input_prefixes_keep_order_scores_and_cohort_capacity() {
    // All tags are artists, so the semantic facets have no repetition
    // penalty. Equal expected grades also leave Train Broadly's gate fixed.
    let tags: Vec<_> = (0..8).map(|tag| vec![tag]).collect();
    let artists: Vec<_> = (0..8).collect();
    let corpus = corpus(&tags, 8, &artists, &[]);
    let love = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375];
    let like: Vec<_> = love.iter().map(|p| 1.0 - p).collect();
    let scores = scores(&[2.0; 8], &like, &love);
    let candidates: Vec<_> = (0..8).collect();
    for facet in [
        "discover",
        "develop",
        "train_broadly",
        "q3_probe",
        "rating_review",
    ] {
        let mut familiarity = Familiarity::build(&corpus, &[]);
        if facet == "develop" {
            familiarity.post_rated_count.fill(1);
        }
        let grades = [Some(Grade::Love); 8];
        let first = gallery(
            &corpus,
            &scores,
            &familiarity,
            &grades,
            facet,
            6,
            &candidates,
            "generation",
        )
        .unwrap();
        assert_eq!(first.len(), 6, "{facet}");
        let rows: Vec<_> = first.iter().map(|&(row, _)| row).collect();
        match facet {
            "discover" | "develop" | "rating_review" => {
                assert_eq!(rows, vec![0, 1, 2, 3, 4, 5], "{facet}");
            }
            "q3_probe" => assert_eq!(rows, vec![7, 6, 5, 4, 3, 2]),
            _ => {}
        }
        for &(_, value) in &first {
            let expected = if facet == "rating_review" { 1.0 } else { 2.0 };
            assert!((value - expected).abs() < 1e-6, "{facet}: {value}");
        }
        for prefix_len in 1..=first.len() {
            let issued: Vec<_> = first[..prefix_len].iter().map(|&(row, _)| row).collect();
            // Issued identities may remain in the caller's candidates or
            // disappear from eligibility; neither may emit them again.
            for candidates in [
                candidates.iter().rev().copied().collect::<Vec<_>>(),
                candidates
                    .iter()
                    .copied()
                    .filter(|row| !issued.contains(row))
                    .collect(),
            ] {
                let rest = super::gallery(
                    &corpus,
                    Some(&scores),
                    &familiarity,
                    &grades,
                    facet,
                    6,
                    &issued,
                    &candidates,
                    "generation",
                    &|| false,
                )
                .unwrap();
                assert_eq!(rest, first[prefix_len..], "{facet}: prefix {prefix_len}");
            }
        }
    }
}

#[test]
fn gallery_reserves_every_credit_and_bounds_multiartist_cohort_suffixes() {
    let corpus = corpus(
        &[
            vec![0, 1],
            vec![0],
            vec![1],
            vec![2, 3],
            vec![2],
            vec![3],
            vec![3],
            vec![3],
            vec![4, 5],
            vec![4],
            vec![5],
            vec![6],
        ],
        7,
        &[0, 1, 2, 3, 4, 5, 6],
        &[],
    );
    let scores = scores(
        &[3.0, 2.9, 2.8, 2.95, 2.0, 1.0, 1.0, 1.0, 2.7, 2.6, 2.5, 1.9],
        &[0.8; 12],
        &[0.5; 12],
    );
    for facet in ["discover", "develop"] {
        let mut ratings = vec![
            rating(6, Grade::Like),
            rating(7, Grade::Like),
            rating(8, Grade::Like),
        ];
        if facet == "develop" {
            ratings.extend([
                rating(1, Grade::Like),
                rating(5, Grade::Like),
                rating(9, Grade::Like),
                rating(12, Grade::Like),
            ]);
        }
        let familiarity = Familiarity::build(&corpus, &ratings);
        // The familiar second credit vetoes row 3 even though its other
        // artist is unseen (Discover) or only seen once (Develop).
        assert_eq!(familiarity.post_rated_count[3], 3);
        let candidates = [0, 1, 2, 3, 4, 8, 9, 10, 11];
        let full = gallery(
            &corpus,
            &scores,
            &familiarity,
            &[None; 12],
            facet,
            4,
            &candidates,
            "generation",
        )
        .unwrap();
        assert_eq!(
            full,
            vec![(0, 3.0), (8, 2.7), (4, 2.0), (11, 1.9)],
            "{facet}"
        );
        for prefix_len in 1..=full.len() {
            let issued: Vec<_> = full[..prefix_len].iter().map(|&(row, _)| row).collect();
            let reversed: Vec<_> = candidates.iter().rev().copied().collect();
            let suffix = super::gallery(
                &corpus,
                Some(&scores),
                &familiarity,
                &[None; 12],
                facet,
                4,
                &issued,
                &reversed,
                "generation",
                &|| false,
            )
            .unwrap();
            assert_eq!(suffix, full[prefix_len..], "{facet}: prefix {prefix_len}");
        }
    }
}

#[test]
fn gallery_overlapping_high_scores_do_not_hide_distinct_available_artists() {
    let mut tags: Vec<_> = (1..=6000).map(|artist| vec![0, artist]).collect();
    tags.push(vec![6001]);
    let artists: Vec<_> = (0..=6001).collect();
    let corpus = corpus(&tags, artists.len(), &artists, &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let mut expected = vec![3.0; tags.len()];
    expected[6000] = 2.0;
    let scores = scores(&expected, &vec![0.9; tags.len()], &vec![0.5; tags.len()]);
    let candidates: Vec<_> = (0..tags.len()).rev().collect();
    assert_eq!(
        gallery(
            &corpus,
            &scores,
            &familiarity,
            &vec![None; tags.len()],
            "discover",
            2,
            &candidates,
            "generation",
        )
        .unwrap(),
        vec![(0, 3.0), (6000, 2.0)],
    );
}

#[test]
fn gallery_diversity_uses_every_prior_row_but_not_artist_or_metadata_overlap() {
    let corpus = corpus(
        &[
            vec![0, 3, 4],
            vec![1, 3, 4],
            vec![0],
            vec![1],
            vec![2, 3, 4],
            vec![2],
        ],
        5,
        &[3],
        &[4],
    );
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[2.5; 6], &[1.0; 6], &[0.5; 6]);
    let grades = [Some(Grade::Love); 6];
    for facet in ["q3_probe", "rating_review", "train_broadly"] {
        for candidates in [&[0, 1, 2, 3, 4, 5][..], &[2, 3, 4, 5][..]] {
            let selected = super::gallery(
                &corpus,
                Some(&scores),
                &familiarity,
                &grades,
                facet,
                4,
                &[0, 1],
                candidates,
                "generation",
                &|| false,
            )
            .unwrap();
            assert_eq!(selected.len(), 2, "{facet}");
            if facet == "train_broadly" {
                assert!(matches!(selected[0].0, 4 | 5));
                assert!(selected.iter().all(|&(_, score)| score == 2.5));
            } else {
                let score = if facet == "rating_review" { 0.5 } else { 2.5 };
                assert_eq!(selected, vec![(4, score), (2, score)], "{facet}");
            }
        }
    }
}

#[test]
fn gallery_rejects_out_of_corpus_prefix_even_with_no_remaining_candidates() {
    let corpus = corpus(&[vec![]], 0, &[], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[2.5], &[1.0], &[0.5]);
    let error = super::gallery(
        &corpus,
        Some(&scores),
        &familiarity,
        &[None],
        "q3_probe",
        1,
        &[1],
        &[],
        "generation",
        &|| false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("issued row outside corpus"));
}

#[test]
fn auto_inclusive_probability_floors_never_override_known_grades() {
    for (preset, floor) in [("favorites", 0.8), ("balanced", 0.7), ("fresh", 0.6)] {
        assert!(auto_eligible(preset, None, Some(floor)));
        assert!(!auto_eligible(preset, None, Some(floor - 0.001)));
        assert!(!auto_eligible(preset, None, Some(f32::NAN)));
        for grade in [Grade::Like, Grade::Love] {
            assert!(auto_eligible(preset, Some(grade), Some(0.0)));
        }
        for grade in [Grade::Meh, Grade::Dislike] {
            assert!(!auto_eligible(preset, Some(grade), Some(1.0)));
        }
    }
}

#[test]
fn row_emissions_survive_candidate_reorder_and_shrink() {
    let corpus = corpus(&[vec![], vec![], vec![], vec![]], 0, &[], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[2.7; 4], &[0.9; 4], &[0.5; 4]);
    let emitted = HashSet::from([0, 2]);
    for candidates in [vec![0, 1, 2, 3], vec![3, 2, 1, 0], vec![3, 0, 1]] {
        let page = auto_page(
            &corpus,
            &scores,
            &familiarity,
            &[None; 4],
            &candidates,
            &emitted,
            &[],
            "balanced",
            4,
            42,
            0,
        )
        .unwrap();
        assert_eq!(
            page.rows.iter().copied().collect::<HashSet<_>>(),
            HashSet::from([1, 3])
        );
        // Unheld old emissions remain replayable after bounded rollover.
        assert!(!page.exhausted);
    }
}

#[test]
fn every_auto_page_rechecks_current_probabilities_negatives_and_row_emissions() {
    let corpus = corpus(&[vec![0], vec![0], vec![0], vec![0], vec![0]], 1, &[0], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let mut scores = scores(&[2.7; 5], &[0.9; 5], &[0.5; 5]);
    let mut grades = [None; 5];
    let candidates = [4, 3, 2, 1, 0];
    let emitted = HashSet::from([2]);
    grades[0] = Some(Grade::Dislike);
    grades[1] = Some(Grade::Meh);
    scores.like_or_love[3] = 0.0;
    for preset in ["favorites", "balanced", "fresh"] {
        let page = auto_page(
            &corpus,
            &scores,
            &familiarity,
            &grades,
            &candidates,
            &emitted,
            &[],
            preset,
            5,
            42,
            1,
        )
        .unwrap();
        assert_eq!(page.rows, vec![4]);
        assert!(!page.rows.iter().any(|row| emitted.contains(row)));
    }
    grades[3] = Some(Grade::Like);
    for preset in ["favorites", "balanced"] {
        let page = auto_page(
            &corpus,
            &scores,
            &familiarity,
            &grades,
            &candidates,
            &emitted,
            &[],
            preset,
            5,
            42,
            1,
        )
        .unwrap();
        assert!(page.rows.contains(&3));
    }
    let fresh = auto_page(
        &corpus,
        &scores,
        &familiarity,
        &grades,
        &candidates,
        &emitted,
        &[],
        "fresh",
        5,
        42,
        1,
    )
    .unwrap();
    assert_eq!(fresh.rows, vec![4]);
}

#[test]
fn fresh_rechecks_unrated_artist_and_coartist_exposure_each_page() {
    let corpus = corpus(
        &[vec![0, 1], vec![0, 1, 2], vec![0, 2], vec![0, 3], vec![0]],
        4,
        &[1, 2, 3],
        &[],
    );
    let scores = scores(&[2.7; 5], &[0.9; 5], &[0.5; 5]);
    let initial = Familiarity::build(&corpus, &[]);
    let before = auto_page(
        &corpus,
        &scores,
        &initial,
        &[None; 5],
        &[0, 1, 2, 3, 4],
        &HashSet::new(),
        &[],
        "fresh",
        5,
        1,
        0,
    )
    .unwrap();
    assert_eq!(before.rows.len(), 4);
    let updated = Familiarity::build(&corpus, &[rating(3, Grade::Like)]);
    let mut grades = [None; 5];
    grades[2] = Some(Grade::Like);
    let after = auto_page(
        &corpus,
        &scores,
        &updated,
        &grades,
        &[0, 1, 2, 3, 4],
        &HashSet::new(),
        &[],
        "fresh",
        5,
        1,
        0,
    )
    .unwrap();
    let selected: HashSet<_> = after.rows.into_iter().collect();
    assert_eq!(selected, HashSet::from([0, 3]));
}

#[test]
fn balanced_artist_quota_diversifies_then_relaxes_to_fill() {
    let corpus = corpus(
        &[vec![0, 1], vec![0, 1], vec![0, 2], vec![0, 3]],
        4,
        &[1, 2, 3],
        &[],
    );
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[2.7; 4], &[0.9; 4], &[0.5; 4]);
    let page = auto_page(
        &corpus,
        &scores,
        &familiarity,
        &[None; 4],
        &[0, 1, 2, 3],
        &HashSet::new(),
        &[],
        "balanced",
        4,
        -42,
        3,
    )
    .unwrap();
    let artists: HashSet<_> = page
        .rows
        .iter()
        .take(3)
        .flat_map(|&row| corpus.post_artists(row).iter().copied())
        .collect();
    assert_eq!(artists.len(), 3);
    assert_eq!(page.rows.iter().copied().collect::<HashSet<_>>().len(), 4);
    let replay = auto_page(
        &corpus,
        &scores,
        &familiarity,
        &[None; 4],
        &[0, 1, 2, 3],
        &HashSet::new(),
        &[],
        "balanced",
        4,
        -42,
        3,
    )
    .unwrap();
    assert_eq!(page.rows, replay.rows);
    assert_eq!(page.scores, replay.scores);
}

#[test]
fn fresh_portfolio_averages_only_three_best_current_eligible_rows() {
    let corpus = corpus(&[vec![0], vec![0], vec![0], vec![0], vec![0]], 1, &[0], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[3.0, 2.7, 2.4, 1.2, 0.9], &[0.9; 5], &[0.5; 5]);
    let page = auto_page(
        &corpus,
        &scores,
        &familiarity,
        &[None; 5],
        &[0, 1, 2, 3, 4],
        &HashSet::from([0]),
        &[],
        "fresh",
        5,
        11,
        0,
    )
    .unwrap();
    // Emitted row 0 must not inflate the remaining artist portfolio.
    let portfolio = (2.7_f32 + 2.4 + 1.2) / 3.0;
    for (&row, &value) in page.rows.iter().zip(&page.scores) {
        assert!((value - (0.5 * scores.expected_grade[row] + 0.5 * portfolio) / 3.0).abs() < 1e-6);
    }
}

#[test]
fn known_positive_boosts_are_served_and_favorites_ignores_recent_penalties() {
    let corpus = corpus(&[vec![0], vec![0], vec![0]], 1, &[0], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.9], &[0.0; 3]);
    let grades = [Some(Grade::Love), Some(Grade::Like), None];
    let page = auto_page(
        &corpus,
        &scores,
        &familiarity,
        &grades,
        &[0, 1, 2],
        &HashSet::new(),
        &[],
        "favorites",
        3,
        17,
        0,
    )
    .unwrap();
    for (&row, &value) in page.rows.iter().zip(&page.scores) {
        assert_eq!(value, [0.9, 0.8, 0.0][row]);
    }
    let recent = vec![0; 500];
    let replay = auto_page(
        &corpus,
        &scores,
        &familiarity,
        &grades,
        &[0, 1, 2],
        &HashSet::new(),
        &recent,
        "favorites",
        3,
        17,
        0,
    )
    .unwrap();
    assert_eq!(page.rows, replay.rows);
    assert_eq!(page.scores, replay.scores);
}

#[test]
fn auto_semantic_repetition_does_not_count_ordinal_metadata() {
    let corpus = corpus(&[vec![0], vec![]], 1, &[], &[0]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[2.7; 2], &[0.9; 2], &[0.5; 2]);
    let baseline = auto_page(
        &corpus,
        &scores,
        &familiarity,
        &[None; 2],
        &[0, 1],
        &HashSet::new(),
        &[],
        "balanced",
        2,
        9,
        0,
    )
    .unwrap();
    let recent = vec![0; 500];
    let repeated = auto_page(
        &corpus,
        &scores,
        &familiarity,
        &[None; 2],
        &[0, 1],
        &HashSet::new(),
        &recent,
        "balanced",
        2,
        9,
        0,
    )
    .unwrap();
    assert_eq!(baseline.rows, repeated.rows);
}

#[test]
fn recent_artist_penalties_change_balanced_selection_without_changing_scores() {
    let corpus = corpus(&[vec![0, 1], vec![0, 2]], 3, &[1, 2], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[2.7; 2], &[0.9; 2], &[0.5; 2]);
    let recent = vec![0; 500];
    let page = auto_page(
        &corpus,
        &scores,
        &familiarity,
        &[None; 2],
        &[0, 1],
        &HashSet::new(),
        &recent,
        "balanced",
        1,
        1,
        0,
    )
    .unwrap();
    assert_eq!(page.rows, vec![1]);
    assert_eq!(page.scores, vec![2.7_f32 / 3.0]);
}

#[test]
fn mixed_gumbel_is_finite_signed_seed_stable_and_epoch_sensitive() {
    assert_eq!(mix64(0), 0xE220A8397B1DCDAF);
    for seed in [i32::MIN, -1, 0, i32::MAX] {
        for row in [0, 1, 10000000] {
            let value = gumbel(seed, 0, row);
            assert!(value.is_finite());
            assert_eq!(value, gumbel(seed, 0, row));
            assert_ne!(value, gumbel(seed, 1, row));
        }
    }
}

#[test]
fn selection_rejects_misaligned_state_and_handles_empty_eligible_universe() {
    let corpus = corpus(&[vec![]], 0, &[], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[0.0], &[0.0], &[0.0]);
    assert!(
        auto_page(
            &corpus,
            &scores,
            &familiarity,
            &[],
            &[0],
            &HashSet::new(),
            &[],
            "balanced",
            1,
            0,
            0
        )
        .is_err()
    );
    assert!(
        auto_page(
            &corpus,
            &scores,
            &familiarity,
            &[None],
            &[0],
            &HashSet::new(),
            &[],
            "balanced",
            0,
            0,
            0
        )
        .is_err()
    );
    let empty = auto_page(
        &corpus,
        &scores,
        &familiarity,
        &[None],
        &[0],
        &HashSet::new(),
        &[],
        "balanced",
        1,
        0,
        0,
    )
    .unwrap();
    assert!(empty.rows.is_empty() && empty.scores.is_empty());
    assert!(empty.exhausted);
    assert!(
        gallery(
            &corpus,
            &scores,
            &familiarity,
            &[None],
            "unknown",
            1,
            &[0],
            "m"
        )
        .is_err()
    );
}

#[test]
fn fresh_averages_all_credit_portfolios_without_credit_count_bonus() {
    let corpus = corpus(
        &[vec![0, 1], vec![0], vec![0], vec![1], vec![1], vec![1]],
        2,
        &[0, 1],
        &[],
    );
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(
        &[1.8, 3.0, 2.4, 0.6, 0.0, 3.0],
        &[0.9, 0.9, 0.9, 0.9, 0.5, 0.9],
        &[0.5; 6],
    );
    let page = auto_page(
        &corpus,
        &scores,
        &familiarity,
        &[None; 6],
        &[0, 1, 2, 3, 4, 5],
        &HashSet::from([5]),
        &[],
        "fresh",
        6,
        11,
        0,
    )
    .unwrap();
    assert_eq!(page.rows.len(), 4);
    let value = page.scores[page.rows.iter().position(|&row| row == 0).unwrap()];
    // Credit 0's top-three mean is 2.4; credit 1's admissible,
    // unemitted mean is 1.2. Neither the first credit nor their sum owns it.
    assert!((value - 0.6).abs() < 1e-6);
}

#[test]
fn every_credit_enforces_auto_quota_before_fallback_fills_the_page() {
    let corpus = corpus(&[vec![0, 1], vec![1, 2], vec![3]], 4, &[0, 1, 2, 3], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[3.0, 2.0, 0.0], &[0.9; 3], &[0.5; 3]);
    let seed = (0..1000)
        .find(|&seed| (0..3).all(|row| gumbel(seed, 0, row).abs() < 1.0))
        .unwrap();
    for preset in ["balanced", "fresh"] {
        let page = auto_page(
            &corpus,
            &scores,
            &familiarity,
            &[None; 3],
            &[0, 1, 2],
            &HashSet::new(),
            &[],
            preset,
            3,
            seed,
            0,
        )
        .unwrap();
        // Row 1 shares only row 0's second credit. It must wait until
        // there is no quota-respecting row, but must not be lost.
        assert_eq!(page.rows, vec![0, 2, 1], "{preset}");
        assert!(page.exhausted);
    }
}

#[test]
fn nonfirst_credit_recent_exposure_changes_auto_order_not_scores() {
    let corpus = corpus(&[vec![0, 1], vec![2], vec![1]], 3, &[0, 1, 2], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[2.7; 3], &[0.9; 3], &[0.5; 3]);
    for preset in ["balanced", "fresh"] {
        let page = auto_page(
            &corpus,
            &scores,
            &familiarity,
            &[None; 3],
            &[0, 1],
            &HashSet::new(),
            &vec![2; 500],
            preset,
            1,
            1,
            0,
        )
        .unwrap();
        assert_eq!(page.rows, vec![1], "{preset}");
        assert!((page.scores[0] - 0.9).abs() < 1e-6);
    }
}

#[test]
fn equivalent_additional_credits_do_not_sum_portfolios_or_diversity_penalties() {
    let single = corpus(
        &[vec![0], vec![1], vec![0], vec![1], vec![2], vec![2]],
        3,
        &[0, 1, 2],
        &[],
    );
    let expanded = corpus(
        &[
            vec![0, 3],
            vec![1, 4],
            vec![0, 3],
            vec![1, 4],
            vec![2, 5],
            vec![2, 5],
        ],
        6,
        &[0, 1, 2, 3, 4, 5],
        &[],
    );
    let scores = scores(&[2.8, 2.7, 2.6, 2.5, 2.4, 2.3], &[0.9; 6], &[0.5; 6]);
    // Many seeds exercise the near ties after quota fallback, where summed
    // current-page, recent or adjacent penalties would change the order.
    for preset in ["balanced", "fresh"] {
        for seed in -20..20 {
            let select = |corpus: &Corpus| {
                auto_page(
                    corpus,
                    &scores,
                    &Familiarity::build(corpus, &[]),
                    &[None; 6],
                    &[0, 1, 2, 3, 4, 5],
                    &HashSet::new(),
                    &[0, 0, 1],
                    preset,
                    6,
                    seed,
                    2,
                )
                .unwrap()
            };
            let baseline = select(&single);
            let duplicated = select(&expanded);
            assert_eq!(baseline.rows, duplicated.rows, "{preset}: seed {seed}");
            assert_eq!(baseline.scores, duplicated.scores, "{preset}: seed {seed}");
        }
    }
}

#[test]
fn artist_index_permutation_preserves_gallery_and_auto_results() {
    let original = corpus(
        &[
            vec![0, 1],
            vec![1, 2],
            vec![0],
            vec![2],
            vec![3],
            vec![1, 3],
        ],
        4,
        &[0, 1, 2, 3],
        &[],
    );
    // Rename artist IDs by [2, 0, 3, 1], changing which artist is first.
    let permuted = corpus(
        &[
            vec![0, 2],
            vec![0, 3],
            vec![2],
            vec![3],
            vec![1],
            vec![0, 1],
        ],
        4,
        &[0, 1, 2, 3],
        &[],
    );
    let scores = scores(&[2.8, 2.7, 2.6, 2.5, 2.4, 2.3], &[0.9; 6], &[0.5; 6]);
    for facet in ["discover", "develop"] {
        let select = |corpus: &Corpus| {
            let ratings = if facet == "develop" {
                vec![
                    rating(1, Grade::Like),
                    rating(4, Grade::Like),
                    rating(5, Grade::Like),
                ]
            } else {
                Vec::new()
            };
            super::gallery(
                corpus,
                Some(&scores),
                &Familiarity::build(corpus, &ratings),
                &[None; 6],
                facet,
                4,
                &[0],
                &[5, 4, 3, 2, 1, 0],
                "model",
                &|| false,
            )
            .unwrap()
        };
        assert_eq!(select(&original), select(&permuted), "{facet}");
    }
    for preset in ["favorites", "balanced", "fresh"] {
        for seed in -10..10 {
            let select = |corpus: &Corpus| {
                auto_page(
                    corpus,
                    &scores,
                    &Familiarity::build(corpus, &[]),
                    &[None; 6],
                    &[5, 4, 3, 2, 1, 0],
                    &HashSet::new(),
                    &[1, 1, 5],
                    preset,
                    6,
                    seed,
                    2,
                )
                .unwrap()
            };
            let baseline = select(&original);
            let renamed = select(&permuted);
            assert_eq!(baseline.rows, renamed.rows, "{preset}: seed {seed}");
            assert_eq!(baseline.scores, renamed.scores, "{preset}: seed {seed}");
        }
    }
}

#[test]
fn auto_partial_artist_overlap_has_fractional_page_and_adjacent_penalties() {
    let corpus = corpus(&[vec![0, 1], vec![1], vec![0, 2]], 3, &[0, 1, 2], &[]);
    let familiarity = Familiarity::build(&corpus, &[]);
    let scores = scores(&[3.0, 2.7, 2.4], &[0.9; 3], &[0.5; 3]);
    let seed = (0..10000)
        .find(|&seed| (0..3).all(|row| gumbel(seed, 0, row).abs() < 0.25))
        .unwrap();
    for preset in ["balanced", "fresh"] {
        let page = auto_page(
            &corpus,
            &scores,
            &familiarity,
            &[None; 3],
            &[0, 1, 2],
            &HashSet::new(),
            &[],
            preset,
            3,
            seed,
            0,
        )
        .unwrap();
        // After row 0 both remaining rows require quota fallback. Row 1
        // gets the full 0.06 + 0.24 penalty; row 2 gets half of each.
        // Treating any adjacency as full overlap reverses this choice.
        assert_eq!(page.rows, vec![0, 2, 1], "{preset}");
    }
}

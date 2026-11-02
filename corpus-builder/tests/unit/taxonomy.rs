use std::{fs::File, io::Write};

use flate2::{Compression, write::GzEncoder};

use super::*;

const TAGS: &str =
    "id,name,category\n1,alias_a,0\n2,alias_b,0\n3,fox,0\n4,canine,0\n5,mammal,0\n6,other,0\n";

fn fixture(tags: &str, aliases: &str, implications: &str) -> Result<Taxonomy> {
    let directory = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let paths = ["tags.csv.gz", "aliases.csv.gz", "implications.csv.gz"]
        .map(|name| directory.path().join(name));
    let aliases = format!("antecedent_name,consequent_name,status\n{aliases}");
    let implications = format!("antecedent_name,consequent_name,status\n{implications}");
    for (path, content) in paths.iter().zip([tags, &aliases, &implications]) {
        let mut encoder = GzEncoder::new(File::create(path).unwrap(), Compression::fast());
        encoder.write_all(content.as_bytes()).unwrap();
        encoder.finish().unwrap();
    }
    load(&paths[0], &paths[1], &paths[2])
}

fn assert_memberships(normalizer: &mut Normalizer<'_>, input: &str, expected: &[usize]) {
    let mut actual = normalizer.normalize(input).unwrap().to_vec();
    actual.sort_unstable();
    assert_eq!(actual, expected);
}

#[test]
fn canonicalizes_both_endpoints_counts_rows_and_reuses_dedup_storage() {
    let taxonomy = fixture(
        TAGS,
        "alias_a,alias_b,active\nalias_b,fox,active\nalias_a,alias_b,active\nexternal,alias_a,active\nother,fox,deleted\n\u{10000},fox,active\n\u{e000},fox,active\n",
        "alias_b,canine,active\nfox,canine,active\ncanine,mammal,active\nfox,alias_a,active\nother,alias_b,active\nmissing,also_missing,active\ncanine,fox,deleted\n",
    ).unwrap();
    assert_eq!(taxonomy.canonical, [2, 2, 2, 3, 4, 5]);
    assert_eq!(taxonomy.active_aliases, 6);
    assert_eq!(taxonomy.active_implications, 3);
    assert_eq!(taxonomy.implications[2], [3, 4]);
    assert_eq!(taxonomy.implications[5], [2, 3, 4]);
    assert_eq!(
        taxonomy
            .aliases
            .iter()
            .map(|(name, target)| (name.as_str(), *target))
            .collect::<Vec<_>>(),
        [
            ("alias_a", 2),
            ("alias_b", 2),
            ("external", 2),
            ("\u{e000}", 2),
            ("\u{10000}", 2),
        ]
    );
    let mut normalizer = taxonomy.normalizer();
    assert_memberships(
        &mut normalizer,
        "  alias_a fox canine alias_b  ",
        &[2, 3, 4],
    );
    assert_memberships(&mut normalizer, "other other", &[2, 3, 4, 5]);
    assert!(normalizer.normalize("   ").unwrap().is_empty());
    // Exported aliases without a source tag are not silently accepted on posts.
    assert!(normalizer.normalize("external").is_err());
    assert!(normalizer.normalize("fox\tcanine").is_err());
    assert!(normalizer.normalize("fox unknown").is_err());
    assert_memberships(&mut normalizer, "mammal", &[4]);
    normalizer.generation = u32::MAX;
    normalizer.seen.fill(1);
    assert_memberships(&mut normalizer, "alias_a", &[2, 3, 4]);
}

#[test]
fn rejects_invalid_aliases_duplicate_names_and_canonical_graph_cycles() {
    for aliases in [
        "alias_a,alias_b,active\nalias_b,alias_a,active\n",
        "outside,outside,active\n",
        "alias_a,fox,active\nalias_a,canine,active\n",
        "alias_a,missing,active\n",
        "outside,missing,active\n",
    ] {
        assert!(fixture(TAGS, aliases, "").is_err(), "{aliases}");
    }
    assert!(fixture(&format!("{TAGS}7,fox,1\n"), "", "").is_err());
    for implications in [
        "fox,canine,active\ncanine,alias_a,active\n",
        "fox,missing,active\n",
        "external,missing,active\n",
    ] {
        assert!(
            fixture(
                TAGS,
                "alias_a,fox,active\nexternal,fox,active\n",
                implications
            )
            .is_err(),
            "{implications}"
        );
    }
    // Unknown antecedents (even with missing targets) and inactive bad rows
    // are ignored; canonical self-edges do not become cycles or count edges.
    let taxonomy = fixture(
        TAGS,
        "alias_a,fox,active\noutside,missing,deleted\n",
        "unknown,missing,active\nalias_a,fox,active\nfox,missing,deleted\n",
    )
    .unwrap();
    assert_eq!(taxonomy.active_aliases, 1);
    assert_eq!(taxonomy.active_implications, 0);
    assert!(taxonomy.implications.iter().all(Vec::is_empty));
}

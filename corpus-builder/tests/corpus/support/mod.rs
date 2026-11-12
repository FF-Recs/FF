use ff_corpus_builder::BuildOptions;
use flate2::{Compression, write::GzEncoder};
use std::{fs, io::Write, path::Path};

pub(super) mod mutations;

pub(super) fn gzip(path: &Path, text: &str) {
    let mut gzip = GzEncoder::new(fs::File::create(path).unwrap(), Compression::fast());
    gzip.write_all(text.as_bytes()).unwrap();
    gzip.finish().unwrap();
}
pub(super) fn fixture_score(id: i64) -> i64 {
    match id {
        1 => i64::MIN,
        2 => i64::MAX,
        3 => -1,
        4 => 0,
        5 => 1,
        6 => 9_007_199_254_740_993,
        7 => -9_007_199_254_740_993,
        _ => id - 32,
    }
}
pub(super) fn fixture(root: &Path) -> BuildOptions {
    fs::create_dir_all(root).unwrap();
    let options = BuildOptions {
        posts: root.join("posts.csv.gz"),
        tags: root.join("tags.csv.gz"),
        aliases: root.join("tag_aliases.csv.gz"),
        implications: root.join("tag_implications.csv.gz"),
        work: root.join("work"),
        releases: root.join("releases"),
    };
    gzip(
        &options.tags,
        "id,name,category\n10,fox,0\n11,canine,0\n12,rare_content,0\n20,alias_old,0\n21,alias_chain,0\n22,flash_only_content,0\n100,artist_a,1\n101,artist_b,1\n102,unknown_artist,1\n103,rare_artist,1\n104,flash_only_artist,1\n200,supported_metadata,7\n201,implied_metadata,7\n202,boundary49_metadata,7\n203,metadata_alias,7\n204,metadata_alias_chain,7\n205,unused_metadata,7\n206,unsupported_category,9\n207,animated,7\n208,flash_only_metadata,7\n",
    );
    let mut tags = String::new();
    let mut decoder = flate2::read::MultiGzDecoder::new(fs::File::open(&options.tags).unwrap());
    std::io::Read::read_to_string(&mut decoder, &mut tags).unwrap();
    tags.push_str("105,conditional_dnp,1\n106,third-party_edit,1\n107,sound_warning,1\n108,ai_assisted,1\n109,ai_generated_background,1\n110,anonymous_artist,1\n111,epilepsy_warning,1\n112,avoid_posting,1\n113,jumpscare_warning,1\n114,ai_generated_reference,1\n115,eyestrain_warning,1\n116,motion_sickness_warning,1\n300,copyright50,3\n301,copyright49,3\n302,character50,4\n303,character49,4\n304,species50,5\n305,species49,5\n");
    gzip(&options.tags, &tags);
    gzip(
        &options.aliases,
        "id,antecedent_name,consequent_name,status\n1,alias_old,alias_chain,active\n2,alias_chain,fox,active\n3,metadata_alias,metadata_alias_chain,active\n4,metadata_alias_chain,supported_metadata,active\n",
    );
    gzip(
        &options.implications,
        "id,antecedent_name,consequent_name,status\n1,fox,canine,active\n2,fox,implied_metadata,active\n3,implied_metadata,supported_metadata,active\n",
    );
    let mut posts = String::from("id,md5,rating,file_ext,is_deleted,tag_string,score\n");
    for id in 1..=62 {
        let mut names = String::from("unsupported_category");
        if id == 1 {
            // A formerly exempt common name must satisfy the same support threshold.
            names.push_str(" animated");
        }
        if id <= 50 {
            names.push_str(" alias_old fox canine copyright50 character50 species50");
        }
        if id <= 25 {
            // Repeated aliases, explicit tags and implications count once per post.
            names.push_str(
                " metadata_alias metadata_alias_chain supported_metadata supported_metadata implied_metadata",
            );
        }
        if id <= 49 {
            names.push_str(" rare_content boundary49_metadata boundary49_metadata copyright49 character49 species49");
        }
        if id <= 10 {
            names.push_str(" artist_a unknown_artist conditional_dnp third-party_edit sound_warning ai_assisted ai_generated_background anonymous_artist epilepsy_warning avoid_posting jumpscare_warning ai_generated_reference eyestrain_warning motion_sickness_warning");
        }
        if (6..=17).contains(&id) {
            names.push_str(" artist_b");
        }
        if id <= 9 {
            names.push_str(" rare_artist");
        }
        if id == 61 {
            // This Flash row would cross every admission threshold and inflate
            // already-admitted frequencies if it reached normalization.
            names
                .push_str(" alias_old metadata_alias rare_content rare_artist boundary49_metadata");
        }
        let extension = match id {
            55 => "png",
            56 => "gif",
            57 => "webm",
            58 => "mp4",
            59 => "webp",
            61 => "swf",
            _ => "jpg",
        };
        posts.push_str(&format!(
            "{id},{id:032x},s,{extension},f,{names},{}\n",
            fixture_score(id)
        ));
    }
    // Unsupported-only tags would qualify in every category if Flash counted.
    for id in 63..=112 {
        posts.push_str(&format!(
            "{id},{id:032x},s,swf,f,flash_only_content flash_only_artist flash_only_metadata,invalid\n"
        ));
    }
    // No irrelevant payload field, including ID or an unknown tag, is parsed.
    posts.push_str("invalid,invalid,invalid,swf,f,unknown_flash_tag,invalid\n");
    // Deleted rows are ignored before media/tag parsing, just like the existing builder.
    posts.push_str("0,invalid,invalid,invalid,t,unknown_deleted_tag,invalid\n");
    posts.push_str(
        "0,invalid,invalid,invalid,t,supported_metadata boundary49_metadata unused_metadata,invalid\n",
    );
    gzip(&options.posts, &posts);
    options
}

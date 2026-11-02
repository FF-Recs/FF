use super::*;

pub(super) fn fixture(post_count: usize) -> Result<(tempfile::TempDir, AppConfig)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.workdir/ff-rs/app");
    std::fs::create_dir_all(&root)?;
    let temporary = tempfile::tempdir_in(root)?;
    let config = AppConfig {
        corpus_path: temporary.path().join("corpus"),
        interaction_path: temporary.path().join("interactions.sqlite3"),
        settings_path: temporary.path().join("import.toml"),
    };
    ff_test_support::corpus::write(&config.corpus_path, post_count)?;
    Ok((temporary, config))
}

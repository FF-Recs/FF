use crate::{
    app::{AppConfig, AppHandle},
    corpus,
};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

fn repository() -> Option<PathBuf> {
    let current = std::env::current_dir().ok()?;
    current
        .ancestors()
        .find(|p| p.join("corpus-release.json").is_file())
        .map(Path::to_path_buf)
        .or_else(|| {
            let source = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
            source
                .join("corpus-release.json")
                .is_file()
                .then(|| source.to_owned())
        })
}
fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}
fn user_roots() -> Result<(PathBuf, PathBuf)> {
    if cfg!(target_os = "windows") {
        let root = env_path("APPDATA")
            .context("APPDATA must identify the user configuration directory")?
            .join("FF");
        Ok((root.clone(), root))
    } else {
        let home = env_path("HOME").context("HOME must identify the user home directory")?;
        Ok((
            env_path("XDG_DATA_HOME")
                .unwrap_or_else(|| home.join(".local/share"))
                .join("ff"),
            env_path("XDG_CONFIG_HOME")
                .unwrap_or_else(|| home.join(".config"))
                .join("ff"),
        ))
    }
}
pub fn run(desktop: Option<fn(AppConfig, tokio::runtime::Handle) -> Result<()>>) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ff_core=info,ff_rs=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let mut corpus_path = env_path("FF_CORPUS_PATH");
    let mut interaction_path = env_path("FF_INTERACTION_PATH");
    let mut settings_path = env_path("FF_SETTINGS_PATH");
    let mut verify = false;
    let mut check = false;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        let (key, inline) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(a, b)| (a, Some(b)));
        match key {
            "--help" | "-h" => {
                println!(
                    "FF Rust{}\n\n  --corpus-path PATH       Immutable corpus release (or FF_CORPUS_PATH)\n  --interaction-path PATH SQLite interaction journal and projections (or FF_INTERACTION_PATH)\n  --settings-path PATH    Initial version-1 TOML preferences import (or FF_SETTINGS_PATH)\n  --verify-corpus         Verify all declared payload hashes; do not open user state\n  --check                 Open backend, fit current labels, print health, then exit\n\n{}",
                    if desktop.is_some() {
                        "/Slint desktop"
                    } else {
                        " core"
                    },
                    if desktop.is_some() {
                        "Without a command, opens the native desktop application."
                    } else {
                        "Choose --check or --verify-corpus; this binary has no desktop dependencies."
                    }
                );
                return Ok(());
            }
            "--verify-corpus" => verify = true,
            "--check" => check = true,
            "--corpus-path" | "--interaction-path" | "--settings-path" => {
                let value = inline
                    .map(str::to_owned)
                    .or_else(|| args.next())
                    .with_context(|| format!("{key} requires a path"))?;
                if value.is_empty() {
                    bail!("{key} requires a nonempty path");
                }
                let value = Some(PathBuf::from(value));
                match key {
                    "--corpus-path" => corpus_path = value,
                    "--interaction-path" => interaction_path = value,
                    _ => settings_path = value,
                }
            }
            _ => bail!("Unknown argument '{argument}'; use --help"),
        }
    }
    if verify && check {
        bail!("Choose either --verify-corpus or --check");
    }
    if !verify && !check && desktop.is_none() {
        bail!("The headless binary requires --check or --verify-corpus; use --help");
    }
    let repository = repository();
    let corpus_path = corpus_path.map(Ok).unwrap_or_else(|| {
        corpus::resolve_corpus(
            repository
                .as_deref()
                .context("No source corpus pin found; supply --corpus-path")?,
        )
    })?;
    if verify {
        corpus::verify_corpus(&corpus_path)?;
        println!("Verified corpus {}", corpus_path.display());
        return Ok(());
    }
    let (data_root, config_root) = user_roots()?;
    let config = AppConfig {
        corpus_path,
        interaction_path: interaction_path.unwrap_or_else(|| {
            repository
                .as_ref()
                .map(|p| p.join("data/private/interactions.sqlite3"))
                .unwrap_or_else(|| data_root.join("interactions.sqlite3"))
        }),
        settings_path: settings_path.unwrap_or_else(|| config_root.join("config.toml")),
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .thread_name("ff-io")
        .build()?;
    if check {
        runtime.block_on(async {
            let app = AppHandle::open(config).await?;
            let result = app.health().await;
            let shutdown = app.shutdown().await;
            println!("{}", serde_json::to_string_pretty(&result?)?);
            shutdown
        })
    } else {
        desktop.context("No desktop launcher configured")?(config, runtime.handle().clone())
    }
}

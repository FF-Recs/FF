use anyhow::{Context, Result, bail, ensure};
use ff_corpus_builder::{
    BuildOptions,
    acquisition::{DownloadOptions, download},
    build, equivalence,
};
use serde::Serialize;
use std::{collections::BTreeMap, fs::OpenOptions, io::Write, path::PathBuf};

const HELP: &str = "ff-corpus-builder download YYYY-MM-DD INPUTS_ROOT USER_AGENT\nff-corpus-builder build --posts PATH --tags PATH --aliases PATH --implications PATH\n  [--work .workdir/corpus-builder] --releases NEW_RELEASE_ROOT [--report NEW_JSON]\nff-corpus-builder compare REFERENCE_RELEASE CANDIDATE_RELEASE [--report NEW_JSON]\n\nDownload publishes four verified, immutable same-date input exports.\nBuild publishes immutable format-5 corpora from explicit retained gzip CSV inputs, including post scores.\nNo default-corpus selection or application state writes.\nCompare verifies every ordered logical output independently of serializer bytes.";

fn emit(value: &impl Serialize, report: Option<PathBuf>) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if let Some(path) = report {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("Creating new report {}", path.display()))?;
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&bytes)?;
    stdout.write_all(b"\n")?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        println!("{HELP}");
        return Ok(());
    };
    if command == "--help" || command == "-h" {
        println!("{HELP}");
        return Ok(());
    }
    if command == "download" {
        let date = args
            .next()
            .context("download requires YYYY-MM-DD INPUTS_ROOT USER_AGENT")?;
        if date == "--help" || date == "-h" {
            println!("{HELP}");
            return Ok(());
        }
        let destination = args
            .next()
            .context("download requires an input snapshot root")?
            .into();
        let user_agent = args
            .next()
            .context("download requires an identifiable User-Agent")?;
        ensure!(args.next().is_none(), "Unexpected extra download argument");
        return emit(
            &download(&DownloadOptions {
                date,
                destination,
                user_agent,
            })?,
            None,
        );
    }
    let mut positional = Vec::new();
    let mut options = BTreeMap::<String, PathBuf>::new();
    while let Some(argument) = args.next() {
        if argument.starts_with("--") {
            ensure!(
                [
                    "--posts",
                    "--tags",
                    "--aliases",
                    "--implications",
                    "--work",
                    "--releases",
                    "--report"
                ]
                .contains(&argument.as_str()),
                "Unknown option {argument}"
            );
            let value = args
                .next()
                .with_context(|| format!("{argument} requires a path"))?;
            ensure!(
                !value.is_empty() && !value.starts_with("--"),
                "{argument} requires a path"
            );
            ensure!(
                options
                    .insert(argument.clone(), PathBuf::from(value))
                    .is_none(),
                "Duplicate option {argument}"
            );
        } else {
            positional.push(PathBuf::from(argument));
        }
    }
    let report = options.remove("--report");
    if let Some(path) = &report {
        ensure!(
            !path.try_exists()?,
            "Report already exists: {}",
            path.display()
        );
    }
    match command.as_str() {
        "build" => {
            ensure!(
                positional.is_empty(),
                "build does not accept positional paths"
            );
            let mut required = |key: &str| {
                options
                    .remove(key)
                    .with_context(|| format!("Missing {key}"))
            };
            let posts = required("--posts")?;
            let tags = required("--tags")?;
            let aliases = required("--aliases")?;
            let implications = required("--implications")?;
            let releases = required("--releases")?;
            let input = BuildOptions {
                posts,
                tags,
                aliases,
                implications,
                releases,
                work: options
                    .remove("--work")
                    .unwrap_or_else(|| ".workdir/corpus-builder".into()),
            };
            ensure!(options.is_empty(), "Unsupported build options");
            emit(&build(&input)?, report)
        }
        "compare" => {
            ensure!(
                options.is_empty() && positional.len() == 2,
                "compare requires REFERENCE CANDIDATE and optional --report only"
            );
            emit(
                &equivalence::compare(&positional[0], &positional[1])?,
                report,
            )
        }
        _ => bail!("Unknown command {command}; use --help"),
    }
}

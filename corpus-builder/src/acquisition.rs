// Bounded acquisition of a pinned, immutable set of upstream CSV exports.
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use chrono::{DateTime, NaiveDate};
use reqwest::{Url, blocking::Client, header::HeaderValue};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{build, input};

const CATALOG_LIMIT: usize = 1 << 20;
const MAX_EXPORT_BYTES: u64 = 20 * 1024 * 1024 * 1024;

pub struct DownloadOptions {
    pub date: String,
    pub destination: PathBuf,
    pub user_agent: String,
}

#[derive(Debug, Serialize)]
pub struct DownloadReport {
    pub snapshot_path: PathBuf,
    pub file_count: usize,
}

struct Endpoints {
    catalog: Url,
    exports: Url,
}

struct Settings {
    timeout: Duration,
    retry_delays: [Duration; 2],
    reserve_bytes: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(120),
            retry_delays: [Duration::from_secs(1), Duration::from_secs(4)],
            reserve_bytes: 1 << 30,
        }
    }
}

fn production_endpoints() -> Result<Endpoints> {
    Ok(Endpoints {
        catalog: Url::parse("https://e621.net/db_exports.json")?,
        exports: Url::parse("https://static1.e621.net/data/db_export/")?,
    })
}

/// Download only the requested date from the fixed, approved production endpoints.
pub fn download(options: &DownloadOptions) -> Result<DownloadReport> {
    acquire(options, &production_endpoints()?, &Settings::default())
}

struct Dataset {
    name: &'static str,
    columns: &'static [&'static str],
    nonempty: bool,
}

const DATASETS: [Dataset; 4] = [
    Dataset {
        name: "posts",
        columns: &[
            "id",
            "md5",
            "rating",
            "file_ext",
            "is_deleted",
            "tag_string",
            "score",
        ],
        nonempty: true,
    },
    Dataset {
        name: "tags",
        columns: &["id", "name", "category"],
        nonempty: true,
    },
    Dataset {
        name: "tag_aliases",
        columns: &["antecedent_name", "consequent_name", "status"],
        nonempty: false,
    },
    Dataset {
        name: "tag_implications",
        columns: &["antecedent_name", "consequent_name", "status"],
        nonempty: false,
    },
];

struct Member<'a> {
    dataset: &'a Dataset,
    url: Url,
    bytes: u64,
    sha256: String,
    updated_at: String,
}

#[derive(Serialize)]
struct Manifest<'a> {
    version: u32,
    export_date: &'a str,
    source_catalog_url: &'a str,
    source_catalog_sha256: String,
    files: Vec<ManifestFile>,
}

#[derive(Serialize)]
struct ManifestFile {
    file: String,
    url: String,
    bytes: u64,
    sha256: String,
    rows: u64,
    updated_at: String,
}

fn parse_date(value: &str) -> Result<NaiveDate> {
    let bytes = value.as_bytes();
    ensure!(
        bytes.len() == 10
            && bytes[4] == b'-'
            && bytes[7] == b'-'
            && bytes
                .iter()
                .enumerate()
                .all(|(i, b)| i == 4 || i == 7 || b.is_ascii_digit())
            && &bytes[..4] != b"0000",
        "An explicit YYYY-MM-DD export date is required"
    );
    let date =
        NaiveDate::parse_from_str(value, "%Y-%m-%d").context("Invalid export calendar date")?;
    ensure!(
        date.format("%Y-%m-%d").to_string() == value,
        "Invalid export date"
    );
    Ok(date)
}

fn token(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
}

fn user_agent(value: &str) -> Result<HeaderValue> {
    let header = HeaderValue::from_str(value).context("Invalid User-Agent header")?;
    // RFC 9110: User-Agent starts with a product, followed by products/comments
    // separated by whitespace. A merely nonblank arbitrary header is not enough.
    let bytes = value.as_bytes();
    let mut position = 0;
    let mut first = true;
    loop {
        let start = position;
        while position < bytes.len() && token(bytes[position]) {
            position += 1;
        }
        if position > start {
            if bytes.get(position) == Some(&b'/') {
                position += 1;
                let version = position;
                while position < bytes.len() && token(bytes[position]) {
                    position += 1;
                }
                ensure!(position > version, "Invalid User-Agent product version");
            }
        } else if !first && bytes.get(position) == Some(&b'(') {
            let mut depth = 1_u32;
            position += 1;
            while depth > 0 && position < bytes.len() {
                let byte = bytes[position];
                position += 1;
                match byte {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    b'\\' => {
                        ensure!(position < bytes.len(), "Invalid User-Agent comment escape");
                        let escaped = bytes[position];
                        ensure!(
                            escaped == b'\t' || (b' '..=b'~').contains(&escaped),
                            "Invalid User-Agent comment"
                        );
                        position += 1;
                    }
                    b'\t' | b' '..=b'~' => {}
                    _ => bail!("Invalid User-Agent comment"),
                }
            }
            ensure!(depth == 0, "Unterminated User-Agent comment");
        } else {
            bail!("An identifiable, valid User-Agent product is required");
        }
        first = false;
        if position == bytes.len() {
            return Ok(header);
        }
        let separator = position;
        while matches!(bytes.get(position), Some(b' ' | b'\t')) {
            position += 1;
        }
        ensure!(
            position > separator && position < bytes.len(),
            "Invalid User-Agent separator"
        );
    }
}

fn loopback(url: &Url) -> bool {
    url.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("localhost")
            || host
                .trim_start_matches('[')
                .trim_end_matches(']')
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    })
}

fn client(options: &DownloadOptions, endpoints: &Endpoints, settings: &Settings) -> Result<Client> {
    let mut builder = Client::builder()
        .user_agent(user_agent(&options.user_agent)?)
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .no_zstd()
        .connect_timeout(settings.timeout)
        // reqwest 0.12 blocking applies this separately to send() and each Read.
        // It does NOT set the async client's whole-response deadline. This keeps
        // multi-hour transfers alive as long as each read makes progress.
        .timeout(settings.timeout);
    if loopback(&endpoints.catalog) {
        // Only the private fixture seam can select a loopback catalog. Do not
        // mutate process-wide proxy variables or disable production env proxies.
        builder = builder.no_proxy();
    }
    builder.build().context("Creating export HTTP client")
}

fn catalog(client: &Client, url: &Url, buffer: &mut [u8]) -> Result<Vec<u8>> {
    let mut response = client
        .get(url.clone())
        .send()
        .context("Fetching export catalog")?;
    ensure!(
        response.status().is_success(),
        "Catalog HTTP {}",
        response.status()
    );
    ensure!(
        response
            .content_length()
            .is_none_or(|n| n <= CATALOG_LIMIT as u64),
        "Export catalog exceeds 1 MiB"
    );
    let mut bytes = Vec::with_capacity(8192);
    loop {
        let count = response.read(buffer).context("Reading export catalog")?;
        if count == 0 {
            break;
        }
        ensure!(
            count <= CATALOG_LIMIT - bytes.len(),
            "Export catalog exceeds 1 MiB"
        );
        bytes.extend_from_slice(&buffer[..count]);
    }
    Ok(bytes)
}

fn member_url(raw: &str, dataset: &str, endpoints: &Endpoints) -> Result<Url> {
    let url = Url::parse(raw).context("Invalid export URL")?;
    let expected = endpoints.exports.join(&format!("{dataset}.csv.gz"))?;
    // Check the raw path as well: URL parsing must not normalize traversal into
    // an allowed path, or erase an empty credential component before validation.
    let (authority, raw_path) = raw
        .split_once("://")
        .and_then(|(_, rest)| rest.split_once('/'))
        .context("Export URL must have an absolute path")?;
    ensure!(
        url.scheme() == expected.scheme()
            && url.host_str() == expected.host_str()
            && url.port_or_known_default() == expected.port_or_known_default()
            && url.username().is_empty()
            && url.password().is_none()
            && !authority.contains('@')
            && url.query().is_none()
            && url.fragment().is_none()
            && url.path() == expected.path()
            && raw_path == expected.path().trim_start_matches('/'),
        "Unexpected export URL for '{dataset}'"
    );
    Ok(url)
}

fn resolve<'a>(
    bytes: &[u8],
    date: NaiveDate,
    datasets: &'a [Dataset],
    endpoints: &Endpoints,
) -> Result<Vec<Member<'a>>> {
    let catalog: serde_json::Value =
        serde_json::from_slice(bytes).context("Parsing export catalog")?;
    let entries = catalog
        .as_array()
        .context("Export catalog must be an array")?;
    datasets
        .iter()
        .map(|dataset| {
            // Unrelated exports have no authority over this acquisition, even if
            // their catalog entries are malformed or absent.
            let mut matches = entries
                .iter()
                .filter(|entry| entry.get("name").and_then(|v| v.as_str()) == Some(dataset.name));
            let entry = matches
                .next()
                .with_context(|| format!("Catalog has no '{}' export", dataset.name))?;
            ensure!(
                matches.next().is_none(),
                "Catalog must contain exactly one '{}' export",
                dataset.name
            );
            let updated_at = entry
                .get("updated_at")
                .and_then(|v| v.as_str())
                .context("Missing export updated_at")?;
            let updated = DateTime::parse_from_rfc3339(updated_at)
                .context("Export updated_at must be an offset-aware timestamp")?;
            ensure!(
                updated.date_naive() == date,
                "Export '{}' is not from requested date {date}",
                dataset.name
            );
            let url = member_url(
                entry
                    .get("url")
                    .and_then(|v| v.as_str())
                    .context("Missing export URL")?,
                dataset.name,
                endpoints,
            )?;
            let bytes = entry
                .get("file_size")
                .and_then(|v| v.as_u64())
                .context("Invalid export file_size")?;
            let hash = entry
                .get("checksum")
                .and_then(|v| v.as_str())
                .context("Missing export checksum")?;
            ensure!(
                (18..=MAX_EXPORT_BYTES).contains(&bytes),
                "Invalid size for '{}'",
                dataset.name
            );
            ensure!(
                hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
                "Invalid checksum for '{}'",
                dataset.name
            );
            Ok(Member {
                dataset,
                url,
                bytes,
                sha256: hash.to_ascii_lowercase(),
                updated_at: updated_at.to_owned(),
            })
        })
        .collect()
}

enum AttemptError {
    Retry(anyhow::Error),
    Fatal(anyhow::Error),
}

fn transient_read(error: &io::Error) -> bool {
    if matches!(
        error.kind(),
        io::ErrorKind::Interrupted
            | io::ErrorKind::UnexpectedEof
            | io::ErrorKind::TimedOut
            | io::ErrorKind::WouldBlock
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::NotConnected
            | io::ErrorKind::BrokenPipe
    ) {
        return true;
    }
    let mut source = error
        .get_ref()
        .map(|inner| inner as &(dyn std::error::Error + 'static));
    while let Some(inner) = source {
        if let Some(http) = inner.downcast_ref::<reqwest::Error>() {
            if http.is_timeout() || http.is_connect() || http.is_body() {
                return true;
            }
        }
        source = inner.source();
    }
    false
}

fn attempt(
    client: &Client,
    member: &Member<'_>,
    path: &Path,
    buffer: &mut [u8],
) -> std::result::Result<[u8; 32], AttemptError> {
    let mut response = client.get(member.url.clone()).send().map_err(|error| {
        if error.is_timeout() || error.is_connect() || error.is_request() {
            AttemptError::Retry(error.into())
        } else {
            AttemptError::Fatal(error.into())
        }
    })?;
    let status = response.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
        return Err(AttemptError::Retry(anyhow!("Export HTTP {status}")));
    }
    if !status.is_success() {
        return Err(AttemptError::Fatal(anyhow!("Export HTTP {status}")));
    }
    if response.content_length().is_some_and(|n| n > member.bytes) {
        return Err(AttemptError::Fatal(anyhow!(
            "Export exceeds its pinned catalog size"
        )));
    }
    // Disk errors are always fatal, never confused with transport IO errors.
    // Each retry owns this same staging path and restarts from byte zero.
    let mut target = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("Creating staged export {}", path.display()))
        .map_err(AttemptError::Fatal)?;
    let mut hash = Sha256::new();
    let mut total = 0_u64;
    loop {
        let count = response.read(buffer).map_err(|error| {
            let retry = transient_read(&error);
            let error = anyhow::Error::new(error).context("Reading export response");
            if retry {
                AttemptError::Retry(error)
            } else {
                AttemptError::Fatal(error)
            }
        })?;
        if count == 0 {
            break;
        }
        if count as u64 > member.bytes - total {
            return Err(AttemptError::Fatal(anyhow!(
                "Export exceeds its pinned catalog size"
            )));
        }
        target
            .write_all(&buffer[..count])
            .context("Writing staged export")
            .map_err(AttemptError::Fatal)?;
        hash.update(&buffer[..count]);
        total += count as u64;
    }
    if total != member.bytes {
        return Err(AttemptError::Retry(anyhow!(
            "Export size {total} differs from pinned catalog size {}",
            member.bytes
        )));
    }
    target
        .sync_all()
        .context("Syncing staged export")
        .map_err(AttemptError::Fatal)?;
    Ok(hash.finalize().into())
}

fn payload(
    client: &Client,
    member: &Member<'_>,
    path: &Path,
    settings: &Settings,
    buffer: &mut [u8],
) -> Result<()> {
    let mut expected = [0_u8; 32];
    hex::decode_to_slice(&member.sha256, &mut expected).context("Decoding pinned checksum")?;
    for index in 0..3 {
        match attempt(client, member, path, buffer) {
            Ok(hash) => {
                ensure!(
                    hash == expected,
                    "Checksum mismatch for '{}'; the pinned export may have rotated",
                    member.dataset.name
                );
                return Ok(());
            }
            Err(AttemptError::Fatal(error)) => return Err(error),
            Err(AttemptError::Retry(error)) => {
                if index == 2 {
                    return Err(error.context("Export failed after three attempts"));
                }
                thread::sleep(settings.retry_delays[index]);
            }
        }
    }
    unreachable!("all three payload attempts return or retry")
}

#[cfg(unix)]
fn available_space(path: &Path) -> Result<u64> {
    let stats = rustix::fs::statvfs(path).context("Reading destination filesystem capacity")?;
    stats
        .f_bavail
        .checked_mul(stats.f_frsize)
        .context("Destination capacity overflow")
}

#[cfg(windows)]
fn available_space(path: &Path) -> Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut available = 0_u64;
    // SAFETY: wide is NUL-terminated and all output pointers are valid or null.
    let success = unsafe {
        windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if success == 0 {
        return Err(io::Error::last_os_error()).context("Reading destination filesystem capacity");
    }
    Ok(available)
}

#[cfg(not(any(unix, windows)))]
fn available_space(_path: &Path) -> Result<u64> {
    bail!("Input acquisition requires a supported Unix or Windows filesystem")
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("Creating {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    // Match the corpus publisher's supported durability convention: file syncs
    // on all platforms and directory syncs on Unix (not ordinary Windows handles).
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn acquire(
    options: &DownloadOptions,
    endpoints: &Endpoints,
    settings: &Settings,
) -> Result<DownloadReport> {
    let date = parse_date(&options.date)?;
    let client = client(options, endpoints, settings)?;
    let mut buffer = vec![0_u8; 1 << 20];
    let catalog = catalog(&client, &endpoints.catalog, &mut buffer)?;
    let members = resolve(&catalog, date, &DATASETS, endpoints)?;
    // Resolve the entire selected catalog before creating staging or fetching a
    // payload. Capacity is measured at the actual destination, including mounts.
    fs::create_dir_all(&options.destination).context("Creating inputs root")?;
    let root = fs::canonicalize(&options.destination).context("Resolving inputs root")?;
    let required = members
        .iter()
        .try_fold(settings.reserve_bytes, |sum, member| {
            sum.checked_add(member.bytes)
                .context("Required disk space overflow")
        })?;
    ensure!(
        available_space(&root)? >= required,
        "Insufficient free space for pinned inputs plus {} byte reserve",
        settings.reserve_bytes
    );
    let stage = tempfile::Builder::new()
        .prefix(".download-")
        .tempdir_in(&root)?;
    write_new(&stage.path().join("source-catalog.json"), &catalog)?;
    let mut files = Vec::with_capacity(members.len());
    let mut record = csv::ByteRecord::new();
    for member in members {
        let file = format!("{}.csv.gz", member.dataset.name);
        let path = stage.path().join(&file);
        payload(&client, &member, &path, settings, &mut buffer)
            .with_context(|| format!("Downloading '{}'", member.dataset.name))?;
        let mut csv = input::open(&path, member.dataset.columns)?;
        let mut rows = 0_u64;
        while csv.read(&mut record)? {
            rows = rows.checked_add(1).context("CSV row count overflow")?;
        }
        ensure!(
            !member.dataset.nonempty || rows > 0,
            "Required export '{}' is empty",
            member.dataset.name
        );
        files.push(ManifestFile {
            file,
            url: member.url.to_string(),
            bytes: member.bytes,
            sha256: member.sha256,
            rows,
            updated_at: member.updated_at,
        });
    }
    let file_count = files.len();
    let manifest = Manifest {
        version: 1,
        export_date: &options.date,
        source_catalog_url: endpoints.catalog.as_str(),
        source_catalog_sha256: hex::encode(Sha256::digest(&catalog)),
        files,
    };
    // This native serialization defines its own snapshot ID. Never pretend its
    // bytes or directory IDs are those of the retired managed serializer.
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    manifest_bytes.push(b'\n');
    write_new(&stage.path().join("manifest.json"), &manifest_bytes)?;
    let snapshot_path = root.join(hex::encode(Sha256::digest(&manifest_bytes)));
    sync_directory(stage.path())?;
    sync_directory(&root)?;
    build::publish(stage.path(), &snapshot_path)?;
    sync_directory(&root)?;
    // TempDir only ever removes its original, uniquely-owned staging path.
    Ok(DownloadReport {
        snapshot_path,
        file_count,
    })
}

#[cfg(test)]
#[path = "../tests/unit/acquisition.rs"]
mod tests;

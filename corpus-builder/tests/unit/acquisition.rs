use super::{DownloadOptions, Endpoints, Settings, acquire, production_endpoints};
use flate2::{Compression, write::GzEncoder};
use parking_lot::{Condvar, Mutex};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::Arc,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const DATE: &str = "2026-01-02";
const UPDATED: &str = "2026-01-02T23:30:00-04:00";
const AGENT: &str = "FF-CorpusBuilder-tests/1.0 (local synthetic fixture)";
const CORE: [&str; 4] = ["posts", "tags", "tag_aliases", "tag_implications"];

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn gzip(text: &str) -> Vec<u8> {
    let mut output = GzEncoder::new(Vec::new(), Compression::fast());
    output.write_all(text.as_bytes()).unwrap();
    output.finish().unwrap()
}

fn path(name: &str) -> String {
    format!("/data/db_export/{name}.csv.gz")
}

#[derive(Clone, Debug)]
struct Request {
    path: String,
    headers: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TransferPhase {
    Progress,
    Stalled,
    Complete,
}

#[derive(Default)]
struct Transfer {
    events: Mutex<Vec<(TransferPhase, Instant)>>,
    changed: Condvar,
}

impl Transfer {
    fn record(&self, phase: TransferPhase) {
        self.events.lock().push((phase, Instant::now()));
        self.changed.notify_all();
    }

    fn wait_for_phase(&self, phase: TransferPhase) {
        let deadline = Instant::now() + Duration::from_secs(15);
        let mut events = self.events.lock();
        while !events.iter().any(|(observed, _)| *observed == phase) {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .unwrap_or_else(|| panic!("server did not reach {phase:?}: {events:?}"));
            self.changed.wait_for(&mut events, remaining);
        }
    }
}

#[derive(Clone)]
struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    declared: Option<usize>,
    omit_length: bool,
    disconnect: bool,
    stall: bool,
    chunk_interval: Option<Duration>,
    transfer: Option<Arc<Transfer>>,
}

impl Reply {
    fn ok(body: Vec<u8>) -> Self {
        Self {
            status: 200,
            headers: Vec::new(),
            body,
            declared: None,
            omit_length: false,
            disconnect: false,
            stall: false,
            chunk_interval: None,
            transfer: None,
        }
    }

    fn status(status: u16) -> Self {
        Self {
            status,
            ..Self::ok(Vec::new())
        }
    }

    fn redirect(location: String) -> Self {
        let mut response = Self::status(302);
        response.headers.push(("Location".into(), location));
        response
    }

    fn short(body: &[u8], declared: usize) -> Self {
        Self {
            declared: Some(declared),
            ..Self::ok(body.to_vec())
        }
    }
}

struct Scenario {
    catalog: Vec<u8>,
    payloads: BTreeMap<String, Vec<u8>>,
    scripts: HashMap<String, Vec<Reply>>,
    requests: Vec<Request>,
}

impl Scenario {
    fn respond(&mut self, request: Request) -> Reply {
        let target = request.path.clone();
        self.requests.push(request);
        let attempt = self.requests.iter().filter(|r| r.path == target).count();
        if let Some(reply) = self.scripts.get(&target).and_then(|s| s.get(attempt - 1)) {
            return reply.clone();
        }
        if target == "/db_exports.json" {
            return Reply::ok(self.catalog.clone());
        }
        self.payloads
            .get(&target)
            .cloned()
            .map(Reply::ok)
            .unwrap_or_else(|| Reply::status(404))
    }
}

// The accept thread never waits on a response. Stalled workers wait on an explicit
// shutdown event, so a timeout test does not impose a sleep on fixture teardown.
struct Server {
    address: SocketAddr,
    state: Arc<Mutex<Scenario>>,
    shutdown: Arc<(Mutex<bool>, Condvar)>,
    worker: Option<JoinHandle<()>>,
}

impl Server {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let base = format!("http://{address}");
        let payloads = BTreeMap::from([
            (
                path("posts"),
                gzip(
                    "id,md5,rating,file_ext,is_deleted,tag_string,score\n1,00000000000000000000000000000001,s,jpg,f,fox,0\n",
                ),
            ),
            (path("tags"), gzip("id,name,category\n10,fox,0\n")),
            (
                path("tag_aliases"),
                gzip("antecedent_name,consequent_name,status\n"),
            ),
            (
                path("tag_implications"),
                gzip("antecedent_name,consequent_name,status\n"),
            ),
        ]);
        let entries: Vec<_> = CORE
            .iter()
            .copied()
            .map(|name| {
                let bytes = &payloads[&path(name)];
                json!({
                    "name": name,
                    "file_name": format!("{name}.csv.gz"),
                    "file_size": bytes.len(),
                    "checksum": digest(bytes),
                    "updated_at": UPDATED,
                    "url": format!("{base}{}", path(name)),
                })
            })
            .collect();
        let state = Arc::new(Mutex::new(Scenario {
            catalog: serde_json::to_vec_pretty(&entries).unwrap(),
            payloads,
            scripts: HashMap::new(),
            requests: Vec::new(),
        }));
        let shutdown = Arc::new((Mutex::new(false), Condvar::new()));
        let shared = Arc::clone(&state);
        let stop = Arc::clone(&shutdown);
        let worker = thread::spawn(move || {
            let mut workers = Vec::new();
            for connection in listener.incoming() {
                let (lock, _) = &*stop;
                if *lock.lock() {
                    break;
                }
                let stream = connection.unwrap();
                let shared = Arc::clone(&shared);
                let stop = Arc::clone(&stop);
                workers.push(thread::spawn(move || serve(stream, &shared, &stop)));
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            address,
            state,
            shutdown,
            worker: Some(worker),
        }
    }

    fn url(&self, suffix: &str) -> reqwest::Url {
        format!("http://{}{suffix}", self.address).parse().unwrap()
    }

    fn requests(&self) -> Vec<Request> {
        self.state.lock().requests.clone()
    }

    fn calls(&self, target: &str) -> usize {
        self.requests().iter().filter(|r| r.path == target).count()
    }

    fn script(&self, target: &str, responses: Vec<Reply>) {
        self.state.lock().scripts.insert(target.into(), responses);
    }

    fn catalog(&self) -> Value {
        serde_json::from_slice(&self.state.lock().catalog).unwrap()
    }

    fn set_catalog(&self, catalog: Value) {
        self.state.lock().catalog = serde_json::to_vec(&catalog).unwrap();
    }

    // Re-pin intentional invalid gzip/CSV bytes, distinguishing parser failures
    // from the earlier transport checksum guard.
    fn repin(&self, name: &str, bytes: Vec<u8>) {
        let mut catalog = self.catalog();
        let entry = catalog
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|v| v["name"] == name)
            .unwrap();
        entry["checksum"] = json!(digest(&bytes));
        entry["file_size"] = json!(bytes.len());
        self.set_catalog(catalog);
        self.state.lock().payloads.insert(path(name), bytes);
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let (lock, event) = &*self.shutdown;
        *lock.lock() = true;
        event.notify_all();
        let _ = TcpStream::connect_timeout(&self.address, Duration::from_secs(2));
        self.worker.take().unwrap().join().unwrap();
    }
}

fn serve(mut stream: TcpStream, state: &Mutex<Scenario>, shutdown: &(Mutex<bool>, Condvar)) {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut chunk = [0; 1024];
    while !bytes.windows(4).any(|w| w == b"\r\n\r\n") {
        let count = match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(count) => count,
        };
        bytes.extend_from_slice(&chunk[..count]);
        assert!(
            bytes.len() <= 32 * 1024,
            "unexpectedly large fixture request"
        );
    }
    let headers = String::from_utf8(bytes).unwrap();
    let target = headers
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .to_owned();
    let reply = state.lock().respond(Request {
        path: target,
        headers,
    });
    if reply.disconnect {
        return;
    }
    let mut header = format!("HTTP/1.1 {} Fixture\r\nConnection: close\r\n", reply.status);
    if !reply.omit_length {
        header.push_str(&format!(
            "Content-Length: {}\r\n",
            reply.declared.unwrap_or(reply.body.len())
        ));
    }
    for (name, value) in reply.headers {
        header.push_str(&format!("{name}: {value}\r\n"));
    }
    header.push_str("\r\n");
    if stream.write_all(header.as_bytes()).is_err() {
        return;
    }
    if let Some(interval) = reply.chunk_interval {
        let chunk_size = reply.body.len().div_ceil(16).max(1);
        let mut deadline = Instant::now();
        for (index, chunk) in reply.body.chunks(chunk_size).enumerate() {
            if index != 0 {
                deadline += interval;
                let (lock, event) = shutdown;
                let mut stopped = lock.lock();
                while !*stopped {
                    let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                        break;
                    };
                    event.wait_for(&mut stopped, remaining);
                }
                if *stopped {
                    return;
                }
            }
            if stream.write_all(chunk).is_err() {
                return;
            }
            if let Some(transfer) = &reply.transfer {
                transfer.record(TransferPhase::Progress);
            }
        }
    } else {
        if stream.write_all(&reply.body).is_err() {
            return;
        }
        if let Some(transfer) = &reply.transfer {
            transfer.record(TransferPhase::Progress);
        }
    }
    if reply.stall {
        if let Some(transfer) = &reply.transfer {
            transfer.record(TransferPhase::Stalled);
        }
        let (lock, event) = shutdown;
        let mut stopped = lock.lock();
        while !*stopped {
            event.wait(&mut stopped);
        }
    } else if let Some(transfer) = &reply.transfer {
        transfer.record(TransferPhase::Complete);
    }
}

struct Fixture {
    server: Server,
    root: tempfile::TempDir,
    options: DownloadOptions,
    endpoints: Endpoints,
    settings: Settings,
}

impl Fixture {
    fn new() -> Self {
        let scratch = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join(".workdir/rust-downloader/DownloadFixtures");
        fs::create_dir_all(&scratch).unwrap();
        let root = tempfile::Builder::new()
            .prefix("acquire-")
            .tempdir_in(scratch)
            .unwrap();
        let server = Server::new();
        let endpoints = Endpoints {
            catalog: server.url("/db_exports.json"),
            exports: server.url("/data/db_export/"),
        };
        let options = DownloadOptions {
            date: DATE.into(),
            destination: root.path().join("inputs"),
            user_agent: AGENT.into(),
        };
        let settings = Settings {
            timeout: Duration::from_secs(2),
            retry_delays: [Duration::ZERO; 2],
            reserve_bytes: 0,
        };
        Self {
            server,
            root,
            options,
            endpoints,
            settings,
        }
    }

    fn run(&self) -> anyhow::Result<super::DownloadReport> {
        acquire(&self.options, &self.endpoints, &self.settings)
    }

    fn assert_clean(&self) {
        if self.options.destination.exists() {
            let entries: Vec<_> = fs::read_dir(&self.options.destination)
                .unwrap()
                .map(|e| e.unwrap().file_name())
                .collect();
            assert!(
                entries.is_empty(),
                "unpublished files or stages remain: {entries:?}"
            );
        }
    }

    fn fail(&self) {
        assert!(
            self.run().is_err(),
            "invalid acquisition unexpectedly published"
        );
        self.assert_clean();
    }

    fn assert_manifest(&self, snapshot: &Path) {
        let manifest_bytes = fs::read(snapshot.join("manifest.json")).unwrap();
        assert_eq!(
            snapshot.file_name().unwrap().to_str().unwrap(),
            digest(&manifest_bytes)
        );
        let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
        assert_eq!(manifest["version"], 1);
        assert_eq!(manifest["export_date"], self.options.date);
        assert_eq!(
            manifest["source_catalog_url"],
            self.endpoints.catalog.as_str()
        );
        let state = self.server.state.lock();
        assert_eq!(
            fs::read(snapshot.join("source-catalog.json")).unwrap(),
            state.catalog
        );
        assert_eq!(manifest["source_catalog_sha256"], digest(&state.catalog));
        let files = manifest["files"].as_array().unwrap();
        assert_eq!(files.len(), CORE.len());
        for name in CORE {
            let file_name = format!("{name}.csv.gz");
            let entries: Vec<_> = files.iter().filter(|f| f["file"] == file_name).collect();
            assert_eq!(entries.len(), 1);
            let entry = entries[0];
            let expected = &state.payloads[&path(name)];
            assert_eq!(fs::read(snapshot.join(&file_name)).unwrap(), *expected);
            assert_eq!(entry["bytes"], expected.len());
            assert_eq!(entry["sha256"], digest(expected));
            assert_eq!(entry["url"], self.server.url(&path(name)).as_str());
            assert_eq!(entry["updated_at"], UPDATED);
            assert_eq!(entry["rows"], usize::from(matches!(name, "posts" | "tags")));
        }
        assert_eq!(fs::read_dir(snapshot).unwrap().count(), CORE.len() + 2);
        assert_eq!(fs::read_dir(&self.options.destination).unwrap().count(), 1);
    }
}

#[test]
fn four_core_inputs_publish_exact_bytes_and_manifest_identity() {
    for agent in [
        AGENT,
        r"FF-CorpusBuilder-tests/1.0 (outer (nested) escaped \) and \\) Helper/2.3 Extra",
    ] {
        let mut fixture = Fixture::new();
        fixture.options.user_agent = agent.into();
        let report = fixture.run().unwrap();
        assert_eq!(report.file_count, CORE.len());
        fixture.assert_manifest(&report.snapshot_path);
        let expected_requests: Vec<_> = std::iter::once("/db_exports.json".to_owned())
            .chain(CORE.into_iter().map(path))
            .collect();
        assert_eq!(
            fixture
                .server
                .requests()
                .into_iter()
                .map(|request| request.path)
                .collect::<Vec<_>>(),
            expected_requests
        );
        assert_eq!(fixture.server.calls(&path("pools")), 0);
        assert_eq!(fixture.server.calls(&path("post_replacements")), 0);
        for request in fixture.server.requests() {
            assert!(request.headers.lines().any(|line| {
                line.split_once(':').is_some_and(|(key, value)| {
                    key.eq_ignore_ascii_case("user-agent") && value.trim() == agent
                })
            }));
        }
    }
}

#[test]
fn irrelevant_absent_or_malformed_exports_do_not_block_acquisition() {
    for malformed in [false, true] {
        let fixture = Fixture::new();
        let mut catalog = fixture.server.catalog();
        if malformed {
            catalog.as_array_mut().unwrap().extend([
                json!({"name":"pools", "url":false, "file_size":"broken"}),
                json!({"name":"pools"}),
                json!({"name":"post_replacements", "url":"http://evil.invalid/"}),
            ]);
        }
        fixture.server.set_catalog(catalog);
        let report = fixture.run().unwrap();
        fixture.assert_manifest(&report.snapshot_path);
        assert_eq!(fixture.server.requests().len(), CORE.len() + 1);
        assert_eq!(fixture.server.calls(&path("pools")), 0);
        assert_eq!(fixture.server.calls(&path("post_replacements")), 0);
    }
}

#[test]
fn every_selected_catalog_entry_is_validated_before_any_payload_request() {
    let cases = [
        "missing",
        "duplicate",
        "missing_name",
        "wrong_date",
        "invalid_timestamp",
        "no_offset",
        "missing_url",
        "hash_short",
        "hash_nonhex",
        "missing_hash",
        "size_too_small",
        "size_too_large",
        "negative_size",
        "fractional_size",
        "string_size",
        "missing_size",
    ];
    for case in cases {
        let fixture = Fixture::new();
        let mut catalog = fixture.server.catalog();
        // Last required dataset catches implementations that validate lazily.
        match case {
            "missing" => {
                catalog.as_array_mut().unwrap().remove(3);
            }
            "duplicate" => {
                let entry = catalog[3].clone();
                catalog.as_array_mut().unwrap().push(entry);
            }
            "missing_name" => {
                catalog[3].as_object_mut().unwrap().remove("name");
            }
            "wrong_date" => catalog[3]["updated_at"] = json!("2026-01-03T00:00:00+14:00"),
            "invalid_timestamp" => catalog[3]["updated_at"] = json!("not a timestamp"),
            "no_offset" => catalog[3]["updated_at"] = json!("2026-01-02T12:00:00"),
            "missing_url" => {
                catalog[3].as_object_mut().unwrap().remove("url");
            }
            "hash_short" => catalog[3]["checksum"] = json!("abcd"),
            "hash_nonhex" => catalog[3]["checksum"] = json!("g".repeat(64)),
            "missing_hash" => {
                catalog[3].as_object_mut().unwrap().remove("checksum");
            }
            "size_too_small" => catalog[3]["file_size"] = json!(17),
            "size_too_large" => catalog[3]["file_size"] = json!(20_u64 * 1024 * 1024 * 1024 + 1),
            "negative_size" => catalog[3]["file_size"] = json!(-1),
            "fractional_size" => catalog[3]["file_size"] = json!(42.5),
            "string_size" => catalog[3]["file_size"] = json!("42"),
            "missing_size" => {
                catalog[3].as_object_mut().unwrap().remove("file_size");
            }
            _ => unreachable!(),
        }
        fixture.server.set_catalog(catalog);
        assert!(fixture.run().is_err(), "accepted catalog defect {case}");
        fixture.assert_clean();
        assert_eq!(
            fixture.server.requests().len(),
            1,
            "payload requested for {case}"
        );
    }
}

#[test]
fn catalog_shape_syntax_and_streamed_size_are_bounded_before_payloads() {
    for bytes in [b"{".to_vec(), b"{}".to_vec(), b"null".to_vec()] {
        let fixture = Fixture::new();
        fixture.server.state.lock().catalog = bytes;
        fixture.fail();
        assert_eq!(fixture.server.requests().len(), 1);
    }
    // Valid JSON beyond the bound, with no Content-Length, exercises the
    // streaming cap rather than succeeding/failing on parsing or header checks.
    let fixture = Fixture::new();
    let mut padded = fixture.server.state.lock().catalog.clone();
    padded.resize(1024 * 1024 + 1, b' ');
    fixture.server.script(
        "/db_exports.json",
        vec![Reply {
            omit_length: true,
            ..Reply::ok(padded)
        }],
    );
    fixture.fail();
    assert_eq!(fixture.server.requests().len(), 1);
    // A small valid catalog hidden behind a dishonest oversized Content-Length
    // cannot pass the same bound either.
    let fixture = Fixture::new();
    let catalog = fixture.server.state.lock().catalog.clone();
    fixture.server.script(
        "/db_exports.json",
        vec![Reply::short(&catalog, 1024 * 1024 + 1)],
    );
    fixture.fail();
    assert_eq!(fixture.server.requests().len(), 1);
}

#[test]
fn production_export_allowlist_rejects_untrusted_urls_before_payloads() {
    let malicious = [
        "http://static1.e621.net/data/db_export/tag_implications.csv.gz",
        "https://evil.invalid/data/db_export/tag_implications.csv.gz",
        "https://static1.e621.net.evil.invalid/data/db_export/tag_implications.csv.gz",
        "https://user@static1.e621.net/data/db_export/tag_implications.csv.gz",
        "https://user:secret@static1.e621.net/data/db_export/tag_implications.csv.gz",
        "https://static1.e621.net:444/data/db_export/tag_implications.csv.gz",
        "https://static1.e621.net/data/db_export/posts.csv.gz",
        "https://static1.e621.net/data/db_export/tag_implications.csv.gz?mirror=1",
        "https://static1.e621.net/data/db_export/tag_implications.csv.gz#fragment",
        "https://static1.e621.net/other/tag_implications.csv.gz",
        "https://static1.e621.net/data/db_export/%74ag_implications.csv.gz",
        "/data/db_export/tag_implications.csv.gz",
    ];
    for url in malicious {
        let mut fixture = Fixture::new();
        fixture.endpoints.exports = production_endpoints().unwrap().exports;
        let mut catalog = fixture.server.catalog();
        for entry in catalog.as_array_mut().unwrap() {
            let name = entry["name"].as_str().unwrap();
            entry["url"] = json!(format!("https://static1.e621.net{}", path(name)));
        }
        catalog[3]["url"] = json!(url);
        fixture.server.set_catalog(catalog);
        assert!(fixture.run().is_err(), "accepted untrusted URL {url}");
        fixture.assert_clean();
        assert_eq!(fixture.server.requests().len(), 1);
    }
}

#[test]
fn normalized_dot_segments_and_empty_credentials_are_rejected_before_payloads() {
    for raw_path in [
        "/data/db_export/x/../tag_implications.csv.gz",
        "/data/db_export/tag_implications.csv.gz",
    ] {
        let fixture = Fixture::new();
        let authority = if raw_path.contains("/../") {
            fixture.server.address.to_string()
        } else {
            format!("@{}", fixture.server.address)
        };
        let raw = format!("http://{authority}{raw_path}");
        // Both parse successfully into the approved URL. Only the raw-URL
        // admission guard distinguishes them; every earlier export is valid
        // and locally available if resolution wrongly starts downloading.
        assert_eq!(
            reqwest::Url::parse(&raw).unwrap(),
            fixture.server.url(&path("tag_implications"))
        );
        let mut catalog = fixture.server.catalog();
        catalog[3]["url"] = json!(raw);
        fixture.server.set_catalog(catalog);
        fixture.fail();
        assert_eq!(
            fixture
                .server
                .requests()
                .into_iter()
                .map(|request| request.path)
                .collect::<Vec<_>>(),
            ["/db_exports.json"]
        );
    }
}

#[test]
fn strict_dates_and_invalid_user_agents_fail_without_network_activity() {
    for date in [
        "2026-1-02",
        "2026-01-2",
        "2026-02-29",
        "2026-04-31",
        "2026-00-02",
        "2026-13-01",
        "2026-01-00",
        "2026-01-02 ",
        " 2026-01-02",
        "2026-01-02T00:00:00Z",
        "２０２６-01-02",
        "0000-01-01",
        "+2026-01-02",
        "10000-01-01",
    ] {
        let mut fixture = Fixture::new();
        fixture.options.date = date.into();
        fixture.fail();
        assert!(
            fixture.server.requests().is_empty(),
            "requested catalog for invalid date {date}"
        );
    }
    for agent in [
        "",
        " ",
        "\t",
        "bad\r\nInjected: true",
        "bad\0agent",
        "(comment only)",
        "app/",
        "app/1.0(comment)",
    ] {
        let mut fixture = Fixture::new();
        fixture.options.user_agent = agent.into();
        fixture.fail();
        assert!(fixture.server.requests().is_empty());
    }
}

#[test]
fn represented_calendar_date_not_utc_date_controls_selection_and_leap_days_work() {
    for (date, timestamp) in [
        (DATE, UPDATED),
        (DATE, "2026-01-02T00:30:00+14:00"),
        ("2024-02-29", "2024-02-29T23:59:59Z"),
        ("0001-01-01", "0001-01-01T00:00:00Z"),
        ("9999-12-31", "9999-12-31T23:59:59Z"),
    ] {
        let mut fixture = Fixture::new();
        fixture.options.date = date.into();
        let mut catalog = fixture.server.catalog();
        for entry in catalog.as_array_mut().unwrap() {
            entry["updated_at"] = json!(timestamp);
        }
        fixture.server.set_catalog(catalog);
        let report = fixture.run().unwrap();
        let manifest: Value =
            serde_json::from_slice(&fs::read(report.snapshot_path.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["export_date"], date);
        assert!(
            manifest["files"]
                .as_array()
                .unwrap()
                .iter()
                .all(|entry| entry["updated_at"] == timestamp)
        );
    }
}

#[test]
fn redirects_are_not_followed_for_catalog_or_payload() {
    for target in ["/db_exports.json".to_owned(), path("posts")] {
        let fixture = Fixture::new();
        fixture.server.script(
            &target,
            vec![Reply::redirect(
                fixture.server.url("/redirect-target").to_string(),
            )],
        );
        fixture.fail();
        assert_eq!(fixture.server.calls(&target), 1);
        assert_eq!(fixture.server.calls("/redirect-target"), 0);
    }
}

#[test]
fn catalog_transport_and_http_errors_are_not_retried() {
    for status in [403, 404, 429, 503] {
        let fixture = Fixture::new();
        fixture
            .server
            .script("/db_exports.json", vec![Reply::status(status); 3]);
        fixture.fail();
        assert_eq!(fixture.server.requests().len(), 1);
    }
    let fixture = Fixture::new();
    let bytes = fixture.server.state.lock().catalog.clone();
    fixture.server.script(
        "/db_exports.json",
        vec![Reply::short(&bytes[..bytes.len() / 2], bytes.len()); 3],
    );
    fixture.fail();
    assert_eq!(fixture.server.requests().len(), 1);
}

#[test]
fn transient_statuses_connections_and_short_bodies_restart_from_zero() {
    for failure in [
        "429",
        "500",
        "503",
        "disconnect",
        "short_framed",
        "short_complete",
    ] {
        let fixture = Fixture::new();
        let body = fixture.server.state.lock().payloads[&path("posts")].clone();
        let reply = match failure {
            "429" => Reply::status(429),
            "500" => Reply::status(500),
            "503" => Reply::status(503),
            "disconnect" => Reply {
                disconnect: true,
                ..Reply::ok(Vec::new())
            },
            "short_framed" => Reply::short(&body[..body.len() / 2], body.len()),
            "short_complete" => Reply::ok(body[..body.len() / 2].to_vec()),
            _ => unreachable!(),
        };
        fixture
            .server
            .script(&path("posts"), vec![reply.clone(), reply]);
        let report = fixture
            .run()
            .unwrap_or_else(|error| panic!("{failure} did not recover: {error:#}"));
        fixture.assert_manifest(&report.snapshot_path);
        assert_eq!(
            fixture.server.calls(&path("posts")),
            3,
            "wrong attempt bound for {failure}"
        );
        assert_eq!(
            fixture.server.calls("/db_exports.json"),
            1,
            "catalog must remain pinned"
        );
        assert!(fixture.server.requests().iter().all(|r| {
            !r.headers
                .lines()
                .any(|line| line.to_ascii_lowercase().starts_with("range:"))
        }));
    }
}

#[test]
fn exhausted_transients_stop_at_three_attempts_and_remove_owned_stage() {
    for failure in ["429", "503", "disconnect", "short_framed", "short_complete"] {
        let fixture = Fixture::new();
        let body = fixture.server.state.lock().payloads[&path("posts")].clone();
        let reply = match failure {
            "429" => Reply::status(429),
            "503" => Reply::status(503),
            "disconnect" => Reply {
                disconnect: true,
                ..Reply::ok(Vec::new())
            },
            "short_framed" => Reply::short(&body[..body.len() / 2], body.len()),
            "short_complete" => Reply::ok(body[..body.len() / 2].to_vec()),
            _ => unreachable!(),
        };
        fixture.server.script(&path("posts"), vec![reply; 3]);
        fixture.fail();
        assert_eq!(
            fixture.server.calls(&path("posts")),
            3,
            "wrong attempt bound for {failure}"
        );
        assert_eq!(fixture.server.requests().len(), 4);
    }
}

#[test]
fn read_inactivity_timeouts_retry_boundedly_without_sleeping_fixture_threads() {
    for recover in [false, true] {
        let mut fixture = Fixture::new();
        fixture.settings.timeout = Duration::from_secs(1);
        let length = fixture.server.state.lock().payloads[&path("posts")].len();
        let transfers: Vec<_> = (0..if recover { 1 } else { 3 })
            .map(|_| Arc::new(Transfer::default()))
            .collect();
        fixture.server.script(
            &path("posts"),
            transfers
                .iter()
                .map(|transfer| Reply {
                    stall: true,
                    declared: Some(length),
                    transfer: Some(Arc::clone(transfer)),
                    ..Reply::ok(vec![0x1f])
                })
                .collect(),
        );
        let result = thread::scope(|scope| {
            let download = scope.spawn(|| fixture.run());
            // Each real connection must have delivered a body prefix and
            // reached its held-open read stall, not failed during send/setup.
            for transfer in &transfers {
                transfer.wait_for_phase(TransferPhase::Stalled);
            }
            download.join().unwrap()
        });
        let last_stall = transfers
            .last()
            .unwrap()
            .events
            .lock()
            .iter()
            .find(|(phase, _)| *phase == TransferPhase::Stalled)
            .unwrap()
            .1;
        assert!(
            last_stall.elapsed() >= fixture.settings.timeout / 2,
            "held-open response failed before a read-inactivity interval"
        );
        if recover {
            let report = result.unwrap();
            fixture.assert_manifest(&report.snapshot_path);
            assert_eq!(fixture.server.calls(&path("posts")), 2);
        } else {
            assert!(result.is_err());
            fixture.assert_clean();
            assert_eq!(fixture.server.calls(&path("posts")), 3);
        }
        assert_eq!(fixture.server.calls("/db_exports.json"), 1);
        assert!(fixture.server.requests().iter().all(|request| {
            !request.headers.lines().any(|line| {
                line.split_once(':')
                    .is_some_and(|(key, _)| key.eq_ignore_ascii_case("range"))
            })
        }));
    }
}

#[test]
fn a_progressing_transfer_can_exceed_one_operation_timeout() {
    let mut fixture = Fixture::new();
    fixture.settings.timeout = Duration::from_secs(2);
    let bytes = fixture.server.state.lock().payloads[&path("posts")].clone();
    // Writes are paced far inside the read timeout. The proof below uses the
    // observed first-to-last body progress, never the nominal wait count.
    let transfer = Arc::new(Transfer::default());
    fixture.server.script(
        &path("posts"),
        vec![Reply {
            chunk_interval: Some(Duration::from_millis(250)),
            transfer: Some(Arc::clone(&transfer)),
            ..Reply::ok(bytes)
        }],
    );
    let report = thread::scope(|scope| {
        let download = scope.spawn(|| fixture.run());
        transfer.wait_for_phase(TransferPhase::Progress);
        download.join().unwrap().unwrap()
    });
    transfer.wait_for_phase(TransferPhase::Complete);
    let events = transfer.events.lock();
    let mut progress = events
        .iter()
        .filter(|(phase, _)| *phase == TransferPhase::Progress)
        .map(|(_, at)| *at);
    let first = progress.next().unwrap();
    let last = progress.last().expect("body must make further progress");
    assert!(
        last.duration_since(first) > fixture.settings.timeout,
        "actual body transfer did not exceed one operation timeout"
    );
    assert_eq!(events.last().unwrap().0, TransferPhase::Complete);
    fixture.assert_manifest(&report.snapshot_path);
    assert_eq!(fixture.server.calls(&path("posts")), 1);
}

#[test]
fn content_encoding_does_not_change_the_pinned_compressed_bytes() {
    let fixture = Fixture::new();
    let bytes = fixture.server.state.lock().payloads[&path("posts")].clone();
    let mut response = Reply::ok(bytes);
    response
        .headers
        .push(("Content-Encoding".into(), "gzip".into()));
    fixture.server.script(&path("posts"), vec![response]);
    let report = fixture.run().unwrap();
    fixture.assert_manifest(&report.snapshot_path);
    assert_eq!(fixture.server.calls(&path("posts")), 1);
}

#[test]
fn permanent_http_failures_are_not_retried_or_published() {
    for status in [400, 401, 403, 404] {
        let fixture = Fixture::new();
        fixture
            .server
            .script(&path("tags"), vec![Reply::status(status)]);
        fixture.fail();
        assert_eq!(fixture.server.calls(&path("tags")), 1);
        assert_eq!(fixture.server.calls("/db_exports.json"), 1);
    }
}

#[test]
fn same_size_rotation_and_oversized_payloads_fail_without_retry_or_catalog_refresh() {
    for oversized in [false, true] {
        let fixture = Fixture::new();
        let mut bytes = fixture.server.state.lock().payloads[&path("posts")].clone();
        if oversized {
            bytes.extend_from_slice(b"unexpected bytes");
        } else {
            bytes[10] ^= 1;
        }
        fixture.server.script(
            &path("posts"),
            vec![Reply {
                omit_length: oversized,
                ..Reply::ok(bytes)
            }],
        );
        fixture.fail();
        assert_eq!(fixture.server.calls(&path("posts")), 1);
        assert_eq!(fixture.server.calls("/db_exports.json"), 1);
    }
}

#[test]
fn oversized_declared_payload_lengths_are_fatal_even_without_a_large_body() {
    for body in [Vec::new(), vec![0x1f]] {
        let fixture = Fixture::new();
        let pinned_length = fixture.server.state.lock().payloads[&path("posts")].len();
        fixture.server.script(
            &path("posts"),
            vec![Reply {
                declared: Some(pinned_length + 1),
                ..Reply::ok(body)
            }],
        );
        // Without the advertised-size gate this is a retryable short body,
        // followed by the valid default reply. One fatal attempt proves the
        // header guard, not an oversized streamed body or parser rejection.
        fixture.fail();
        assert_eq!(fixture.server.calls(&path("posts")), 1);
        assert_eq!(fixture.server.calls("/db_exports.json"), 1);
        assert_eq!(fixture.server.requests().len(), 2);
    }
}

#[test]
fn pinned_gzip_and_csv_validation_rejects_corruption_headers_and_ragged_rows() {
    let valid = gzip("id,name,category\n10,fox,0\n");
    let mut crc = valid.clone();
    let trailer = crc.len() - 8;
    crc[trailer] ^= 1;
    let mut size = valid.clone();
    let size_trailer = size.len() - 4;
    size[size_trailer] ^= 1;
    let mut later_member = gzip("id,name,category\n10,fox,0\n");
    later_member.extend_from_slice(&crc);
    let cases = [
        ("truncated trailer", valid[..valid.len() - 4].to_vec()),
        ("corrupt CRC", crc),
        ("corrupt uncompressed size", size),
        ("not gzip", b"id,name,category\n10,fox,0\n".to_vec()),
        ("corrupt later member", later_member),
        ("missing required header", gzip("id,name\n10,fox\n")),
        (
            "duplicate required header",
            gzip("id,name,category,name\n10,fox,0,fox\n"),
        ),
        ("ragged short row", gzip("id,name,category\n10,fox\n")),
        (
            "ragged long row",
            gzip("id,name,category\n10,fox,0,extra\n"),
        ),
        ("unclosed quote", gzip("id,name,category\n10,\"fox,0\n")),
        (
            "quote in unquoted field",
            gzip("id,name,category\n10,fo\"x,0\n"),
        ),
        (
            "text after closing quote",
            gzip("id,name,category\n10,\"fox\"junk,0\n"),
        ),
        ("no header", gzip("")),
    ];
    for (description, bytes) in cases {
        let fixture = Fixture::new();
        fixture.server.repin("tags", bytes);
        assert!(fixture.run().is_err(), "accepted {description}");
        fixture.assert_clean();
        assert_eq!(
            fixture.server.calls(&path("tags")),
            1,
            "retried {description}"
        );
    }
}

#[test]
fn each_selected_dataset_enforces_its_own_required_columns() {
    for (name, csv) in [
        (
            "posts",
            "id,rating,file_ext,is_deleted,tag_string,score\n1,s,jpg,f,fox,0\n",
        ),
        (
            "posts",
            "id,md5,rating,file_ext,is_deleted,tag_string\n1,00000000000000000000000000000001,s,jpg,f,fox\n",
        ),
        ("tags", "id,name\n10,fox\n"),
        ("tag_aliases", "antecedent_name,consequent_name\nold,fox\n"),
        (
            "tag_implications",
            "antecedent_name,consequent_name\nfox,canine\n",
        ),
    ] {
        let fixture = Fixture::new();
        fixture.server.repin(name, gzip(csv));
        fixture.fail();
        assert_eq!(fixture.server.calls(&path(name)), 1);
    }
}

#[test]
fn uppercase_catalog_checksums_and_valid_multiline_csv_are_accepted() {
    let fixture = Fixture::new();
    let bytes = gzip(
        "\u{feff}\"id\",name,category\r\n10,\"first\r\nsecond,\"\"quoted\"\"\",0\r\n11,canine,0\r\n",
    );
    fixture.server.repin("tags", bytes.clone());
    let mut catalog = fixture.server.catalog();
    for entry in catalog.as_array_mut().unwrap() {
        entry["checksum"] = json!(entry["checksum"].as_str().unwrap().to_ascii_uppercase());
    }
    fixture.server.set_catalog(catalog);
    let report = fixture.run().unwrap();
    assert_eq!(
        fs::read(report.snapshot_path.join("tags.csv.gz")).unwrap(),
        bytes
    );
    let manifest: Value =
        serde_json::from_slice(&fs::read(report.snapshot_path.join("manifest.json")).unwrap())
            .unwrap();
    let tags = manifest["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["file"] == "tags.csv.gz")
        .unwrap();
    assert_eq!(tags["rows"], 2);
    assert_eq!(tags["sha256"], digest(&bytes));
}

#[test]
fn posts_and_tags_must_have_data_but_relations_may_be_header_only() {
    for (name, header) in [
        (
            "posts",
            "id,md5,rating,file_ext,is_deleted,tag_string,score\n",
        ),
        ("tags", "id,name,category\n"),
    ] {
        let fixture = Fixture::new();
        fixture.server.repin(name, gzip(header));
        fixture.fail();
        assert_eq!(fixture.server.calls(&path(name)), 1);
    }
    let fixture = Fixture::new();
    let report = fixture.run().unwrap();
    let manifest: Value =
        serde_json::from_slice(&fs::read(report.snapshot_path.join("manifest.json")).unwrap())
            .unwrap();
    for name in ["tag_aliases", "tag_implications"] {
        let file = format!("{name}.csv.gz");
        let entry = manifest["files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["file"] == file)
            .unwrap();
        assert_eq!(entry["rows"], 0);
    }
}

#[test]
fn valid_concatenated_gzip_members_are_read_through_actual_eof() {
    let fixture = Fixture::new();
    let mut bytes = gzip("id,name,category\n10,fox,0\n");
    bytes.extend(gzip("11,canine,0\n"));
    fixture.server.repin("tags", bytes.clone());
    let report = fixture.run().unwrap();
    assert_eq!(
        fs::read(report.snapshot_path.join("tags.csv.gz")).unwrap(),
        bytes
    );
    let manifest: Value =
        serde_json::from_slice(&fs::read(report.snapshot_path.join("manifest.json")).unwrap())
            .unwrap();
    let tags = manifest["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["file"] == "tags.csv.gz")
        .unwrap();
    assert_eq!(tags["rows"], 2);
}

fn tree(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fs::read_dir(root)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            assert!(entry.file_type().unwrap().is_file());
            (entry.file_name().into(), fs::read(entry.path()).unwrap())
        })
        .collect()
}

#[test]
fn repeated_snapshot_and_existing_empty_destination_are_never_clobbered() {
    let mut fixture = Fixture::new();
    let report = fixture.run().unwrap();
    let first = tree(&report.snapshot_path);
    assert!(fixture.run().is_err());
    assert_eq!(tree(&report.snapshot_path), first);
    assert_eq!(
        fs::read_dir(&fixture.options.destination).unwrap().count(),
        1
    );

    // Identical catalog bytes pin the same manifest identity under another input
    // root. Even an existing empty target must make new-only publication fail.
    fixture.options.destination = fixture.root.path().join("second-inputs");
    fs::create_dir_all(&fixture.options.destination).unwrap();
    let occupied = fixture
        .options
        .destination
        .join(report.snapshot_path.file_name().unwrap());
    fs::create_dir(&occupied).unwrap();
    assert!(fixture.run().is_err());
    assert!(occupied.is_dir());
    assert_eq!(fs::read_dir(&occupied).unwrap().count(), 0);
    assert_eq!(
        fs::read_dir(&fixture.options.destination).unwrap().count(),
        1
    );
    assert_eq!(tree(&report.snapshot_path), first);
}

#[test]
fn failure_cleanup_preserves_foreign_stages_and_unrelated_files() {
    let fixture = Fixture::new();
    fs::create_dir_all(fixture.options.destination.join(".download-foreign")).unwrap();
    fs::write(
        fixture
            .options
            .destination
            .join(".download-foreign/owned-by-other-run"),
        b"preserve",
    )
    .unwrap();
    fs::write(
        fixture.options.destination.join("unrelated"),
        b"preserve too",
    )
    .unwrap();
    fixture
        .server
        .script(&path("tags"), vec![Reply::status(404)]);
    assert!(fixture.run().is_err());
    assert_eq!(
        fs::read(
            fixture
                .options
                .destination
                .join(".download-foreign/owned-by-other-run")
        )
        .unwrap(),
        b"preserve"
    );
    assert_eq!(
        fs::read(fixture.options.destination.join("unrelated")).unwrap(),
        b"preserve too"
    );
    assert_eq!(
        fs::read_dir(&fixture.options.destination).unwrap().count(),
        2
    );
}

#[test]
fn impossible_capacity_and_unusable_destination_fail_without_payloads() {
    for reserve in [u64::MAX, u64::MAX / 2] {
        let mut fixture = Fixture::new();
        fixture.settings.reserve_bytes = reserve;
        fixture.fail();
        assert_eq!(fixture.server.requests().len(), 1);
    }

    let mut fixture = Fixture::new();
    fixture.options.destination = fixture.root.path().join("not-a-directory");
    fs::write(&fixture.options.destination, b"do not replace").unwrap();
    assert!(fixture.run().is_err());
    assert_eq!(
        fs::read(&fixture.options.destination).unwrap(),
        b"do not replace"
    );
    assert!(
        fixture
            .server
            .requests()
            .iter()
            .all(|r| r.path == "/db_exports.json")
    );
}

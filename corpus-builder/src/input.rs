use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
    str,
};

use anyhow::{Context, Result, bail};
use csv::{ByteRecord, Reader, ReaderBuilder};
use flate2::bufread::MultiGzDecoder;

#[derive(Clone, Copy)]
enum FieldState {
    Start,
    Unquoted,
    Quoted,
    AfterQuote,
}

// The csv crate deliberately accepts malformed quotes. Validate the byte stream
// without copying it, retaining quote state across decompression/read boundaries.
struct StrictCsv<R> {
    inner: R,
    state: FieldState,
    bom_prefix: u8,
}
impl<R: Read> Read for StrictCsv<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        let count = self.inner.read(output)?;
        if count == 0 && matches!(self.state, FieldState::Quoted) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unterminated quoted CSV field",
            ));
        }
        for &byte in &output[..count] {
            // csv::Reader accepts an optional initial UTF-8 BOM. It must not turn
            // a quoted first header into an apparent quote inside an unquoted field.
            match (self.bom_prefix, byte) {
                (0, 0xef) => {
                    self.bom_prefix = 1;
                    continue;
                }
                (1, 0xbb) => {
                    self.bom_prefix = 2;
                    continue;
                }
                (2, 0xbf) => {
                    self.bom_prefix = 3;
                    continue;
                }
                (1 | 2, _) => {
                    self.bom_prefix = 3;
                    self.state = FieldState::Unquoted;
                }
                (0, _) => self.bom_prefix = 3,
                _ => {}
            }
            self.state = match (self.state, byte) {
                (FieldState::Start, b'"') => FieldState::Quoted,
                (
                    FieldState::Start | FieldState::Unquoted | FieldState::AfterQuote,
                    b',' | b'\n' | b'\r',
                ) => FieldState::Start,
                (FieldState::Start, _) => FieldState::Unquoted,
                (FieldState::Unquoted, b'"') => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Quote inside unquoted CSV field",
                    ));
                }
                (FieldState::Unquoted, _) => FieldState::Unquoted,
                (FieldState::Quoted, b'"') => FieldState::AfterQuote,
                (FieldState::Quoted, _) => FieldState::Quoted,
                (FieldState::AfterQuote, b'"') => FieldState::Quoted,
                (FieldState::AfterQuote, _) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Unexpected character after closing CSV quote",
                    ));
                }
            };
        }
        Ok(count)
    }
}

pub struct GzipCsv {
    pub headers: ByteRecord,
    reader: Reader<StrictCsv<MultiGzDecoder<BufReader<File>>>>,
}

pub fn open(path: &Path, required: &[&str]) -> Result<GzipCsv> {
    let file = File::open(path)
        .with_context(|| format!("Opening required CSV input {}", path.display()))?;
    let gzip = MultiGzDecoder::new(BufReader::with_capacity(1 << 20, file));
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .buffer_capacity(1 << 20)
        .from_reader(StrictCsv {
            inner: gzip,
            state: FieldState::Start,
            bom_prefix: 0,
        });
    let headers = reader
        .byte_headers()
        .with_context(|| format!("Reading CSV headers from {}", path.display()))?
        .clone();
    for header in &headers {
        str::from_utf8(header).context("CSV header is not valid UTF-8")?;
    }
    let csv = GzipCsv { headers, reader };
    for name in required {
        csv.column(name)?;
    }
    Ok(csv)
}

impl GzipCsv {
    pub fn column(&self, name: &str) -> Result<usize> {
        let mut matches = self
            .headers
            .iter()
            .enumerate()
            .filter_map(|(index, header)| (header == name.as_bytes()).then_some(index));
        let Some(index) = matches.next() else {
            bail!("CSV input has no '{name}' column.");
        };
        if matches.next().is_some() {
            bail!("CSV input has multiple '{name}' columns.");
        }
        Ok(index)
    }

    // Call until false: gzip checksums/trailers are verified by reading actual EOF.
    // The caller owns the record so its field storage is reused across rows.
    pub fn read(&mut self, record: &mut ByteRecord) -> Result<bool> {
        self.reader
            .read_byte_record(record)
            .context("Reading gzip CSV record")
    }
}

pub fn text(record: &ByteRecord, column: usize) -> Result<&str> {
    let value = record
        .get(column)
        .with_context(|| format!("CSV record has no field at column {column}"))?;
    str::from_utf8(value).with_context(|| format!("CSV column {column} is not valid UTF-8"))
}

pub fn integer(record: &ByteRecord, column: usize) -> Result<i64> {
    let value = text(record, column)?;
    value
        .trim()
        .parse::<i64>()
        .with_context(|| format!("CSV column {column} is not an Int64: '{value}'"))
}

#[cfg(test)]
#[path = "../tests/unit/input.rs"]
mod tests;

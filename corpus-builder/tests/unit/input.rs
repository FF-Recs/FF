use std::{fs, io::Write};

use flate2::{Compression, write::GzEncoder};

use super::*;

fn gzip(path: &Path, bytes: &[u8]) {
    let mut encoder = GzEncoder::new(File::create(path).unwrap(), Compression::fast());
    encoder.write_all(bytes).unwrap();
    encoder.finish().unwrap();
}

#[test]
fn reusable_records_preserve_multiline_fields_and_checked_integers() {
    let directory = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let path = directory.path().join("input.csv.gz");
    gzip(&path, "id,name\r\n\" \t\u{2003}+42\u{2003}\r \",\"first\nsecond,\"\"quoted\"\"\"\r\n-9223372036854775808,last\r\n".as_bytes());
    let mut csv = open(&path, &["id", "name"]).unwrap();
    let id = csv.column("id").unwrap();
    let name = csv.column("name").unwrap();
    let mut row = ByteRecord::new();
    assert!(csv.read(&mut row).unwrap());
    assert_eq!(integer(&row, id).unwrap(), 42);
    assert_eq!(text(&row, name).unwrap(), "first\nsecond,\"quoted\"");
    assert!(csv.read(&mut row).unwrap());
    assert_eq!(integer(&row, id).unwrap(), i64::MIN);
    assert_eq!(text(&row, name).unwrap(), "last");
    assert!(!csv.read(&mut row).unwrap());
    for invalid in [
        "9223372036854775808",
        "-9223372036854775809",
        "1.0",
        "1 2",
        "",
        "+",
        "１２",
    ] {
        assert!(
            integer(&ByteRecord::from(vec![invalid]), 0).is_err(),
            "{invalid}"
        );
    }
    let invalid_utf8 = ByteRecord::from(vec![&b"\xff"[..]]);
    assert!(text(&invalid_utf8, 0).is_err());
    assert!(text(&ByteRecord::new(), 0).is_err());
}

#[test]
fn rejects_missing_duplicate_headers_ragged_rows_and_broken_gzip() {
    let directory = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let path = directory.path().join("input.csv.gz");
    for bytes in [b"name\nvalue\n".as_slice(), b"id,id\n1,2\n".as_slice()] {
        gzip(&path, bytes);
        assert!(open(&path, &["id"]).is_err());
    }
    for bytes in [b"id,name\n1\n".as_slice(), b"id,name\n1,a,b\n".as_slice()] {
        gzip(&path, bytes);
        let mut csv = open(&path, &["id"]).unwrap();
        assert!(csv.read(&mut ByteRecord::new()).is_err());
    }
    gzip(&path, b"id,name\n1,ok\n");
    let original = fs::read(&path).unwrap();
    let mut corrupt = original.clone();
    let crc = corrupt.len() - 8;
    corrupt[crc] ^= 1;
    for bytes in [&corrupt[..], &original[..original.len() - 4]] {
        fs::write(&path, bytes).unwrap();
        let result = (|| -> Result<()> {
            let mut csv = open(&path, &["id"])?;
            let mut row = ByteRecord::new();
            while csv.read(&mut row)? {}
            Ok(())
        })();
        assert!(result.is_err());
    }
}

#[test]
fn malformed_quoting_cannot_be_accepted_as_valid_fields_at_eof() {
    let directory = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let path = directory.path().join("input.csv.gz");
    for text in [
        "id,name\n1,\"unterminated",
        "id,name\n1,\"closed\"junk\n",
        "id,name\n1,unquoted\"quote\n",
    ] {
        gzip(&path, text.as_bytes());
        let result = (|| -> Result<()> {
            let mut csv = open(&path, &["id", "name"])?;
            let mut row = ByteRecord::new();
            while csv.read(&mut row)? {}
            Ok(())
        })();
        assert!(result.is_err(), "Accepted malformed quoting: {text:?}");
    }
}

#[test]
fn quoted_fields_preserve_bom_escapes_and_multiline_across_read_buffers() {
    let directory = tempfile::Builder::new()
        .prefix("ff-rust-corpus-test-")
        .tempdir()
        .unwrap();
    let path = directory.path().join("input.csv.gz");
    let prefix = "x".repeat((1 << 20) + 7);
    gzip(
        &path,
        format!("\u{feff}\"id\",name\r\n1,\"{prefix}\"\"tail\nline\"\r\n").as_bytes(),
    );
    let mut csv = open(&path, &["id", "name"]).unwrap();
    let mut row = ByteRecord::new();
    assert!(csv.read(&mut row).unwrap());
    assert_eq!(integer(&row, 0).unwrap(), 1);
    assert_eq!(text(&row, 1).unwrap(), format!("{prefix}\"tail\nline"));
    assert!(!csv.read(&mut row).unwrap());
}

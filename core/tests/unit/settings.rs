use super::*;

fn fixture() -> Result<tempfile::TempDir> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.workdir/ff-rs/settings");
    fs::create_dir_all(&root)?;
    Ok(tempfile::tempdir_in(root)?)
}

#[test]
fn old_settings_without_playback_keep_muted_and_reject_newer_versions() {
    let directory = fixture().unwrap();
    let path = directory.path().join("config.toml");
    fs::write(
        &path,
        "version=1\n[input]\nleft_hand_mode=true\n[auto]\nselection='fresh'\ndelay_seconds=99\n",
    )
    .unwrap();
    let loaded = load(&path).unwrap();
    assert!(loaded.left_handed && loaded.video_muted);
    assert_eq!(loaded.auto_preset, "fresh");
    assert_eq!(loaded.auto_delay_seconds, 30);
    fs::write(&path, "version=1\n[input]\n[auto]\ndelay_seconds=-1\n").unwrap();
    assert_eq!(load(&path).unwrap().auto_delay_seconds, 2);
    fs::write(&path, "version=2\n[input]\n[auto]\n").unwrap();
    assert!(load(&path).is_err());
}

#[test]
fn missing_settings_use_runtime_defaults_and_start_muted() -> Result<()> {
    let directory = fixture()?;
    let loaded = load(&directory.path().join("absent.toml"))?;
    assert_eq!(loaded, UserSettings::default());
    assert!(loaded.video_muted);
    Ok(())
}

#[test]
fn imports_presets_explicit_playback_and_in_range_intervals() -> Result<()> {
    let directory = fixture()?;
    let path = directory.path().join("config.toml");
    for preset in ["favorites", "balanced", "fresh"] {
        for (left_handed, delay) in [(false, 2), (true, 13), (false, 30)] {
            fs::write(
                &path,
                format!(
                    "version=1\n[input]\nleft_hand_mode={left_handed}\n\
                         [playback]\nmuted=false\n[auto]\nselection='{preset}'\ndelay_seconds={delay}\n"
                ),
            )?;
            let loaded = load(&path)?;
            assert_eq!(loaded.left_handed, left_handed);
            assert!(!loaded.video_muted);
            assert_eq!(loaded.auto_preset, preset);
            assert_eq!(loaded.auto_delay_seconds, delay);
        }
    }
    // Required sections may be empty; their fields, unlike the sections,
    // are optional in the supported version-1 import.
    fs::write(&path, "version=1\n[input]\n[playback]\n[auto]\n")?;
    assert!(load(&path)?.video_muted);
    Ok(())
}

#[test]
fn imports_clamp_intervals_at_both_endpoints() -> Result<()> {
    let directory = fixture()?;
    let path = directory.path().join("config.toml");
    for (input, expected) in [(i32::MIN, 2), (1, 2), (31, 30), (i32::MAX, 30)] {
        fs::write(
            &path,
            format!("version=1\n[input]\n[auto]\nselection='balanced'\ndelay_seconds={input}\n"),
        )?;
        assert_eq!(load(&path)?.auto_delay_seconds, expected);
    }
    Ok(())
}

#[test]
fn rejects_unknown_presets_malformed_toml_and_missing_required_sections() -> Result<()> {
    let directory = fixture()?;
    let path = directory.path().join("config.toml");
    // All unrelated required fields are valid, so preset validation is the
    // rejecting gate rather than a parse or missing-section failure.
    fs::write(
        &path,
        "version=1\n[input]\n[auto]\nselection='unsupported'\ndelay_seconds=10\n",
    )?;
    let error = load(&path).unwrap_err();
    assert!(error.downcast_ref::<toml::de::Error>().is_none());
    for input in [
        "version=1\n[input]\n[auto]\nselection=[\n",
        "version=1\n[auto]\nselection='balanced'\n",
        "version=1\n[input]\nleft_hand_mode=false\n",
        "[input]\n[auto]\nselection='balanced'\n",
    ] {
        fs::write(&path, input)?;
        let error = load(&path).unwrap_err();
        assert!(error.downcast_ref::<toml::de::Error>().is_some());
    }
    Ok(())
}

#[test]
fn existing_unreadable_imports_do_not_fall_back_to_defaults() -> Result<()> {
    let directory = fixture()?;
    // A directory fails reading even under a privileged test runner, unlike
    // a permissions-only fixture. Invalid UTF-8 is a second read failure.
    let error = load(directory.path()).unwrap_err();
    assert!(error.downcast_ref::<std::io::Error>().is_some());
    let path = directory.path().join("invalid-utf8.toml");
    fs::write(&path, [0xff])?;
    let error = load(&path).unwrap_err();
    assert!(matches!(
        error.downcast_ref::<std::io::Error>(),
        Some(error) if error.kind() == std::io::ErrorKind::InvalidData
    ));
    Ok(())
}

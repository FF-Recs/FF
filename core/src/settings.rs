use crate::types::UserSettings;
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Deserialize)]
struct Document {
    version: u32,
    input: Input,
    #[serde(default)]
    playback: Playback,
    auto: Auto,
}
#[derive(Default, Deserialize)]
#[serde(default)]
struct Input {
    left_hand_mode: bool,
}
#[derive(Deserialize)]
#[serde(default)]
struct Playback {
    muted: bool,
}
impl Default for Playback {
    fn default() -> Self {
        Self { muted: true }
    }
}
#[derive(Deserialize)]
#[serde(default)]
struct Auto {
    selection: String,
    delay_seconds: i32,
}
impl Default for Auto {
    fn default() -> Self {
        Self {
            selection: "balanced".into(),
            delay_seconds: 5,
        }
    }
}

pub fn validate(mut settings: UserSettings) -> Result<UserSettings> {
    ensure!(
        matches!(
            settings.auto_preset.as_str(),
            "favorites" | "balanced" | "fresh"
        ),
        "Unknown Auto preset '{}'",
        settings.auto_preset
    );
    settings.auto_delay_seconds = settings.auto_delay_seconds.clamp(2, 30);
    Ok(settings)
}

pub fn load(path: &Path) -> Result<UserSettings> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(UserSettings::default());
        }
        Err(error) => {
            return Err(error).with_context(|| format!("Read settings {}", path.display()));
        }
    };
    let document: Document =
        toml::from_str(&text).with_context(|| format!("Parse settings {}", path.display()))?;
    ensure!(
        document.version == 1,
        "Unsupported settings version {}",
        document.version
    );
    validate(UserSettings {
        left_handed: document.input.left_hand_mode,
        video_muted: document.playback.muted,
        auto_preset: document.auto.selection,
        auto_delay_seconds: document.auto.delay_seconds.clamp(2, 30) as u32,
        blacklist: Vec::new(),
    })
}

#[cfg(test)]
#[path = "../tests/unit/settings.rs"]
mod tests;

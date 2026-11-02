use anyhow::{Context, Result};
use ff_core::{
    app::{AppConfig, AppHandle},
    model,
    store::Store,
    types::{AutoStartRequest, GalleryStartRequest, Grade, UserSettings},
};
use std::collections::HashSet;

mod lifecycle;
mod settings;
mod support;

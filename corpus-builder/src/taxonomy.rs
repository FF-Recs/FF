use std::{collections::HashMap, path::Path, sync::Arc};

use anyhow::{Context, Result, bail};
use csv::ByteRecord;

use crate::{contract::TagDefinition, input};

pub struct Taxonomy {
    pub tags: Vec<TagDefinition>,
    pub canonical: Vec<usize>,
    pub implications: Vec<Vec<usize>>,
    pub aliases: Vec<(String, usize)>,
    pub active_aliases: usize,
    pub active_implications: usize,
    names: HashMap<Arc<str>, usize>,
}

impl Taxonomy {
    pub fn normalizer(&self) -> Normalizer<'_> {
        Normalizer {
            taxonomy: self,
            seen: vec![0; self.tags.len()],
            generation: 0,
            values: Vec::with_capacity(128),
        }
    }
}

pub struct Normalizer<'a> {
    taxonomy: &'a Taxonomy,
    seen: Vec<u32>,
    generation: u32,
    values: Vec<usize>,
}

impl Normalizer<'_> {
    /// Return unique canonical tags and their transitive implications in unspecified order.
    /// The returned slice borrows storage reused by the next normalization.
    pub fn normalize(&mut self, input: &str) -> Result<&[usize]> {
        if self.generation == u32::MAX {
            self.seen.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
        self.values.clear();
        // Production separates only ASCII spaces: tabs/newlines are part of a name.
        for name in input.split(' ').filter(|name| !name.is_empty()) {
            let source = self
                .taxonomy
                .names
                .get(name)
                .with_context(|| format!("Post contains unknown tag '{name}'."))?;
            let canonical = self.taxonomy.canonical[*source];
            // Already-seen canonical tags have had their full transitive closure
            // added, including when they were implied by an earlier input tag.
            if self.seen[canonical] == self.generation {
                continue;
            }
            self.seen[canonical] = self.generation;
            self.values.push(canonical);
            for &implied in &self.taxonomy.implications[canonical] {
                if self.seen[implied] != self.generation {
                    self.seen[implied] = self.generation;
                    self.values.push(implied);
                }
            }
        }
        Ok(&self.values)
    }
}

fn read_tags(path: &Path) -> Result<(Vec<TagDefinition>, HashMap<Arc<str>, usize>)> {
    let mut csv = input::open(path, &["id", "name", "category"])?;
    let id = csv.column("id")?;
    let name = csv.column("name")?;
    let category = csv.column("category")?;
    let mut tags = Vec::new();
    let mut names = HashMap::new();
    let mut row = ByteRecord::new();
    while csv.read(&mut row)? {
        let name: Arc<str> = Arc::from(input::text(&row, name)?);
        let index = tags.len();
        if names.insert(Arc::clone(&name), index).is_some() {
            bail!("Duplicate tag name '{name}'.");
        }
        tags.push(TagDefinition {
            source_id: input::integer(&row, id)?,
            name,
            category: i32::try_from(input::integer(&row, category)?)
                .context("Tag category is outside Int32 range")?,
        });
    }
    Ok((tags, names))
}

// Resolve each alias chain once. The iterative walk handles long chains without
// recursion or a fresh visited allocation for every one of the source tags.
fn read_aliases(
    path: &Path,
    names: &HashMap<Arc<str>, usize>,
) -> Result<(HashMap<String, usize>, usize)> {
    let mut csv = input::open(path, &["antecedent_name", "consequent_name", "status"])?;
    let antecedent = csv.column("antecedent_name")?;
    let consequent = csv.column("consequent_name")?;
    let status = csv.column("status")?;
    let mut edges: HashMap<String, usize> = HashMap::new();
    let mut targets: Vec<String> = Vec::new();
    let mut active = 0;
    let mut row = ByteRecord::new();
    while csv.read(&mut row)? {
        if input::text(&row, status)? != "active" {
            continue;
        }
        active += 1;
        let source = input::text(&row, antecedent)?;
        let target = input::text(&row, consequent)?;
        if let Some(&index) = edges.get(source) {
            if targets[index] != target {
                bail!("Alias '{source}' has multiple active targets.");
            }
        } else {
            edges.insert(source.to_owned(), targets.len());
            targets.push(target.to_owned());
        }
    }

    let mut next = Vec::with_capacity(targets.len());
    let mut terminal = vec![usize::MAX; targets.len()];
    for (index, target) in targets.iter().enumerate() {
        if let Some(&target_index) = edges.get(target.as_str()) {
            next.push(Some(target_index));
        } else {
            terminal[index] = *names
                .get(target.as_str())
                .with_context(|| format!("Alias target '{target}' is absent from tags.csv.gz."))?;
            next.push(None);
        }
    }
    let mut state = vec![0u8; targets.len()];
    let mut stack = Vec::new();
    for start in 0..targets.len() {
        if state[start] == 2 {
            continue;
        }
        let mut current = start;
        loop {
            if state[current] == 2 {
                break;
            }
            if state[current] == 1 {
                bail!("Active alias cycle includes '{}'.", targets[current]);
            }
            state[current] = 1;
            stack.push(current);
            match next[current] {
                Some(target) => current = target,
                None => break,
            }
        }
        let target = terminal[current];
        for index in stack.drain(..).rev() {
            terminal[index] = target;
            state[index] = 2;
        }
    }
    Ok((
        edges
            .into_iter()
            .map(|(name, index)| (name, terminal[index]))
            .collect(),
        active,
    ))
}

fn read_implications(
    path: &Path,
    names: &HashMap<Arc<str>, usize>,
    aliases: &HashMap<String, usize>,
    tag_count: usize,
) -> Result<(Vec<Vec<usize>>, usize)> {
    let resolve = |name: &str| aliases.get(name).or_else(|| names.get(name)).copied();
    let mut csv = input::open(path, &["antecedent_name", "consequent_name", "status"])?;
    let antecedent = csv.column("antecedent_name")?;
    let consequent = csv.column("consequent_name")?;
    let status = csv.column("status")?;
    let mut adjacency = vec![Vec::new(); tag_count];
    let mut row = ByteRecord::new();
    while csv.read(&mut row)? {
        if input::text(&row, status)? != "active" {
            continue;
        }
        let left = resolve(input::text(&row, antecedent)?);
        let target_name = input::text(&row, consequent)?;
        if let Some(left) = left {
            let right = resolve(target_name).with_context(|| {
                format!("Implication target '{target_name}' is absent from tags.csv.gz.")
            })?;
            if left != right {
                adjacency[left].push(right);
            }
        }
    }
    let mut active = 0;
    let mut indegree = vec![0usize; tag_count];
    for targets in &mut adjacency {
        targets.sort_unstable();
        targets.dedup();
        active += targets.len();
        for &target in targets.iter() {
            indegree[target] += 1;
        }
    }
    // The topological order also acts as the queue, without a second allocation.
    let mut order = Vec::with_capacity(tag_count);
    for (index, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            order.push(index);
        }
    }
    let mut offset = 0;
    while offset < order.len() {
        let source = order[offset];
        offset += 1;
        for &target in &adjacency[source] {
            indegree[target] -= 1;
            if indegree[target] == 0 {
                order.push(target);
            }
        }
    }
    if order.len() != tag_count {
        bail!("The canonical active implication graph contains a cycle.");
    }
    let mut closure: Vec<Vec<usize>> = vec![Vec::new(); tag_count];
    let mut seen = vec![0usize; tag_count];
    for &source in order.iter().rev() {
        if adjacency[source].is_empty() {
            continue;
        }
        let mut descendants = Vec::new();
        let generation = source + 1;
        for &target in &adjacency[source] {
            if seen[target] != generation {
                seen[target] = generation;
                descendants.push(target);
            }
            for &descendant in &closure[target] {
                if seen[descendant] != generation {
                    seen[descendant] = generation;
                    descendants.push(descendant);
                }
            }
        }
        descendants.sort_unstable();
        closure[source] = descendants;
    }
    Ok((closure, active))
}

pub fn load(tags: &Path, aliases: &Path, implications: &Path) -> Result<Taxonomy> {
    let (tags, names) = read_tags(tags)?;
    let (alias_targets, active_aliases) = read_aliases(aliases, &names)?;
    let canonical = tags
        .iter()
        .enumerate()
        .map(|(index, tag)| {
            alias_targets
                .get(tag.name.as_ref())
                .copied()
                .unwrap_or(index)
        })
        .collect();
    let (implications, active_implications) =
        read_implications(implications, &names, &alias_targets, tags.len())?;
    let mut aliases: Vec<_> = alias_targets.into_iter().collect();
    aliases.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    Ok(Taxonomy {
        tags,
        canonical,
        implications,
        aliases,
        active_aliases,
        active_implications,
        names,
    })
}

#[cfg(test)]
#[path = "../tests/unit/taxonomy.rs"]
mod tests;

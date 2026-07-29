use crate::embedding::matryoshka64;
use crate::record::MemoryRecord;
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use chrono::{DateTime, Utc};
use nalgebra::{UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use uuid::Uuid;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SplatGeometry {
    pub position: [f32; 3],
    pub scale: [f32; 3],
    pub rotation: [f32; 4],
    pub color_rgba: [u8; 4],
    pub physics_props: [u8; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Splat {
    pub memory_id: Uuid,
    pub semantics: Vec<f32>,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub scale: [f32; 3],
    pub rotation: [f32; 4],
    pub color_rgba: [u8; 4],
    pub mass: f32,
    pub radiance: f32,
    pub domain: String,
    pub basin_id: Option<String>,
    #[serde(default)]
    pub lineage: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Basin {
    pub id: String,
    pub parent_id: Option<String>,
    pub label: String,
    pub path: String,
    pub summary: String,
    pub label_state: String,
    pub stability: f32,
    pub centroid: [f32; 3],
    pub member_ids: Vec<Uuid>,
    pub representative_ids: Vec<Uuid>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotState {
    pub version: u32,
    pub dream_cycle: u64,
    pub last_dream_at: Option<DateTime<Utc>>,
    pub kinetic_energy: f32,
    pub splats: Vec<Splat>,
    pub basins: Vec<Basin>,
}

impl Default for HotState {
    fn default() -> Self {
        Self {
            version: 1,
            dream_cycle: 0,
            last_dream_at: None,
            kinetic_energy: 0.0,
            splats: Vec::new(),
            basins: Vec::new(),
        }
    }
}

impl Splat {
    pub fn from_embedding(record: &MemoryRecord, full_embedding: &[f32]) -> Result<Self> {
        let semantics = matryoshka64(full_embedding)?;
        let position = project_3d(&semantics);
        let direction = Vector3::new(position[0], position[1], position[2])
            .try_normalize(1e-6)
            .unwrap_or_else(Vector3::x);
        let rotation = UnitQuaternion::rotation_between(&Vector3::x(), &direction)
            .unwrap_or_else(UnitQuaternion::identity);
        let quaternion = rotation.quaternion();
        Ok(Self {
            memory_id: record.id,
            semantics,
            position,
            velocity: [0.0; 3],
            scale: [1.35, 0.38, 0.38],
            rotation: [quaternion.i, quaternion.j, quaternion.k, quaternion.w],
            color_rgba: domain_color(&record.domain),
            mass: 1.0,
            radiance: 1.0,
            domain: record.domain.clone(),
            basin_id: None,
            lineage: vec![record.id],
        })
    }

    pub fn packed(&self) -> SplatGeometry {
        SplatGeometry {
            position: self.position,
            scale: self.scale,
            rotation: self.rotation,
            color_rgba: self.color_rgba,
            physics_props: [
                unit_byte(self.radiance / (self.radiance + 1.0)),
                unit_byte(self.mass / (self.mass + 1.0)),
                self.basin_id.is_some() as u8,
                0,
            ],
        }
    }
}

impl HotState {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid hot state {}", path.display()))
    }

    pub fn save(&self, json_path: &Path, packed_path: &Path) -> Result<()> {
        if let Some(parent) = json_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if let Some(parent) = packed_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json_temp = json_path.with_extension("json.tmp");
        {
            let file = File::create(&json_temp)?;
            serde_json::to_writer(BufWriterWithSync::new(file), self)?;
        }
        fs::rename(json_temp, json_path)?;

        let packed_temp = packed_path.with_extension("bin.tmp");
        let mut file = File::create(&packed_temp)?;
        file.write_all(b"SPLTRAG\0")?;
        file.write_all(&1_u32.to_le_bytes())?;
        file.write_all(&(self.splats.len() as u64).to_le_bytes())?;
        file.write_all(&(std::mem::size_of::<SplatGeometry>() as u32).to_le_bytes())?;
        for splat in &self.splats {
            file.write_all(bytemuck::bytes_of(&splat.packed()))?;
        }
        file.sync_all()?;
        fs::rename(packed_temp, packed_path)?;
        Ok(())
    }

    pub fn add_embeddings(
        &mut self,
        records: &[MemoryRecord],
        embeddings: &[Vec<f32>],
    ) -> Result<usize> {
        if records.len() != embeddings.len() {
            anyhow::bail!("hot geometry record/vector count mismatch");
        }
        let existing: HashMap<_, _> = self
            .splats
            .iter()
            .enumerate()
            .map(|(index, splat)| (splat.memory_id, index))
            .collect();
        let mut added = 0;
        for (record, embedding) in records.iter().zip(embeddings) {
            if existing.contains_key(&record.id) {
                continue;
            }
            self.splats.push(Splat::from_embedding(record, embedding)?);
            added += 1;
        }
        self.splats.sort_by_key(|splat| splat.memory_id);
        Ok(added)
    }

    pub fn splat_map(&self) -> HashMap<Uuid, &Splat> {
        self.splats
            .iter()
            .map(|splat| (splat.memory_id, splat))
            .collect()
    }

    pub fn basin_map(&self) -> HashMap<&str, &Basin> {
        self.basins
            .iter()
            .map(|basin| (basin.id.as_str(), basin))
            .collect()
    }
}

struct BufWriterWithSync {
    inner: std::io::BufWriter<File>,
}

impl BufWriterWithSync {
    fn new(file: File) -> Self {
        Self {
            inner: std::io::BufWriter::new(file),
        }
    }
}

impl Write for BufWriterWithSync {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buffer)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl Drop for BufWriterWithSync {
    fn drop(&mut self) {
        let _ = self.inner.flush();
        let _ = self.inner.get_ref().sync_all();
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

fn project_3d(vector: &[f32]) -> [f32; 3] {
    let mut projected = [0.0_f32; 3];
    for (index, value) in vector.iter().enumerate() {
        let phase = index as f32 + 1.0;
        projected[0] += value * (phase * 0.731).sin();
        projected[1] += value * (phase * 1.113).cos();
        projected[2] += value * (phase * 1.733).sin();
    }
    let point = Vector3::new(projected[0], projected[1], projected[2])
        .try_normalize(1e-6)
        .unwrap_or_else(Vector3::zeros)
        * 6.0;
    [point.x, point.y, point.z]
}

fn domain_color(domain: &str) -> [u8; 4] {
    let digest = crate::record::sha256_hex(domain.as_bytes());
    let byte = |offset: usize| u8::from_str_radix(&digest[offset..offset + 2], 16).unwrap_or(128);
    [
        byte(0).saturating_div(2).saturating_add(80),
        byte(2).saturating_div(2).saturating_add(80),
        byte(4).saturating_div(2).saturating_add(80),
        220,
    ]
}

fn unit_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_geometry_keeps_historical_48_byte_contract() {
        assert_eq!(std::mem::size_of::<SplatGeometry>(), 48);
    }
}

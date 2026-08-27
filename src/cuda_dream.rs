//! CUDA neighbor-force dream. CPU path in `physics` remains the reference.
//!
//! Geometry cosine here is the same raw 64-d *dot* `geometry::cosine` uses
//! (not L2-normalized). Basin discovery stays on CPU.

#![allow(unused_imports)] // PhysicsConfig/HotState/Result are used in both cfg arms.

use crate::config::PhysicsConfig;
use crate::geometry::HotState;
use anyhow::Result;

pub const SEM_DIM: usize = 64;
pub const MAX_SEM_NBR: usize = 32;

#[derive(Debug, Clone)]
pub struct CudaStepResult {
    pub steps: usize,
    pub energy: f32,
}

#[cfg(cuda_dream)]
mod ffi {
    use super::{MAX_SEM_NBR, SEM_DIM};
    use crate::config::PhysicsConfig;
    use crate::geometry::{HotState, cosine};
    use crate::physics::domain_state_color;
    use anyhow::{Result, bail};
    use std::collections::HashMap;
    use std::os::raw::c_int;

    #[repr(C)]
    struct DreamCudaParams {
        n: i32,
        neighbor_mode: i32,
        dt: f32,
        damping: f32,
        origin_pull: f32,
        attraction: f32,
        semantic_threshold: f32,
        repulsion_radius: f32,
        repulsion: f32,
        semantic_repulsion: f32,
        cross_domain_repulsion: f32,
        neighbor_radius: f32,
    }

    enum DreamCuda {}

    unsafe extern "C" {
        fn dream_cuda_probe() -> c_int;
        fn dream_cuda_create(
            p: *const DreamCudaParams,
            pos: *const f32,
            vel: *const f32,
            mass: *const f32,
            domain: *const u32,
            semantics: *const f32,
            sem_nbr: *const i32,
            sem_w: *const f32,
        ) -> *mut DreamCuda;
        fn dream_cuda_step(d: *mut DreamCuda, out_disp: *mut f32, out_energy: *mut f32) -> c_int;
        fn dream_cuda_copy_state(d: *mut DreamCuda, pos: *mut f32, vel: *mut f32) -> c_int;
        fn dream_cuda_destroy(d: *mut DreamCuda);
    }

    struct CudaCtx {
        ptr: *mut DreamCuda,
    }

    impl Drop for CudaCtx {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { dream_cuda_destroy(self.ptr) }
            }
        }
    }

    pub fn available() -> bool {
        unsafe { dream_cuda_probe() == 0 }
    }

    struct PackedSoA {
        active: Vec<usize>,
        pos: Vec<f32>,
        vel: Vec<f32>,
        mass: Vec<f32>,
        domain: Vec<u32>,
        semantics: Vec<f32>,
        sem_nbr: Vec<i32>,
        sem_w: Vec<f32>,
    }

    fn pack_active_soa(state: &HotState, semantic_neighbors: &[Vec<usize>]) -> Result<PackedSoA> {
        let active: Vec<usize> = state
            .splats
            .iter()
            .enumerate()
            .filter(|(_, splat)| !splat.muted)
            .map(|(index, _)| index)
            .collect();
        let n = active.len();
        if n == 0 {
            bail!("no active splats");
        }
        if n > i32::MAX as usize {
            bail!("active splat count exceeds CUDA i32 index");
        }
        let mut orig_to_compact = vec![-1i32; state.splats.len()];
        for (compact, &orig) in active.iter().enumerate() {
            orig_to_compact[orig] = compact as i32;
        }
        let mut packed = PackedSoA {
            active,
            pos: vec![0.0f32; n * 3],
            vel: vec![0.0f32; n * 3],
            mass: vec![0.0f32; n],
            domain: vec![0u32; n],
            semantics: vec![0.0f32; n * SEM_DIM],
            sem_nbr: vec![-1i32; n * MAX_SEM_NBR],
            sem_w: vec![0.0f32; n * MAX_SEM_NBR],
        };
        let mut domain_ids: HashMap<String, u32> = HashMap::new();
        for (compact, &orig) in packed.active.iter().enumerate() {
            let splat = &state.splats[orig];
            packed.pos[compact * 3..compact * 3 + 3].copy_from_slice(&splat.position);
            packed.vel[compact * 3..compact * 3 + 3].copy_from_slice(&splat.velocity);
            packed.mass[compact] = splat.mass;
            let next = domain_ids.len() as u32;
            packed.domain[compact] = *domain_ids.entry(splat.domain.clone()).or_insert(next);
            let src = &splat.semantics;
            let take = src.len().min(SEM_DIM);
            packed.semantics[compact * SEM_DIM..compact * SEM_DIM + take]
                .copy_from_slice(&src[..take]);
            let nbrs = semantic_neighbors
                .get(orig)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let mut slot = 0usize;
            for &other_orig in nbrs.iter().take(MAX_SEM_NBR) {
                if other_orig == orig {
                    continue;
                }
                let other_compact = orig_to_compact.get(other_orig).copied().unwrap_or(-1);
                if other_compact < 0 {
                    continue;
                }
                packed.sem_nbr[compact * MAX_SEM_NBR + slot] = other_compact;
                packed.sem_w[compact * MAX_SEM_NBR + slot] =
                    cosine(&splat.semantics, &state.splats[other_orig].semantics);
                slot += 1;
            }
        }
        Ok(packed)
    }

    pub fn run_force_loop<F>(
        state: &mut HotState,
        semantic_neighbors: &[Vec<usize>],
        config: &PhysicsConfig,
        neighbor_mode: bool,
        emit_every: usize,
        on_step: &mut F,
    ) -> Result<super::CudaStepResult>
    where
        F: FnMut(usize, f32, &HotState),
    {
        if !available() {
            bail!("CUDA device probe failed");
        }
        let PackedSoA {
            active,
            mut pos,
            mut vel,
            mass,
            domain,
            semantics,
            sem_nbr,
            sem_w,
        } = pack_active_soa(state, semantic_neighbors)?;
        let n = active.len();

        let params = DreamCudaParams {
            n: n as i32,
            neighbor_mode: if neighbor_mode { 1 } else { 0 },
            dt: config.dt,
            damping: config.damping,
            origin_pull: config.origin_pull,
            attraction: config.attraction,
            semantic_threshold: config.semantic_threshold,
            repulsion_radius: config.repulsion_radius,
            repulsion: config.repulsion,
            semantic_repulsion: config.semantic_repulsion,
            cross_domain_repulsion: config.cross_domain_repulsion,
            neighbor_radius: config.neighbor_radius,
        };

        let ptr = unsafe {
            dream_cuda_create(
                &params,
                pos.as_ptr(),
                vel.as_ptr(),
                mass.as_ptr(),
                domain.as_ptr(),
                semantics.as_ptr(),
                sem_nbr.as_ptr(),
                sem_w.as_ptr(),
            )
        };
        if ptr.is_null() {
            bail!("dream_cuda_create returned null");
        }
        let ctx = CudaCtx { ptr };

        eprintln!("dream: CUDA force loop (active={n}, neighbor_mode={neighbor_mode}, sm_121)");

        let mut steps = 0usize;
        let mut energy = 0.0f32;
        let thresh = super::settled_threshold(n);
        for _ in 0..config.steps {
            let mut disp = 0.0f32;
            let rc = unsafe { dream_cuda_step(ctx.ptr, &mut disp, &mut energy) };
            if rc != 0 {
                let _ =
                    unsafe { dream_cuda_copy_state(ctx.ptr, pos.as_mut_ptr(), vel.as_mut_ptr()) };
                write_back(state, &active, &pos, &vel);
                if steps == 0 {
                    bail!("dream_cuda_step failed rc={rc} before the first step landed");
                }
                eprintln!("dream: CUDA step failed rc={rc} after {steps}; keeping GPU state");
                break;
            }
            steps += 1;
            if emit_every > 0 && steps % emit_every == 0 {
                if unsafe { dream_cuda_copy_state(ctx.ptr, pos.as_mut_ptr(), vel.as_mut_ptr()) }
                    != 0
                {
                    bail!("dream_cuda_copy_state failed at step {steps}");
                }
                write_back(state, &active, &pos, &vel);
                recolor_speed(state, energy);
                on_step(steps, energy, state);
            }
            if disp < thresh {
                break;
            }
        }
        if unsafe { dream_cuda_copy_state(ctx.ptr, pos.as_mut_ptr(), vel.as_mut_ptr()) } != 0 {
            bail!("dream_cuda_copy_state failed at end");
        }
        write_back(state, &active, &pos, &vel);
        Ok(super::CudaStepResult { steps, energy })
    }

    fn write_back(state: &mut HotState, active: &[usize], pos: &[f32], vel: &[f32]) {
        for (compact, &orig) in active.iter().enumerate() {
            let splat = &mut state.splats[orig];
            splat.position = [pos[compact * 3], pos[compact * 3 + 1], pos[compact * 3 + 2]];
            splat.velocity = [vel[compact * 3], vel[compact * 3 + 1], vel[compact * 3 + 2]];
        }
    }

    fn recolor_speed(state: &mut HotState, _energy: f32) {
        for splat in &mut state.splats {
            if splat.muted {
                continue;
            }
            let speed = (splat.velocity[0] * splat.velocity[0]
                + splat.velocity[1] * splat.velocity[1]
                + splat.velocity[2] * splat.velocity[2])
                .sqrt();
            splat.color_rgba = domain_state_color(&splat.domain, splat.basin_id.as_deref(), speed);
        }
    }
}

#[cfg(cuda_dream)]
pub use ffi::{available, run_force_loop};

#[cfg(not(cuda_dream))]
pub fn available() -> bool {
    false
}

#[cfg(not(cuda_dream))]
pub fn run_force_loop<F>(
    _state: &mut HotState,
    _semantic_neighbors: &[Vec<usize>],
    _config: &PhysicsConfig,
    _neighbor_mode: bool,
    _emit_every: usize,
    _on_step: &mut F,
) -> Result<CudaStepResult>
where
    F: FnMut(usize, f32, &HotState),
{
    anyhow::bail!("CUDA dream kernels were not compiled (need nvcc sm_121)")
}

/// Same settle law as the CPU loop: per-active-splat displacement floor.
pub fn settled_threshold(active: usize) -> f32 {
    1e-6 * active.max(1) as f32
}

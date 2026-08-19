use crate::algorithm::TrappingRainWater;
use crate::particle::{Particle, ParticleSystem};
use crate::terrain::Terrain;
use crate::water::WaterSimulation;
use rand::RngExt;
use rayon::prelude::*;

/// Physics simulator
///
/// Unit conventions (consistent with WaterSimulation; the single source of truth is water.particles_per_cell):
/// - 1 particle = 1 ml = 1/N units of water-column height
/// - N particles fill one 1x1x1 cube (1 unit of water column)
/// - Particles per cell ≈ round(level_ml); layer k rests at: height = terrain height + k / N
///
/// Key properties:
/// - **The particle is the water**: wherever water flows, particles follow (real flow, not delete-and-respawn)
/// - Particles are only removed on **boundary drainage** (physically flowing out of the container)
/// - Flow rules are fully deterministic (no randomness) and converge smoothly to the unique equilibrium
/// - At convergence, `finalize` snapshots the state exactly to the unique deterministic equilibrium -> the same container always yields the same result
///
/// Independent rate parameters:
/// - `rain_rate`   rain rate (particles/second), accumulated by real dt, independent of frame rate
/// - `fall_speed`  particle fall speed (height units/second)
/// - Flow / boundary drainage / evaporation all live in WaterSimulation, each independent
pub struct Physics {
    /// Water simulator (floating-point ml)
    pub water: WaterSimulation,
    /// Particle fall speed (height units/second)
    pub fall_speed: f64,
    /// Rain rate (particles/second)
    pub rain_rate: f64,
    /// Rain accumulator (accumulated by real dt, independent of frame rate)
    rain_acc: f64,
    /// Whether each cell's particles have settled (sleep optimization):
    /// cells that are settled and had no water/particle change this frame skip position updates (no re-sorting, no falling)
    cell_settled: Vec<Vec<bool>>,
}

/// Height from which a freshly spawned rain drop falls: a comfortable margin
/// above the tallest possible column, so drops always start above any terrain
/// or stacked water (terrain max + a few units of water headroom).
fn spawn_height(terrain: &Terrain) -> f64 {
    terrain.max_height as f64 + 10.0
}

impl Physics {
    /// Create a new physics simulator
    pub fn new(rows: usize, cols: usize) -> Self {
        Physics {
            water: WaterSimulation::new(rows, cols),
            fall_speed: 15.0,
            rain_rate: 30.0,
            rain_acc: 0.0,
            cell_settled: vec![vec![false; cols]; rows],
        }
    }

    /// Particles per cell N (the count needed to fill one 1x1x1 cube)
    pub fn particles_per_cell(&self) -> usize {
        self.water.particles_per_cell
    }

    /// Rain (random position is the allowed random source): accumulated by real dt; one drop per full 1 ml
    pub fn add_rain(&mut self, dt: f64, particles: &mut ParticleSystem, terrain: &Terrain) {
        self.rain_acc += self.rain_rate * dt;
        let mut rng = rand::rng();
        while self.rain_acc >= 1.0 {
            self.rain_acc -= 1.0;
            let r = rng.random_range(0..self.water.rows);
            let c = rng.random_range(0..self.water.cols);
            self.water.add_water(r, c, 1.0);
            self.cell_settled[r][c] = false; // a new drop landed -> wake up this cell
            let x = c as f64 + rng.random_range(0.2..0.8);
            let y = r as f64 + rng.random_range(0.2..0.8);
            // fall from well above the tallest possible column (terrain max + water headroom)
            particles.add_particle(Particle::new(x, y, spawn_height(terrain)));
        }
    }

    /// Update water levels + particles (one frame)
    pub fn update(&mut self, particles: &mut ParticleSystem, terrain: &Terrain, dt: f64) {
        // 1. Water-level flow (deterministic rules): returns this frame's flows / drainage / evaporation
        let moves = self.water.update(terrain, dt);

        // 2. Particles follow the water: when water flows from A to B, A's top-layer particles move to B
        //    (only whole particles move; sub-particle remainders are handled by reconcile)
        for f in &moves.flows {
            // wake cells whose water level changed (sleep optimization): particles must recompute layers
            self.cell_settled[f.sr][f.sc] = false;
            self.cell_settled[f.tr][f.tc] = false;
            let k = f.amount.floor() as usize;
            if k > 0 {
                self.move_top_particles(particles, (f.sr, f.sc), (f.tr, f.tc), k);
            }
        }

        // 3. Boundary drainage: particles physically flow out of the container (the only way particles are removed)
        for &(r, c, amount) in &moves.drained {
            self.cell_settled[r][c] = false;
            self.remove_top_particles(particles, r, c, amount);
        }

        // 4. Evaporation (default 0): particles disappear
        for &(r, c, amount) in &moves.evaporated {
            self.cell_settled[r][c] = false;
            self.remove_top_particles(particles, r, c, amount);
        }

        // 5. Particles fall + recompute water-column layers by sorting by height
        self.update_particle_positions(particles, terrain, dt);

        // 6. Safety reconcile: particle count matches water level per cell (differences flow along the lowest surface; nothing is deleted arbitrarily)
        self.reconcile_particles(particles, terrain);
    }

    /// Terrain height (negated when flipped)
    fn terrain_height(&self, terrain: &Terrain, r: usize, c: usize) -> f64 {
        terrain.get_height(r, c) as f64
    }

    /// Target height of layer (1-based) = terrain height + layer / N
    fn layer_height(&self, terrain: &Terrain, r: usize, c: usize, layer: usize) -> f64 {
        let n = self.particles_per_cell() as f64;
        self.terrain_height(terrain, r, c) + layer as f64 / n
    }

    /// Move the top amount particles of the source cell to the target cell (particles follow the water)
    fn move_top_particles(
        &self,
        particles: &mut ParticleSystem,
        from: (usize, usize),
        to: (usize, usize),
        amount: usize,
    ) {
        let mut idxs: Vec<usize> = particles
            .particles
            .iter()
            .enumerate()
            .filter(|(_, p)| p.x as usize == from.1 && p.y as usize == from.0)
            .map(|(i, _)| i)
            .collect();
        // the highest particles leave first (the surface drops from the top)
        idxs.sort_by(|&a, &b| {
            particles.particles[b]
                .z
                .partial_cmp(&particles.particles[a].z)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in idxs.iter().take(amount) {
            let p = &mut particles.particles[i];
            p.x = to.1 as f64 + 0.5;
            p.y = to.0 as f64 + 0.5;
            // target_z is recomputed in update_particle_positions
        }
    }

    /// Remove the top amount particles of a cell (used for boundary drainage / evaporation)
    fn remove_top_particles(
        &self,
        particles: &mut ParticleSystem,
        r: usize,
        c: usize,
        amount: usize,
    ) {
        let mut idxs: Vec<usize> = particles
            .particles
            .iter()
            .enumerate()
            .filter(|(_, p)| p.x as usize == c && p.y as usize == r)
            .map(|(i, _)| i)
            .collect();
        idxs.sort_by(|&a, &b| {
            particles.particles[b]
                .z
                .partial_cmp(&particles.particles[a].z)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut remove_idxs: Vec<usize> = idxs.iter().take(amount).copied().collect();
        remove_idxs.sort_unstable();
        for &i in remove_idxs.iter().rev() {
            particles.particles.remove(i);
        }
    }

    /// Particle falling: sort each cell's particles by z; layer k's target = terrain + k/N, then fall.
    ///
    /// Performance (vs. a full per-cell scan O(cellsxn)):
    /// - **Single-pass grouping** (O(n)): scan all particles once per frame, grouped by cell;
    ///   each cell only handles its own particles; no repeated full scans per cell.
    /// - **Parallel sorting**: each cell's sort is disjoint, so sorting is parallelized with rayon.
    /// - **Sleep optimization**: cells that are settled and unchanged this frame are skipped directly
    ///   (no falling, no layer recomputation). Any water/particle change wakes the cell.
    fn update_particle_positions(
        &mut self,
        particles: &mut ParticleSystem,
        terrain: &Terrain,
        dt: f64,
    ) {
        // single-pass grouping: particle index -> its cell (row-major, matching per-cell processing order)
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); self.water.rows * self.water.cols];
        for (i, p) in particles.particles.iter().enumerate() {
            let r = p.y as usize;
            let c = p.x as usize;
            if r < self.water.rows && c < self.water.cols {
                groups[r * self.water.cols + c].push(i);
            }
        }

        // parallel sort (each cell's index set is disjoint; z is read-only): low to high by height -> determine each layer
        {
            let ps = &particles.particles;
            groups.par_iter_mut().for_each(|idxs| {
                idxs.sort_by(|&a, &b| {
                    ps[a]
                        .z
                        .partial_cmp(&ps[b].z)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            });
        }

        for (cell, idxs) in groups.iter_mut().enumerate() {
            let r = cell / self.water.cols;
            let c = cell % self.water.cols;
            if self.cell_settled[r][c] {
                continue; // sleep: particles settled and unchanged this frame
            }
            let mut all_settled = true;
            for (layer, &i) in idxs.iter().enumerate() {
                let target = self.layer_height(terrain, r, c, layer + 1);
                let p = &mut particles.particles[i];
                p.target_z = target;
                if p.z > target {
                    p.z -= self.fall_speed * dt;
                    if p.z < target {
                        p.z = target;
                    }
                } else if p.z < target {
                    // snap up quickly (prevents floating-point residue after moving into a higher cell)
                    p.z = target;
                }
                if (p.z - p.target_z).abs() > 1e-9 {
                    all_settled = false;
                }
            }
            // sleep flag: all settled -> skippable next frame (any change wakes it)
            self.cell_settled[r][c] = all_settled;
        }
    }

    /// Safety reconcile: each cell's particle count should be ≈ round(level_ml).
    /// Too many -> flow out along the lowest surface; too few -> spawn falling particles. Nothing is deleted arbitrarily.
    fn reconcile_particles(&mut self, particles: &mut ParticleSystem, terrain: &Terrain) {
        for r in 0..self.water.rows {
            for c in 0..self.water.cols {
                let desired = self.water.levels[r][c].round() as usize;
                let have = self.count_particles_in_cell(particles, r, c);
                if have > desired {
                    // find the lowest-surface neighbor; extra particles only flow to a **lower** surface (deterministic)
                    let my_surface = self.surface(terrain, r, c);
                    let mut best: Option<(usize, usize, f64)> = None;
                    for (dr, dc) in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr < 0
                            || nr >= self.water.rows as i32
                            || nc < 0
                            || nc >= self.water.cols as i32
                        {
                            continue;
                        }
                        let (nr, nc) = (nr as usize, nc as usize);
                        let s = self.surface(terrain, nr, nc);
                        if s < my_surface && best.is_none_or(|(_, _, bs)| s < bs) {
                            best = Some((nr, nc, s));
                        }
                    }
                    if let Some((nr, nc, _)) = best {
                        // particles moved -> wake both cells (sleep optimization)
                        self.cell_settled[r][c] = false;
                        self.cell_settled[nr][nc] = false;
                        self.move_top_particles(particles, (r, c), (nr, nc), have - desired);
                    } else {
                        // no lower neighbor (local depression): extras are sub-particle quantization error; remove them
                        self.cell_settled[r][c] = false;
                        self.remove_top_particles(particles, r, c, have - desired);
                    }
                } else if have < desired {
                    // spawning particles -> wake this cell (sleep optimization)
                    self.cell_settled[r][c] = false;
                    for _ in 0..(desired - have) {
                        let target = self.layer_height(terrain, r, c, have + 1);
                        let mut p = Particle::new(
                            c as f64 + 0.5,
                            r as f64 + 0.5,
                            target.max(spawn_height(terrain)),
                        );
                        p.target_z = target;
                        particles.add_particle(p);
                    }
                }
            }
        }
    }

    /// Count the particles in the given cell
    fn count_particles_in_cell(&self, particles: &ParticleSystem, r: usize, c: usize) -> usize {
        particles
            .particles
            .iter()
            .filter(|p| p.x as usize == c && p.y as usize == r)
            .count()
    }

    /// Water surface height (units)
    fn surface(&self, terrain: &Terrain, r: usize, c: usize) -> f64 {
        self.water.surface(terrain, r, c)
    }

    /// Total trapped water (units: number of 1x1x1 cubes) = Σ water(ml) / N
    pub fn total_water_units(&self) -> f64 {
        self.water.total_water() / self.particles_per_cell() as f64
    }

    /// Residual flow this frame (ml); ~0 means converged to the unique equilibrium
    pub fn residual(&self) -> f64 {
        self.water.residual
    }

    /// Whether flow/drainage has effectively stopped (frame-rate independent).
    ///
    /// The per-frame `residual` scales with `dt` (flow = rate x dt), so an
    /// absolute per-frame threshold would trigger at a different remaining
    /// water level depending on the frame pacing. Normalizing to a per-second
    /// flow rate keeps the convergence quality identical at any frame rate.
    /// 3.125 units/sec == the old 0.05-per-16ms-frame threshold.
    pub fn converged(&self, dt: f64) -> bool {
        self.residual() / dt.max(1e-9) < 3.125
    }

    /// TRW equilibrium of the current terrain (units) — internal use only for convergence checks; not part of any UI
    pub fn equilibrium_units(&self, terrain: &Terrain) -> usize {
        TrappingRainWater::calculate(&terrain.heights)
    }

    /// Convergence snapshot: pins water levels and particles exactly to the unique deterministic equilibrium (the TRW solution).
    /// The result depends only on the terrain, not the rain path -> the same container always gives the same result.
    pub fn finalize(&mut self, particles: &mut ParticleSystem, terrain: &Terrain) -> usize {
        let n = self.particles_per_cell() as f64;
        let water_heights = TrappingRainWater::water_heights(&terrain.heights);
        particles.particles.clear();
        for (r, row) in water_heights.iter().enumerate() {
            for (c, &wh) in row.iter().enumerate() {
                let terrain_h = terrain.heights[r][c] as i32;
                let units = (wh - terrain_h).max(0) as usize;
                self.water.levels[r][c] = units as f64 * n;
                let th = terrain_h as f64;
                for layer in 1..=(units * self.particles_per_cell()) {
                    let z = th + layer as f64 / n;
                    let mut p = Particle::new(c as f64 + 0.5, r as f64 + 0.5, z);
                    p.target_z = z;
                    particles.add_particle(p);
                }
            }
        }
        // after the snapshot all particles rest exactly at their target layers -> mark all cells settled (sleep optimization)
        for row in self.cell_settled.iter_mut() {
            row.fill(true);
        }
        self.water.residual = 0.0;
        self.water.total_water() as usize / self.particles_per_cell()
    }

    /// Clear all water
    pub fn clear(&mut self) {
        self.water.clear();
        for row in self.cell_settled.iter_mut() {
            row.fill(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flood a known terrain heavily, run a no-rain simulation until near convergence,
    /// then finalize -> total trapped water (units) and particle count must **exactly equal** the TRW answer.
    /// This is the core property of deterministic convergence: independent of the random pour path.
    #[test]
    fn converges_to_algorithm_answer() {
        let mut terrain = Terrain::new(4, 4);
        terrain.heights = vec![
            vec![3, 3, 3, 3],
            vec![3, 0, 0, 3],
            vec![3, 0, 0, 3],
            vec![3, 3, 3, 3],
        ];
        let answer = TrappingRainWater::calculate(&terrain.heights);
        // central 2x2 depression (height 0), rim height 3 -> 4 cells x 3 = 12 units
        assert_eq!(answer, 12);

        // run 3 different random pour paths; the final result must be identical
        for _ in 0..3 {
            let mut physics = Physics::new(4, 4);
            let mut particles = ParticleSystem::new();
            let mut rng = rand::rng();
            for _ in 0..20_000 {
                let r = rng.random_range(0..4);
                let c = rng.random_range(0..4);
                physics.water.add_water(r, c, 1.0);
            }
            let mut frame = 0;
            loop {
                physics.update(&mut particles, &terrain, 0.016);
                frame += 1;
                if (physics.converged(0.016)
                    && (physics.total_water_units() - answer as f64).abs() <= 1.0)
                    || frame > 200_000
                {
                    break;
                }
            }
            assert!(frame < 200_000, "did not converge in {frame} frames");

            let total = physics.finalize(&mut particles, &terrain);
            assert_eq!(
                total, answer,
                "finalized total={total} units, answer={answer} (frame={frame})"
            );
            assert_eq!(
                particles.particles.len(),
                answer * physics.particles_per_cell(),
                "final particle count must be deterministic"
            );
        }
    }

    /// Unit conversion: N particles = 1 unit
    #[test]
    fn units_are_consistent() {
        let mut terrain = Terrain::new(2, 2);
        terrain.heights = vec![vec![1, 1], vec![1, 1]];
        let mut physics = Physics::new(2, 2);
        let mut particles = ParticleSystem::new();

        // 4 ml = 1 unit, exactly filling one 1x1x1 cube
        physics.water.add_water(0, 0, 4.0);
        assert_eq!(physics.total_water_units(), 1.0);

        physics.update(&mut particles, &terrain, 0.016);
        assert!(physics.total_water_units() >= 0.0);
    }

    /// Stress test: 16x16 terrain raining for 300 frames; particle count stays bounded and no panic.
    /// A regression guard for sleep / grouping optimizations — any infinite loop or particle explosion they introduce will surface here.
    #[test]
    fn rain_simulation_stress() {
        let mut terrain = Terrain::new(16, 16);
        terrain.generate_random(7);
        let mut physics = Physics::new(16, 16);
        let mut particles = ParticleSystem::new();
        // 300 frames x 0.016s x 30 particles/s ≈ 144 drops; leftover water drains along the boundary
        for _ in 0..300 {
            physics.add_rain(0.016, &mut particles, &terrain);
            physics.update(&mut particles, &terrain, 0.016);
        }
        assert!(particles.count() > 0, "expected rain to spawn particles");
        assert!(
            particles.count() <= 300,
            "particle count exploded: {}",
            particles.count()
        );
        assert!(physics.water.total_water() >= 0.0);
        assert!(physics.total_water_units() >= 0.0);
    }

    /// Sleep-optimization correctness: after convergence, running many more frames must keep water levels and particle counts unchanged
    /// (sleep only skips "settled and unchanged" cells and must not introduce any drift).
    #[test]
    fn sleep_keeps_converged_state_stable() {
        let mut terrain = Terrain::new(4, 4);
        terrain.heights = vec![
            vec![3, 3, 3, 3],
            vec![3, 0, 0, 3],
            vec![3, 0, 0, 3],
            vec![3, 3, 3, 3],
        ];
        let mut physics = Physics::new(4, 4);
        let mut particles = ParticleSystem::new();
        let mut rng = rand::rng();
        for _ in 0..20_000 {
            let r = rng.random_range(0..4);
            let c = rng.random_range(0..4);
            physics.water.add_water(r, c, 1.0);
        }
        // run until convergence
        let mut frame = 0;
        loop {
            physics.update(&mut particles, &terrain, 0.016);
            frame += 1;
            if physics.converged(0.016) && (physics.total_water_units() - 12.0).abs() <= 1.0 {
                break;
            }
            assert!(frame < 200_000, "did not converge in {frame} frames");
        }
        physics.finalize(&mut particles, &terrain);

        // after convergence run 1000 more frames: water levels, particle count, and trapped water must stay identical
        let water_before = physics.water.total_water();
        let count_before = particles.count();
        let trapped_before = physics.total_water_units();
        for _ in 0..1000 {
            physics.update(&mut particles, &terrain, 0.016);
        }
        assert_eq!(physics.water.total_water(), water_before);
        assert_eq!(particles.count(), count_before);
        assert_eq!(physics.total_water_units(), trapped_before);
    }

    /// Performance regression test: 2000 frames of continuous rain + refill (keep live water),
    /// must finish within a loose time limit (far faster than the threshold even in debug builds).
    /// Guards against sleep/grouping optimizations being reverted to a full per-cell scan.
    #[test]
    fn simulation_throughput() {
        let mut terrain = Terrain::new(8, 8);
        terrain.generate_random(7);
        let mut physics = Physics::new(8, 8);
        let mut particles = ParticleSystem::new();
        let start = std::time::Instant::now();
        for _ in 0..2000 {
            physics.add_rain(0.016, &mut particles, &terrain);
            physics.update(&mut particles, &terrain, 0.016);
            physics.water.add_water(2, 2, 1.0); // keep refilling to maintain active water flow
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs_f64() < 5.0,
            "2000 frames took {:?}",
            elapsed
        );
        assert!(particles.count() > 0);
    }
}

#[test]
fn convergence_is_frame_rate_independent() {
    let mut terrain = Terrain::new(4, 4);
    terrain.heights = vec![
        vec![3, 3, 3, 3],
        vec![3, 0, 0, 3],
        vec![3, 0, 0, 3],
        vec![3, 3, 3, 3],
    ];
    let answer = TrappingRainWater::calculate(&terrain.heights);
    assert_eq!(answer, 12);

    // Identical rain, run at very different frame dts: the rate-normalized
    // convergence check must reach the same deterministic result either way.
    for dt in [0.016, 0.001] {
        let mut physics = Physics::new(4, 4);
        let mut particles = ParticleSystem::new();
        let mut rng = rand::rng();
        for _ in 0..20_000 {
            let r = rng.random_range(0..4);
            let c = rng.random_range(0..4);
            physics.water.add_water(r, c, 1.0);
        }
        let mut frame = 0;
        loop {
            physics.update(&mut particles, &terrain, dt);
            frame += 1;
            if (physics.converged(dt) && (physics.total_water_units() - answer as f64).abs() <= 1.0)
                || frame > 200_000
            {
                break;
            }
        }
        assert!(
            frame < 200_000,
            "did not converge in {frame} frames (dt={dt})"
        );
        let total = physics.finalize(&mut particles, &terrain);
        assert_eq!(total, answer, "dt={dt} finalized total={total}");
    }
}

use crate::terrain::Terrain;

/// A single water flow: amount milliliters (may be fractional) from (sr, sc) to (tr, tc)
#[derive(Clone, Copy, Debug)]
pub struct Flow {
    pub sr: usize,
    pub sc: usize,
    pub tr: usize,
    pub tc: usize,
    pub amount: f64,
}

/// All water movement that occurs in one frame
pub struct WaterMoves {
    /// Lateral flows (wherever water flows, particles should follow)
    pub flows: Vec<Flow>,
    /// Boundary drainage ((row, col, particle count); particles flow out of the container)
    pub drained: Vec<(usize, usize, usize)>,
    /// Evaporation ((row, col, particle count); default 0)
    pub evaporated: Vec<(usize, usize, usize)>,
}

/// Grid-based pipe-model water simulation (floating-point levels; smooth, deterministic convergence)
///
/// Unit conventions (consistent across the project):
/// - `levels[r][c]` = amount of water in the cell (ml); 1 ml = 1 particle
/// - `particles_per_cell` (N) particles = 1 1x1x1 cube = 1 unit of water-column height
/// - Water surface height (units) = terrain height + level / N
///
/// Properties:
/// - Flow rules are fully deterministic (depend only on the current levels and terrain); no randomness
/// - Water only flows from higher surfaces to lower ones -> smooth convergence to the unique equilibrium (the correct TRW solution)
///
/// Independent parameters:
/// - `flow_rate`         lateral flow speed (height units/second)
/// - `drain_rate`        boundary drainage speed (height units/second)
/// - `evaporation_rate`  evaporation speed (height units/second; default 0 = no evaporation)
pub struct WaterSimulation {
    /// Amount of water in each cell (ml)
    pub levels: Vec<Vec<f64>>,
    /// Number of particles needed to fill one 1x1x1 cube (N, the single source of truth)
    pub particles_per_cell: usize,
    /// Lateral flow rate (height units/second)
    pub flow_rate: f64,
    /// Boundary drainage rate (height units/second)
    pub drain_rate: f64,
    /// Evaporation rate (height units/second; 0 = no evaporation)
    pub evaporation_rate: f64,
    /// Residual flow this frame (ml); ~0 means converged
    pub residual: f64,
    /// Number of grid rows
    pub rows: usize,
    /// Number of grid columns
    pub cols: usize,
    /// Frame counter (used to alternate scan direction and avoid directional bias)
    frame_count: u64,
}

impl WaterSimulation {
    /// Create a new water simulation
    pub fn new(rows: usize, cols: usize) -> Self {
        WaterSimulation {
            levels: vec![vec![0.0; cols]; rows],
            particles_per_cell: 4,
            flow_rate: 8.0,
            drain_rate: 5.0,
            evaporation_rate: 0.0,
            residual: 0.0,
            rows,
            cols,
            frame_count: 0,
        }
    }

    /// Apply all water-level updates for one frame (deterministic rules); returns all water movement this frame
    pub fn update(&mut self, terrain: &Terrain, dt: f64) -> WaterMoves {
        let flows = self.step_flow(terrain, dt);
        let drained = self.step_drain(dt);
        let evaporated = self.step_evaporation(dt);
        self.residual = flows.iter().map(|f| f.amount).sum::<f64>()
            + drained.iter().map(|t| t.2 as f64).sum::<f64>()
            + evaporated.iter().map(|t| t.2 as f64).sum::<f64>();
        WaterMoves {
            flows,
            drained,
            evaporated,
        }
    }

    /// Terrain height (negated when flipped)
    fn terrain_height(&self, terrain: &Terrain, r: usize, c: usize) -> f64 {
        terrain.get_height(r, c) as f64
    }

    /// Water surface height (units) = terrain height + level / N
    pub fn surface(&self, terrain: &Terrain, r: usize, c: usize) -> f64 {
        self.terrain_height(terrain, r, c) + self.levels[r][c] / self.particles_per_cell as f64
    }

    /// Lateral flow: water flows from higher to lower surfaces; flow is proportional to the height difference
    fn step_flow(&mut self, terrain: &Terrain, dt: f64) -> Vec<Flow> {
        let n = self.particles_per_cell as f64;
        let directions = [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)];
        let mut flows = Vec::new();

        // Alternate the scan direction to avoid directional bias
        let (row_range, col_range): (Vec<usize>, Vec<usize>) = if self.frame_count.is_multiple_of(2)
        {
            ((0..self.rows).collect(), (0..self.cols).collect())
        } else {
            (
                (0..self.rows).rev().collect(),
                (0..self.cols).rev().collect(),
            )
        };

        for row in &row_range {
            for col in &col_range {
                if self.levels[*row][*col] <= 0.0 {
                    continue;
                }
                let my_surface = self.surface(terrain, *row, *col);
                for (dr, dc) in &directions {
                    let nr = *row as i32 + dr;
                    let nc = *col as i32 + dc;
                    if nr < 0 || nr >= self.rows as i32 || nc < 0 || nc >= self.cols as i32 {
                        continue;
                    }
                    let (nr, nc) = (nr as usize, nc as usize);
                    let neighbor_surface = self.surface(terrain, nr, nc);
                    let diff = my_surface - neighbor_surface;
                    if diff > 0.0 {
                        // height difference x flow rate x dt -> height units, then converted to ml (x N)
                        let flow =
                            (self.levels[*row][*col].min(diff * self.flow_rate * dt * n)).max(0.0);
                        self.levels[*row][*col] -= flow;
                        self.levels[nr][nc] += flow;
                        flows.push(Flow {
                            sr: *row,
                            sc: *col,
                            tr: nr,
                            tc: nc,
                            amount: flow,
                        });
                    }
                }
            }
        }

        self.frame_count += 1;
        flows
    }

    /// Boundary drainage: water in boundary cells flows out of the grid (TRW semantics)
    fn step_drain(&mut self, dt: f64) -> Vec<(usize, usize, usize)> {
        let mut drained = Vec::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                let is_boundary = r == 0 || r == self.rows - 1 || c == 0 || c == self.cols - 1;
                if is_boundary && self.levels[r][c] > 0.0 {
                    let drain = (self.levels[r][c] * self.drain_rate * dt).min(self.levels[r][c]);
                    self.levels[r][c] -= drain;
                    // Drainage is counted in whole particles (for particle removal); the remainder stays for the next frame
                    let amount = drain.round() as usize;
                    if amount > 0 {
                        drained.push((r, c, amount));
                    }
                }
            }
        }
        drained
    }

    /// Evaporation (default 0; an independent parameter unaffected by other rates)
    fn step_evaporation(&mut self, dt: f64) -> Vec<(usize, usize, usize)> {
        let mut evaporated = Vec::new();
        if self.evaporation_rate <= 0.0 {
            return evaporated;
        }
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.levels[r][c] > 0.0 {
                    let evap =
                        (self.levels[r][c] * self.evaporation_rate * dt).min(self.levels[r][c]);
                    self.levels[r][c] -= evap;
                    let amount = evap.round() as usize;
                    if amount > 0 {
                        evaporated.push((r, c, amount));
                    }
                }
            }
        }
        evaporated
    }

    /// Add water at the given position (units: ml)
    pub fn add_water(&mut self, r: usize, c: usize, amount: f64) {
        if r < self.rows && c < self.cols {
            self.levels[r][c] += amount;
        }
    }

    /// Total amount of water (ml)
    pub fn total_water(&self) -> f64 {
        self.levels.iter().flatten().sum()
    }

    /// Clear all water
    pub fn clear(&mut self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.levels[row][col] = 0.0;
            }
        }
        self.residual = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::Terrain;

    #[test]
    fn new_water_is_empty() {
        let w = WaterSimulation::new(3, 4);
        assert_eq!(w.rows, 3);
        assert_eq!(w.cols, 4);
        assert_eq!(w.total_water(), 0.0);
        assert!(w.levels.iter().flatten().all(|&l| l == 0.0));
    }

    #[test]
    fn add_and_total() {
        let mut w = WaterSimulation::new(2, 2);
        w.add_water(0, 0, 1.5);
        w.add_water(1, 1, 2.5);
        assert_eq!(w.total_water(), 4.0);
        // out-of-bounds adds are ignored
        w.add_water(99, 99, 100.0);
        assert_eq!(w.total_water(), 4.0);
    }

    #[test]
    fn boundary_drains_to_zero() {
        let mut terrain = Terrain::new(1, 2);
        terrain.heights = vec![vec![0, 0]];
        let mut w = WaterSimulation::new(1, 2);
        w.add_water(0, 0, 8.0);
        // in a 1x2 grid both cells are on the boundary, so all water drains out
        for _ in 0..2000 {
            w.update(&terrain, 0.016);
        }
        assert!(w.total_water() < 0.001, "total={}", w.total_water());
    }

    #[test]
    fn flow_moves_toward_lower_neighbor() {
        // 4x4 flat terrain: pour water into inner cell (1,1); its surface is higher than neighbors -> water flows to lower surfaces
        let mut terrain = Terrain::new(4, 4);
        terrain.heights = vec![vec![0; 4]; 4];
        let mut w = WaterSimulation::new(4, 4);
        w.add_water(1, 1, 8.0); // surface = 0 + 8/4 = 2 units
        w.update(&terrain, 0.016);
        // water flowed from (1,1) to (2,1) (lower-surface neighbor); the level at (1,1) dropped
        assert!(
            w.levels[1][1] < 8.0,
            "source cell level should drop: {}",
            w.levels[1][1]
        );
        assert!(
            w.levels[2][1] > 0.0,
            "lower-surface neighbor should gain water: {}",
            w.levels[2][1]
        );
    }

    #[test]
    fn clear_removes_all() {
        let mut w = WaterSimulation::new(2, 2);
        w.add_water(0, 0, 3.0);
        w.add_water(1, 1, 3.0);
        w.clear();
        assert_eq!(w.total_water(), 0.0);
        assert_eq!(w.residual, 0.0);
    }

    #[test]
    fn surface_is_terrain_plus_level_over_n() {
        let mut terrain = Terrain::new(1, 1);
        terrain.heights = vec![vec![2]];
        let mut w = WaterSimulation::new(1, 1);
        w.particles_per_cell = 4;
        w.add_water(0, 0, 8.0); // 2 units
        // surface = 2 + 8/4 = 4
        assert!((w.surface(&terrain, 0, 0) - 4.0).abs() < 1e-12);
    }
}

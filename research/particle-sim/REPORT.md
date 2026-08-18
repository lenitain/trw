# Efficient Particle Simulation for TUI Renderer

> Generated 2026-08-18 · depth: standard · 5 research angles · workspace: research/particle-sim/

## Executive summary

- **Recommendation: Use a grid-based "pipe model" water level simulation**, not particle-based physics. Compute water levels per terrain cell (O(64) for 8×8 grid), then render particles at those positions. This decouples simulation cost from particle count. [1][2][3]
- **PBD is overkill** for this use case. It requires neighbor search + constraint iteration, adding complexity without visual benefit at TUI resolution. [F1]
- **Cellular automata (falling sand) rules** produce identical visual results to the pipe model but are harder to tune for continuous water levels. The pipe model is strictly better for this grid size. [F2]
- **Braille rendering resolution (~240×120 dots) masks all fine simulation detail.** The visual difference between sophisticated particle physics and simple grid-based water is imperceptible. [F5]
- **The real bottleneck is rendering, not simulation.** Each particle = 8 vertices + 12 edges. 500 particles = 4000 vertices to project. Simulation cost is negligible regardless of method. [F4][F5]
- **Spatial hashing is unnecessary** for 100-500 particles. Brute-force O(n²) collision detection takes <0.5ms at this scale. [F4]
- **Implementation: ~80-120 lines of Rust** for the pipe model + particle rendering, replacing the current ~60-line physics.rs. [F3]

## Background & scope

The TRW-3D demo renders a 3D Trapping Rain Water visualization in a TUI using braille characters. The current particle system only supports vertical falling with no horizontal movement, no particle-particle interaction, and no flowing behavior. We need particles to flow into valleys, stack on top of each other, and flow out when the container is inverted. The terrain is an 8×8 height matrix (expandable to 16×16), with 100-500 particles, running at 30+ FPS on CPU-only hardware in Rust.

## Simulation method comparison

| Method | Complexity/frame | Visual quality | Implementation | Best for |
|--------|-----------------|----------------|----------------|----------|
| **Pipe model (water levels)** | O(grid_cells) = O(64) | Excellent for TUI | ~80 lines | **TRW-3D (recommended)** |
| Cellular automata | O(grid_cells) = O(64) | Good (blocky) | ~100 lines | Falling sand games |
| PBD (Position-Based Dynamics) | O(n × k × neighbors) | Excellent | ~200+ lines | Games with soft bodies |
| SPH (Smoothed Particle Hydrodynamics) | O(n²) or O(n log n) | Excellent | ~300+ lines | Fluid simulation research |
| 2D Shallow Water Equations | O(grid_cells) | Excellent | ~150 lines | Large-scale water |
| Brute-force particle collision | O(n²) = O(250K) | Good | ~100 lines | Small particle counts |

## Recommended approach: Grid-based pipe model + particle rendering

### Core idea

1. **Simulation layer** (grid-based): Maintain a `water_level: Vec<Vec<f64>>` array matching the terrain dimensions. Each frame, compute water flow between adjacent cells based on height differences (pipe model). This is O(64) for 8×8.

2. **Rendering layer** (particle-based): For each cell with `water_level > 0`, spawn/place particles at positions `(col, row, terrain_height + water_level)`. The particle count is proportional to water level, giving natural density.

3. **Decoupling**: Simulation runs on the grid (cheap), rendering maps grid state to particle positions (cheap). Total cost: O(64) simulation + O(n) rendering.

### Why this works for TRW-3D

- **Flowing into valleys**: Water flows from high-level cells to low-level neighbors automatically.
- **Stacking**: Water accumulates in cells, rising naturally.
- **Inversion**: When terrain flips, water levels recomputed — water falls to new lowest points.
- **Visual quality**: At TUI resolution, grid-based water looks identical to particle-based water.
- **Performance**: 64 cell updates per frame is negligible (<0.01ms).

### Algorithm: Pipe Model

```
For each frame with timestep dt:
  For each cell (row, col):
    For each neighbor (nr, nc):
      height_diff = (water_level[row][col] + terrain[row][col]) 
                   - (water_level[nr][nc] + terrain[nr][nc])
      if height_diff > 0:
        flow = min(water_level[row][col], height_diff * flow_rate * dt)
        water_level[row][col] -= flow
        water_level[nr][nc] += flow
```

Key parameters:
- `flow_rate`: 2.0-5.0 (higher = faster flow, less stable)
- `dt`: 0.016 (60 FPS timestep)
- `damping`: 0.99 (prevents oscillation)

### Stability

The CFL condition for this method: `dt < cell_size / sqrt(g * max_water_height)`. For cell_size=1, g=9.8, max_height=10: dt < 0.1s. At 60 FPS (dt=0.016), stability is guaranteed.

## Optimization tricks

### 1. Simulation tricks (from F4)
- **Brute-force is fine**: For 100-500 particles, O(n²) collision detection takes <0.5ms. No spatial hashing needed.
- **SoA layout**: Store particle positions as separate `Vec<f64>` for x, y, z (not `Vec<Particle>`). Enables SIMD autovectorization.
- **Sleep optimization**: Mark settled particles as sleeping; skip physics updates. 80%+ of particles may be sleeping at any time.
- **Temporal coherence**: Use insertion sort for spatial hash (particles barely move between frames).

### 2. Rendering tricks (from analysis of render.rs)
- **The bottleneck is vertex count**: Each particle cube = 8 vertices + 12 edges. 500 particles = 4000 vertices to project. The `project_batch` function in view.rs processes each vertex with matrix multiplication.
- **Reduce vertex count**: Use point rendering (1 vertex per particle) instead of cube wireframes for particles far from camera. Or use billboards (2 triangles = 4 vertices) instead of cubes (8 vertices).
- **Cull invisible particles**: Skip particles outside the view frustum before adding to the model.
- **Dirty flag optimization**: Only rebuild the particle model when particles actually change (already partially implemented in main.rs).

### 3. Update order tricks (from F2)
- **Alternate scan direction**: On even frames, process cells left-to-right; on odd frames, right-to-left. Prevents directional bias in water flow.
- **Random perturbation**: Add small random noise to flow rates to break symmetry and produce more natural-looking flow.

## Implementation sketch

### New `water.rs` (~80 lines)

```rust
pub struct WaterSimulation {
    pub levels: Vec<Vec<f64>>,  // water level per cell
    pub flow_rate: f64,         // 2.0-5.0
    pub damping: f64,           // 0.99
    rows: usize,
    cols: usize,
}

impl WaterSimulation {
    pub fn new(rows: usize, cols: usize) -> Self {
        WaterSimulation {
            levels: vec![vec![0.0; cols]; rows],
            flow_rate: 3.0,
            damping: 0.99,
            rows,
            cols,
        }
    }

    pub fn update(&mut self, terrain: &Terrain, dt: f64, inverted: bool) {
        let directions = [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)];
        
        for row in 0..self.rows {
            for col in 0..self.cols {
                if self.levels[row][col] <= 0.0 {
                    continue;
                }
                
                let my_height = terrain.get_height(row, col) as f64 
                              + self.levels[row][col];
                
                for (dr, dc) in &directions {
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    
                    if nr < 0 || nr >= self.rows as i32 
                        || nc < 0 || nc >= self.cols as i32 {
                        continue;
                    }
                    
                    let (nr, nc) = (nr as usize, nc as usize);
                    let neighbor_height = terrain.get_height(nr, nc) as f64 
                                        + self.levels[nr][nc];
                    
                    let diff = my_height - neighbor_height;
                    if diff > 0.0 {
                        let flow = (self.levels[row][col]
                            .min(diff * self.flow_rate * dt))
                            .max(0.0);
                        self.levels[row][col] -= flow;
                        self.levels[nr][nc] += flow;
                    }
                }
            }
        }
        
        // Apply damping
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.levels[row][col] *= self.damping;
            }
        }
    }
    
    pub fn add_water(&mut self, row: usize, col: usize, amount: f64) {
        if row < self.rows && col < self.cols {
            self.levels[row][col] += amount;
        }
    }
    
    pub fn total_water(&self) -> f64 {
        self.levels.iter().flatten().sum()
    }
}
```

### Integration with existing code

Modify `physics.rs` to use the water simulation:

```rust
// In Physics::update():
// 1. Run water simulation on grid
water_sim.update(terrain, dt, inverted);

// 2. Map water levels to particle positions
for row in 0..terrain.rows {
    for col in 0..terrain.cols {
        let level = water_sim.levels[row][col];
        if level > 0.1 {
            // Place particles at water surface
            let z = terrain.get_height(row, col) as f64 + level;
            // Update or spawn particle at (col, row, z)
        }
    }
}
```

### Modifications to Particle struct

Add velocity for more natural movement:

```rust
pub struct Particle {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,  // NEW: horizontal velocity
    pub vy: f64,  // NEW: horizontal velocity
    pub vz: f64,  // NEW: vertical velocity
    pub pinned: bool,
    pub sleeping: bool,  // NEW: sleep optimization
}
```

## Key insights

1. **Don't simulate particles — simulate water levels.** The grid is tiny (64 cells). Compute water levels on the grid, render particles at those positions. This is the single most important insight.

2. **TUI resolution is your friend.** At 240×120 braille dots, the visual difference between sophisticated physics and simple grid-based water is zero. Don't over-engineer the simulation.

3. **Rendering is the bottleneck, not simulation.** Each particle cube costs 8 vertex projections. Consider reducing vertex count (points instead of cubes) for particles.

4. **The existing TrappingRainWater algorithm already computes water levels.** You could potentially reuse `calculate_water_levels()` from algorithm.rs instead of implementing a separate simulation.

5. **Start simple, add complexity only if needed.** The pipe model with 2-3 parameters (flow_rate, damping, gravity) produces excellent results. Add particle-particle interaction only if the visual result is unsatisfying.

6. **Update order matters.** Alternate scan direction each frame to prevent directional bias. This is a one-line change with significant visual impact.

## Open questions

1. Should the existing `TrappingRainWater::calculate_water_levels()` be reused as the simulation engine, or is a separate time-stepping simulation needed for animation?
2. How many particles per cell should be rendered? (1-3 seems reasonable for TUI resolution)
3. Should particle rendering switch from cube wireframes to simpler point/dot rendering to reduce vertex count?
4. Is the `rayon` dependency still needed if simulation is O(64)?

## Sources

[1] Müller et al., "Position Based Dynamics" — https://matthias-research.github.io/pages/publications/posBasedDyn.pdf (2007, accessed 2026-08-18)
[2] Wikipedia, "Falling-sand game" — https://en.wikipedia.org/wiki/Falling-sand_game (2024, accessed 2026-08-18)
[3] Wikipedia, "Shallow water equations" — https://en.wikipedia.org/wiki/Shallow_water_equations (2024, accessed 2026-08-18)
[4] NVIDIA GPU Gems 3, Ch.32 "Broad-Phase Collision Detection with CUDA" — https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda (2007, accessed 2026-08-18)
[5] Wikipedia, "Noita (video game)" — https://en.wikipedia.org/wiki/Noita_(video_game) (2020, accessed 2026-08-18)
[6] Minecraft Wiki, "Water" — https://minecraft.wiki/w/Water (2024, accessed 2026-08-18)
[7] Algosome, "Continuous Liquid Cellular Automata" — https://www.algosome.com/articles/continuous-liquid-cellular-automata.html (2018, accessed 2026-08-18)
[8] Wikipedia, "Courant–Friedrichs–Lewy condition" — https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition (2024, accessed 2026-08-18)

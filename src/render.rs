use crate::particle::ParticleSystem;
use crate::terrain::Terrain;
use crate::water::WaterSimulation;
use ratatui_wireframe::model::Model;

type ModelGeom = (Vec<(f64, f64, f64)>, Vec<(usize, usize)>);

/// Build a box model for one cell at (col, row), from y_base to y_base+height.
/// Uses the box8 primitive: 8 vertices, 12 edges.
fn box8_at(col: usize, row: usize, y_base: f64, height: f64) -> ModelGeom {
    let x = col as f64;
    let z = row as f64;
    let y0 = y_base;
    let y1 = y_base + height;

    // Bottom face (y=y0), Top face (y=y1)
    // Convention: 0-3 bottom, 4-7 top
    let vertices = vec![
        (x, y0, z),             // 0: bottom-left-front
        (x + 1.0, y0, z),       // 1: bottom-right-front
        (x + 1.0, y0, z + 1.0), // 2: bottom-right-back
        (x, y0, z + 1.0),       // 3: bottom-left-back
        (x, y1, z),             // 4: top-left-front
        (x + 1.0, y1, z),       // 5: top-right-front
        (x + 1.0, y1, z + 1.0), // 6: top-right-back
        (x, y1, z + 1.0),       // 7: top-left-back
    ];

    let edges = vec![
        // Bottom face
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        // Top face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        // Vertical edges
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    (vertices, edges)
}

/// Box from the ground up (y=0), used for terrain cells.
fn box8(col: usize, row: usize, height: f64) -> ModelGeom {
    box8_at(col, row, 0.0, height)
}

/// Build a small cube centered at a point (for falling rain drops).
fn particle_cube(cx: f64, cy: f64, cz: f64, size: f64) -> ModelGeom {
    let s = size / 2.0;
    let vertices = vec![
        (cx - s, cy - s, cz - s), // 0
        (cx + s, cy - s, cz - s), // 1
        (cx + s, cy - s, cz + s), // 2
        (cx - s, cy - s, cz + s), // 3
        (cx - s, cy + s, cz - s), // 4
        (cx + s, cy + s, cz - s), // 5
        (cx + s, cy + s, cz + s), // 6
        (cx - s, cy + s, cz + s), // 7
    ];

    let edges = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    (vertices, edges)
}

/// Build a Model from the terrain height matrix.
/// Each cell becomes a box with height proportional to the terrain height.
pub fn build_terrain_model(terrain: &Terrain) -> Model {
    let mut all_verts = Vec::new();
    let mut all_edges = Vec::new();

    for row in 0..terrain.rows {
        for col in 0..terrain.cols {
            let h = terrain.heights[row][col] as f64;
            if h <= 0.0 {
                continue; // skip zero-height cells
            }
            let offset = all_verts.len();
            let (verts, edges) = box8(col, row, h);
            all_verts.extend(verts);
            for (a, b) in edges {
                all_edges.push((a + offset, b + offset));
            }
        }
    }

    Model {
        vertices: all_verts,
        edges: all_edges,
    }
}

/// Build the water model.
///
/// Settled water: each cell renders as a **single continuous column** from the terrain top to the
/// water surface (1x1 footprint, height = water level / N), instead of stacked slices — it looks
/// like one solid water column. Falling rain drops are still rendered as small cubes (z > target
/// height = still falling).
pub fn build_water_model(
    water: &WaterSimulation,
    terrain: &Terrain,
    particles: &ParticleSystem,
) -> Model {
    let n = water.particles_per_cell as f64;
    let mut all_verts = Vec::new();
    let mut all_edges = Vec::new();

    let mut push = |geom: ModelGeom| {
        let offset = all_verts.len();
        all_verts.extend(geom.0);
        for (a, b) in geom.1 {
            all_edges.push((a + offset, b + offset));
        }
    };

    // 1. Settled water: one continuous column per cell
    for row in 0..water.rows {
        for col in 0..water.cols {
            let level = water.levels[row][col];
            if level <= 0.0 {
                continue;
            }
            let y_base = terrain.get_height(row, col) as f64;
            let water_height = level / n;
            if water_height > 0.0 {
                push(box8_at(col, row, y_base, water_height));
            }
        }
    }

    // 2. Falling rain drops (still dropping, not yet merged into a column)
    for p in &particles.particles {
        if p.z > p.target_z + 0.01 {
            // particle.x = col, particle.y = row, particle.z = height
            // In 3D: X = col, Y = height, Z = row
            push(particle_cube(p.x, p.z, p.y, 0.3));
        }
    }

    Model {
        vertices: all_verts,
        edges: all_edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::{Particle, ParticleSystem};
    use crate::water::WaterSimulation;

    #[test]
    fn terrain_model_box_per_cell() {
        let mut terrain = Terrain::new(2, 2);
        terrain.heights = vec![vec![1, 0], vec![2, 3]];
        let m = build_terrain_model(&terrain);
        // 3 non-zero cells x (8 vertices + 12 edges)
        assert_eq!(m.vertices.len(), 3 * 8);
        assert_eq!(m.edges.len(), 3 * 12);
    }

    #[test]
    fn terrain_model_geometry() {
        let mut terrain = Terrain::new(1, 1);
        terrain.heights = vec![vec![2]];
        let m = build_terrain_model(&terrain);
        assert_eq!(m.vertices.len(), 8);
        // box8 layout: index 0 is the bottom-left-front of the base (0,0,0), index 4 is the top face (0,2,0)
        assert_eq!(m.vertices[0], (0.0, 0.0, 0.0));
        assert_eq!(m.vertices[4], (0.0, 2.0, 0.0));
        assert_eq!(m.vertices[7], (0.0, 2.0, 1.0));
    }

    #[test]
    fn zero_height_cells_are_skipped() {
        let terrain = Terrain::new(3, 3); // all-zero heights
        let m = build_terrain_model(&terrain);
        assert_eq!(m.vertices.len(), 0);
        assert_eq!(m.edges.len(), 0);
    }

    #[test]
    fn water_model_columns_and_drops() {
        let mut terrain = Terrain::new(1, 1);
        terrain.heights = vec![vec![1]];
        let mut water = WaterSimulation::new(1, 1);
        water.particles_per_cell = 4;
        water.levels = vec![vec![8.0]]; // 2 units of water -> one column
        let mut ps = ParticleSystem::new();
        // 2 falling drops (z > target_z)
        ps.add_particle(Particle::new(0.5, 0.5, 10.0));
        ps.add_particle(Particle::new(0.5, 0.5, 9.0));
        // 1 settled particle (z == target_z, should not be drawn as a drop)
        let mut settled = Particle::new(0.5, 0.5, 1.0);
        settled.target_z = 1.0;
        ps.add_particle(settled);

        let m = build_water_model(&water, &terrain, &ps);
        // 1 water column (8 vertices) + 2 drop cubes (16 vertices)
        assert_eq!(m.vertices.len(), 3 * 8);
        assert_eq!(m.edges.len(), 3 * 12);
    }

    #[test]
    fn water_model_empty_when_no_water() {
        let mut terrain = Terrain::new(2, 2);
        terrain.heights = vec![vec![1, 1], vec![1, 1]];
        let water = WaterSimulation::new(2, 2);
        let ps = ParticleSystem::new();
        let m = build_water_model(&water, &terrain, &ps);
        assert_eq!(m.vertices.len(), 0);
        assert_eq!(m.edges.len(), 0);
    }

    /// Render-scale smoke test: 1000 drops -> 8000 vertices / 12000 edges, all in the model
    #[test]
    fn large_water_model_smoke() {
        let terrain = Terrain::new(16, 16);
        let water = WaterSimulation::new(16, 16);
        let mut ps = ParticleSystem::new();
        for i in 0..1000 {
            ps.add_particle(Particle::new(
                (i % 16) as f64 + 0.5,
                (i / 16) as f64 + 0.5,
                20.0,
            ));
        }
        let m = build_water_model(&water, &terrain, &ps);
        assert_eq!(m.vertices.len(), 1000 * 8);
        assert_eq!(m.edges.len(), 1000 * 12);
    }

    /// Performance regression test: a 2000-drop model + 30 parallel projections
    /// must finish within a loose time limit (a regression guard for the render vertex bottleneck).
    #[test]
    fn render_throughput() {
        use crate::view;
        let terrain = Terrain::new(8, 8);
        let water = WaterSimulation::new(8, 8);
        let mut ps = ParticleSystem::new();
        for i in 0..2000 {
            ps.add_particle(Particle::new(
                (i % 8) as f64 + 0.5,
                (i / 8) as f64 + 0.5,
                20.0,
            ));
        }
        let start = std::time::Instant::now();
        let m = build_water_model(&water, &terrain, &ps);
        let v = view::ViewState::default();
        let n = m.vertices.len();
        let mut out = vec![[0.0; 2]; n];
        let mut ok = vec![false; n];
        let mut depths = vec![0.0; n];
        for _ in 0..30 {
            view::project_batch(&m.vertices, &v, 120, &mut out, &mut ok, &mut depths);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs_f64() < 5.0,
            "render (model+30 projections) took {:?}",
            elapsed
        );
    }
}

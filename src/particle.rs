/// A rain particle (= 1 ml of water)
///
/// The particle is the water itself: it falls, flows with the water (from high to low), stacks in hollows,
/// and spills out of the container at the boundary. The horizontal coordinates (x, y) determine which
/// cell the particle belongs to; z is its current height, and target_z is its target layer height in that cell's water column.
#[derive(Clone)]
pub struct Particle {
    /// Matrix coordinate (column)
    pub x: f64,
    /// Matrix coordinate (row)
    pub y: f64,
    /// Current height (while falling)
    pub z: f64,
    /// Target height (resting position of the water-column layer this particle belongs to)
    pub target_z: f64,
}

impl Particle {
    /// Create a new particle (starts falling from height z)
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Particle {
            x,
            y,
            z,
            target_z: 0.0,
        }
    }

    /// Whether the particle is out of bounds
    pub fn is_out_of_bounds(&self, max_x: f64, max_y: f64, min_z: f64) -> bool {
        self.x < -1.0
            || self.x > max_x + 1.0
            || self.y < -1.0
            || self.y > max_y + 1.0
            || self.z < min_z
    }
}

/// Particle system
pub struct ParticleSystem {
    pub particles: Vec<Particle>,
}

impl ParticleSystem {
    /// Create a new particle system
    pub fn new() -> Self {
        ParticleSystem {
            particles: Vec::new(),
        }
    }

    /// Add a particle
    pub fn add_particle(&mut self, particle: Particle) {
        self.particles.push(particle);
    }

    /// Clear all particles
    pub fn clear(&mut self) {
        self.particles.clear();
    }

    /// Remove particles that are out of bounds
    pub fn remove_out_of_bounds(&mut self, max_x: f64, max_y: f64, min_z: f64) {
        self.particles
            .retain(|p| !p.is_out_of_bounds(max_x, max_y, min_z));
    }

    /// Get the number of particles
    pub fn count(&self) -> usize {
        self.particles.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_particle_fields() {
        let p = Particle::new(1.5, 2.5, 10.0);
        assert_eq!(p.x, 1.5);
        assert_eq!(p.y, 2.5);
        assert_eq!(p.z, 10.0);
        assert_eq!(p.target_z, 0.0); // initial target layer is 0; set by the physics module
    }

    #[test]
    fn add_clear_count() {
        let mut ps = ParticleSystem::new();
        assert_eq!(ps.count(), 0);
        ps.add_particle(Particle::new(0.0, 0.0, 1.0));
        ps.add_particle(Particle::new(1.0, 1.0, 2.0));
        assert_eq!(ps.count(), 2);
        ps.clear();
        assert_eq!(ps.count(), 0);
    }

    #[test]
    fn remove_out_of_bounds_only() {
        let mut ps = ParticleSystem::new();
        ps.add_particle(Particle::new(0.5, 0.5, 1.0)); // in bounds
        ps.add_particle(Particle::new(50.0, 0.5, 1.0)); // x out of bounds
        ps.add_particle(Particle::new(0.5, 0.5, -10.0)); // z below the lower bound
        ps.remove_out_of_bounds(8.0, 8.0, -5.0);
        assert_eq!(ps.count(), 1);
        assert_eq!(ps.particles[0].x, 0.5);
    }

    #[test]
    fn is_out_of_bounds_edges() {
        let p = Particle::new(-2.0, 0.5, 1.0);
        assert!(p.is_out_of_bounds(8.0, 8.0, -5.0));
        let p = Particle::new(0.5, 0.5, -6.0);
        assert!(p.is_out_of_bounds(8.0, 8.0, -5.0));
        let p = Particle::new(9.5, 0.5, 1.0);
        assert!(p.is_out_of_bounds(8.0, 8.0, -5.0));
        let p = Particle::new(0.5, 0.5, 1.0);
        assert!(!p.is_out_of_bounds(8.0, 8.0, -5.0));
    }
}

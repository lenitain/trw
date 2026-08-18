use rand::RngExt;

/// Height matrix (the container)
pub struct Terrain {
    pub heights: Vec<Vec<u8>>,
    pub rows: usize,
    pub cols: usize,
    /// Max height used by the last random generation (0 until the first one).
    /// Lets physics/render know how tall the tallest possible column can be.
    pub max_height: u8,
}

impl Terrain {
    /// Create a new terrain
    pub fn new(rows: usize, cols: usize) -> Self {
        let heights = vec![vec![0; cols]; rows];
        Terrain {
            heights,
            rows,
            cols,
            max_height: 0,
        }
    }

    /// Generate terrain with random heights in [0, max_height]
    pub fn generate_random(&mut self, max_height: u8) {
        self.max_height = max_height;
        let mut rng = rand::rng();
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.heights[i][j] = rng.random_range(0..=max_height);
            }
        }
    }

    /// Get the height at the given position
    pub fn get_height(&self, row: usize, col: usize) -> u8 {
        if row < self.rows && col < self.cols {
            self.heights[row][col]
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_zero_grid() {
        let t = Terrain::new(4, 6);
        assert_eq!(t.rows, 4);
        assert_eq!(t.cols, 6);
        assert_eq!(t.max_height, 0);
        assert!(t.heights.iter().flatten().all(|&h| h == 0));
    }

    #[test]
    fn random_values_within_bounds() {
        let mut t = Terrain::new(16, 16);
        t.generate_random(7);
        assert_eq!(t.max_height, 7);
        assert!(t.heights.iter().flatten().all(|&h| h <= 7));
        // Run a few times to guarantee randomness (all-zero is nearly impossible)
        let mut saw_nonzero = false;
        for _ in 0..5 {
            t.generate_random(7);
            saw_nonzero |= t.heights.iter().flatten().any(|&h| h > 0);
        }
        assert!(saw_nonzero);
    }

    #[test]
    fn get_height_with_bounds_checks() {
        let mut t = Terrain::new(3, 3);
        t.heights[1][2] = 5;
        assert_eq!(t.get_height(1, 2), 5);
        // Out-of-bounds returns 0
        assert_eq!(t.get_height(99, 99), 0);
        assert_eq!(t.get_height(0, 3), 0);
        assert_eq!(t.get_height(3, 0), 0);
    }
}

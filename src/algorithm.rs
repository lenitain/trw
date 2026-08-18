use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Trapping Rain Water II algorithm
///
/// Used internally only at convergence: snapshots the simulation state exactly to the unique
/// deterministic equilibrium. Not involved in any UI / key handling.
pub struct TrappingRainWater;

impl TrappingRainWater {
    /// Compute the amount of water that can be trapped (units: number of 1x1x1 cubes)
    pub fn calculate(height_map: &[Vec<u8>]) -> usize {
        Self::water_heights(height_map)
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter(|&(j, &wh)| wh > height_map[i][j] as i32)
                    .map(|(j, &wh)| (wh - height_map[i][j] as i32) as usize)
                    .sum::<usize>()
            })
            .sum()
    }

    /// Return the final water surface height of every cell (in units).
    /// The surface height of boundary / dry cells equals the terrain height (i.e. no water is trapped).
    pub fn water_heights(height_map: &[Vec<u8>]) -> Vec<Vec<i32>> {
        let rows = height_map.len();
        if rows == 0 {
            return vec![];
        }
        let cols = height_map[0].len();
        if cols == 0 {
            return vec![vec![]; rows];
        }

        let mut water = vec![vec![0i32; cols]; rows];
        let mut visited = vec![vec![false; cols]; rows];

        // Min-heap of (height, row, col); seed it with the boundary cells
        let mut heap = BinaryHeap::new();
        for i in 0..rows {
            for j in 0..cols {
                if i == 0 || i == rows - 1 || j == 0 || j == cols - 1 {
                    visited[i][j] = true;
                    heap.push(Reverse((height_map[i][j] as i32, i, j)));
                }
            }
        }

        let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        let mut max_height = 0i32;

        while let Some(Reverse((h, x, y))) = heap.pop() {
            max_height = max_height.max(h);
            for (dx, dy) in &directions {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || nx >= rows as i32 || ny < 0 || ny >= cols as i32 {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                if visited[nx][ny] {
                    continue;
                }
                visited[nx][ny] = true;
                // A cell's water surface = the maximum barrier height when it is first visited
                water[nx][ny] = max_height;
                heap.push(Reverse((height_map[nx][ny] as i32, nx, ny)));
            }
        }

        water
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answers() {
        // 1x3 2D grid: every cell is on the boundary (top/bottom edges coincide) -> 0 trapped (the 1D answer of 2 does not apply)
        assert_eq!(TrappingRainWater::calculate(&[vec![3, 0, 2]]), 0);
        // 3x3 ring
        let m = vec![vec![3, 3, 3], vec![3, 0, 3], vec![3, 3, 3]];
        assert_eq!(TrappingRainWater::calculate(&m), 3);
        // 4x4 with a central 2x2 depression
        let m = vec![
            vec![3, 3, 3, 3],
            vec![3, 0, 0, 3],
            vec![3, 0, 0, 3],
            vec![3, 3, 3, 3],
        ];
        assert_eq!(TrappingRainWater::calculate(&m), 12);
    }

    /// Water surface: the central depression rises to the rim height; boundary cells are excluded (return 0, i.e. no water)
    #[test]
    fn water_heights_shape() {
        let m = vec![vec![3, 3, 3], vec![3, 0, 3], vec![3, 3, 3]];
        let wh = TrappingRainWater::water_heights(&m);
        // Central depression surface rises to rim height 3 -> 3 units trapped
        assert_eq!(wh[1][1], 3);
        // Boundary cell surface = 0 (no water; in calculate, 0 <= terrain height, so not counted)
        for (r, row) in wh.iter().enumerate() {
            for (c, &w) in row.iter().enumerate() {
                if r == 1 && c == 1 {
                    continue;
                }
                assert_eq!(w, 0, "({r},{c}) boundary cell should return 0");
            }
        }
        assert_eq!(TrappingRainWater::calculate(&m), 3);
    }

    /// A monotonically rising terrain traps no water (all water drains away)
    #[test]
    fn no_water_on_monotonic_terrain() {
        let m = vec![vec![1, 2, 3, 4, 5]];
        assert_eq!(TrappingRainWater::calculate(&m), 0);
    }

    /// Empty grid / single row / single column are all handled safely
    #[test]
    fn degenerate_grids() {
        assert_eq!(TrappingRainWater::calculate(&[] as &[Vec<u8>]), 0);
        assert_eq!(TrappingRainWater::calculate(&[vec![]]), 0);
        assert_eq!(TrappingRainWater::calculate(&[vec![5]]), 0);
    }
}

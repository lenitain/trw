use ratatui_wireframe::model::Model;
use rayon::prelude::*;

/// Vertical field of view in degrees.
pub const FOV_DEG: f64 = 60.0;

/// Auto-fit headroom: the model fills 1/FIT_MARGIN of the screen height.
pub const FIT_MARGIN: f64 = 2.0;

/// A 3x3 row-major rotation matrix (model -> world).
type Mat3 = [[f64; 3]; 3];

/// Identity rotation.
const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Multiply two 3x3 matrices.
fn mat_mul(a: Mat3, b: Mat3) -> Mat3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

/// Rotation around the world X axis.
fn rot_x(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

/// Rotation around the world Y axis.
fn rot_y(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// The camera degrees of freedom.
///
/// `center` is the **rotation center** (model-space coordinates): all vertices are first translated near center,
/// then rotated/scaled around it. Default is (0,0,0); for an n×n×n container it is usually (n/2, n/2, n/2),
/// so the view revolves around the container's center rather than a corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewState {
    pub rot: Mat3,
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
    pub dist: f64,
    pub pan_x: f64,
    pub pan_y: f64,
    /// Rotation center (model space)
    pub center: [f64; 3],
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            rot: IDENTITY,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            dist: 8.0,
            pan_x: 0.0,
            pan_y: 0.0,
            center: [0.0, 0.0, 0.0],
        }
    }
}

/// Wrap an angle into [-PI, PI].
pub fn normalize_angle(a: f64) -> f64 {
    use std::f64::consts::PI;
    let mut a = a % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

impl ViewState {
    pub fn normalize(&mut self) {
        self.yaw = normalize_angle(self.yaw);
        self.pitch = normalize_angle(self.pitch);
        self.roll = normalize_angle(self.roll);
    }

    pub fn add_yaw(&mut self, d: f64) {
        self.rot = mat_mul(rot_y(-d), self.rot);
        self.yaw += d;
    }

    pub fn add_pitch(&mut self, d: f64) {
        self.rot = mat_mul(rot_x(d), self.rot);
        self.pitch += d;
    }

    pub fn spin_local(&mut self, d: f64) {
        self.rot = mat_mul(self.rot, rot_y(d));
        self.yaw += d;
    }

    pub fn add_dist_delta(&mut self, delta: f64) {
        self.dist = (self.dist + delta).clamp(0.05, 100_000.0);
    }

    /// Auto-fit: set the distance so the model fills the view.
    pub fn fit_to(&mut self, extent: f64) {
        let r = extent.max(1e-6);
        self.dist = r / (FOV_DEG / 2.0).to_radians().tan() * FIT_MARGIN;
    }

    pub fn center_origin(&mut self) {
        self.pan_x = 0.0;
        self.pan_y = 0.0;
    }
}

/// Bounding box of the model (computed with a rayon parallel reduction; cheaper with many vertices).
pub fn bounds(m: &Model) -> ([f64; 3], [f64; 3]) {
    let init = || ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    m.vertices
        .par_iter()
        .fold(init, |(mut mn, mut mx), &(x, y, z)| {
            mn[0] = mn[0].min(x);
            mn[1] = mn[1].min(y);
            mn[2] = mn[2].min(z);
            mx[0] = mx[0].max(x);
            mx[1] = mx[1].max(y);
            mx[2] = mx[2].max(z);
            (mn, mx)
        })
        .reduce(init, |(a1, a2), (b1, b2)| {
            (
                [a1[0].min(b1[0]), a1[1].min(b1[1]), a1[2].min(b1[2])],
                [a2[0].max(b2[0]), a2[1].max(b2[1]), a2[2].max(b2[2])],
            )
        })
}

/// The model's geometric-mean length.
pub fn model_extent(m: &Model) -> f64 {
    let (min, max) = bounds(m);
    let (dx, dy, dz) = (
        (max[0] - min[0]).max(1e-9),
        (max[1] - min[1]).max(1e-9),
        (max[2] - min[2]).max(1e-9),
    );
    (dx * dy * dz).cbrt()
}

/// Project a model-space vertex to canvas coordinates; None when behind the camera.
pub fn project_point(p: (f64, f64, f64), v: &ViewState, px_h: usize) -> Option<(f64, f64)> {
    let f = (px_h as f64 / 2.0) / (FOV_DEG / 2.0).to_radians().tan();
    let r = v.rot;
    // First translate near the rotation center, then rotate around it
    let (dx, dy, dz) = (p.0 - v.center[0], p.1 - v.center[1], p.2 - v.center[2]);
    let rx = r[0][0] * dx + r[0][1] * dy + r[0][2] * dz + v.pan_x;
    let ry = r[1][0] * dx + r[1][1] * dy + r[1][2] * dz + v.pan_y;
    let rz = r[2][0] * dx + r[2][1] * dy + r[2][2] * dz;
    let z = v.dist - rz;
    if z <= 0.1 {
        return None;
    }
    let (sr, cr) = v.roll.sin_cos();
    let (dx, dy) = (rx - v.pan_x, ry - v.pan_y);
    let (rxr, ryr) = (v.pan_x + dx * cr - dy * sr, v.pan_y + dx * sr + dy * cr);
    Some((f * rxr / z, f * ryr / z))
}

/// Batch-project all vertices. Also outputs per-vertex view-space depth (z).
pub fn project_batch(
    verts: &[(f64, f64, f64)],
    v: &ViewState,
    px_h: usize,
    out: &mut [[f64; 2]],
    ok: &mut [bool],
    depths: &mut [f32],
) {
    let f = (px_h as f64 / 2.0) / (FOV_DEG / 2.0).to_radians().tan();
    let (sr, cr) = v.roll.sin_cos();
    let r = v.rot;
    let px = v.pan_x;
    let py = v.pan_y;
    let dist = v.dist;
    let (cx, cy, cz) = (v.center[0], v.center[1], v.center[2]);
    // rayon parallelism: vertex projections are independent of each other (the render vertex bottleneck).
    // First map out each vertex's result (pure function, no shared mutable state), then write back sequentially.
    let results: Vec<(bool, [f64; 2], f32)> = verts
        .par_iter()
        .map(|p| {
            // First translate near the rotation center, then rotate around it
            let (dx, dy, dz) = (p.0 - cx, p.1 - cy, p.2 - cz);
            let rx = r[0][0] * dx + r[0][1] * dy + r[0][2] * dz + px;
            let ry = r[1][0] * dx + r[1][1] * dy + r[1][2] * dz + py;
            let rz = r[2][0] * dx + r[2][1] * dy + r[2][2] * dz;
            let z = dist - rz;
            if z <= 0.1 {
                return (false, [0.0; 2], f32::MAX);
            }
            let (dx, dy) = (rx - px, ry - py);
            let rxr = px + dx * cr - dy * sr;
            let ryr = py + dx * sr + dy * cr;
            (true, [f * rxr / z, f * ryr / z], z as f32)
        })
        .collect();
    for (i, (is_ok, pt, depth)) in results.into_iter().enumerate() {
        out[i] = pt;
        ok[i] = is_ok;
        depths[i] = depth;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_center_is_origin() {
        let v = ViewState::default();
        let p = project_point((0.0, 0.0, 0.0), &v, 100).unwrap();
        assert!(p.0.abs() < 1e-6 && p.1.abs() < 1e-6);
    }

    #[test]
    fn project_behind_camera_is_none() {
        let v = ViewState::default();
        // dz=100, dist=8 -> z = 8-100 < 0.1 -> behind the camera
        assert!(project_point((0.0, 0.0, 100.0), &v, 100).is_none());
    }

    #[test]
    fn batch_project_matches_single() {
        let v = ViewState {
            yaw: 0.7,
            pitch: -0.3,
            roll: 0.2,
            dist: 12.0,
            pan_x: 1.0,
            pan_y: -2.0,
            center: [4.0, 4.0, 4.0],
            ..ViewState::default()
        };
        let verts = vec![
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            (7.0, 0.0, 1.0),
            (0.0, 100.0, 0.0), // behind -> None
        ];
        let n = verts.len();
        let mut out = vec![[0.0; 2]; n];
        let mut ok = vec![false; n];
        let mut depths = vec![0.0; n];
        project_batch(&verts, &v, 120, &mut out, &mut ok, &mut depths);
        for (i, &p) in verts.iter().enumerate() {
            let single = project_point(p, &v, 120);
            assert_eq!(ok[i], single.is_some(), "ok mismatch at {i}");
            if let Some(s) = single {
                assert!((out[i][0] - s.0).abs() < 1e-9, "x mismatch at {i}");
                assert!((out[i][1] - s.1).abs() < 1e-9, "y mismatch at {i}");
                assert!(depths[i] > 0.0);
            } else {
                assert_eq!(depths[i], f32::MAX);
            }
        }
    }

    #[test]
    fn bounds_and_extent() {
        let m = Model {
            vertices: vec![
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (0.0, 4.0, 0.0),
                (0.0, 0.0, 8.0),
            ],
            edges: vec![],
        };
        let (min, max) = bounds(&m);
        assert_eq!(min, [0.0, 0.0, 0.0]);
        assert_eq!(max, [2.0, 4.0, 8.0]);
        // geometric-mean side length = (2*4*8)^(1/3) = 64^(1/3) = 4
        assert!((model_extent(&m) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn rotate_then_reset_view() {
        let mut v = ViewState {
            center: [3.0, 3.0, 3.0],
            ..ViewState::default()
        };
        v.add_yaw(1.0);
        v.add_pitch(0.5);
        let reset = ViewState {
            center: [3.0, 3.0, 3.0],
            ..ViewState::default()
        };
        // the view has changed after rotating
        assert_ne!(v.rot, ViewState::default().rot);
        // reset = default view + keep the rotation center
        v = reset;
        assert_eq!(v.center, [3.0, 3.0, 3.0]);
    }
}

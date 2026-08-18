use clap::Parser;
use crossterm::{
    cursor::{Hide, Show},
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    widgets::{Block, Borders, Paragraph, Widget},
};
use ratatui_wireframe::model::Model;

use std::{
    collections::HashMap,
    error::Error,
    io::{self},
    sync::mpsc::{self},
    thread,
    time::{Duration, Instant},
};

// Standard TRW algorithm: used only to snapshot the deterministic equilibrium exactly at convergence (not part of any UI/key handling)
mod algorithm;
mod particle;
mod physics;
mod render;
mod terrain;
mod view;
mod water;

use particle::ParticleSystem;
use physics::Physics;
use terrain::Terrain;
use view::ViewState;

// Wireforge constants
const ROT_RATE: f64 = 169.0 / 128.0;
const MOVE_RATE: f64 = 83.0 / 128.0;
const PRESS_ROT_STEP: f64 = 7.0 / 128.0;
const PRESS_MOVE_FRACTION: f64 = 33.0 / 256.0;
const SPIN_RATE: f64 = 169.0 / 256.0;
const TAP_TIMEOUT: Duration = Duration::from_millis(600);
const HOLD_TIMEOUT: Duration = Duration::from_millis(100);

/// Motion state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Motion {
    YawLeft,
    YawRight,
    PitchUp,
    PitchDown,
    RollPlus,
    RollMinus,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    MoveForward,
    MoveBack,
}

/// Motion input state
#[derive(Debug, Clone, Copy)]
struct MotionInput {
    steady: bool,
    last: Instant,
}

/// Map key to motion
fn motion_for(code: KeyCode, shift: bool) -> Option<Motion> {
    use KeyCode::*;
    Some(match code {
        Left if shift => Motion::MoveLeft,
        Right if shift => Motion::MoveRight,
        Up if shift => Motion::MoveUp,
        Down if shift => Motion::MoveDown,
        Char('h') | Char('H') if shift => Motion::MoveLeft,
        Char('l') | Char('L') if shift => Motion::MoveRight,
        Char('k') | Char('K') if shift => Motion::MoveUp,
        Char('j') | Char('J') if shift => Motion::MoveDown,
        Char('=') | Char('+') => Motion::MoveForward,
        Char('-') => Motion::MoveBack,
        Left | Char('h') => Motion::YawLeft,
        Right | Char('l') => Motion::YawRight,
        Up | Char('k') => Motion::PitchUp,
        Down | Char('j') => Motion::PitchDown,
        Char('r') => Motion::RollPlus,
        Char('e') => Motion::RollMinus,
        _ => return None,
    })
}

/// HUD fold state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hud {
    Collapsed,
    Expanded,
}

/// Rendering engine state
struct Engine {
    screen: render_impl::Screen,
    raster: render_impl::Rasterizer,
    hud_buf: Option<Buffer>,
}

impl Engine {
    fn new(w: usize, h: usize) -> Self {
        Engine {
            screen: render_impl::Screen::new(w, h),
            raster: render_impl::Rasterizer::new(),
            hud_buf: None,
        }
    }

    fn resize(&mut self, w: usize, h: usize) {
        self.screen.resize(w, h);
        self.hud_buf = None;
    }
}

/// Application state
struct App {
    // View state
    view: ViewState,
    held: HashMap<Motion, MotionInput>,
    auto_spin: bool,
    hud: Hud,
    show_axes: bool,
    dirty: bool,
    terrain_model: Model,
    water_model: Model,

    // Project-specific state
    terrain: Terrain,
    particles: ParticleSystem,
    physics: Physics,
    /// Rotation center (container center = (n/2, n/2, n/2))
    center: [f64; 3],
    paused: bool,
    raining: bool,
    trapped_water: usize,
    /// X-Ray mode: see-through display of rain particles hidden behind the terrain
    xray: bool,
    /// Whether water levels have converged after rain stops (physics frozen; trapped stays stable)
    settled: bool,
    grid_size: usize,
}

impl App {
    fn new(grid_size: usize) -> Self {
        let terrain = Terrain::new(grid_size, grid_size);
        let particles = ParticleSystem::new();
        let physics = Physics::new(grid_size, grid_size);
        let terrain_model = render::build_terrain_model(&terrain);
        let water_model = render::build_water_model(&physics.water, &terrain, &particles);
        // rotation center = container center (n/2, n/2, n/2)
        let center = [grid_size as f64 / 2.0; 3];
        let view = ViewState {
            center,
            ..ViewState::default()
        };
        App {
            view,
            held: HashMap::new(),
            auto_spin: false,
            hud: Hud::Collapsed,
            show_axes: true,
            dirty: true,
            terrain_model,
            water_model,
            terrain,
            particles,
            physics,
            center,
            paused: false,
            raining: false,
            trapped_water: 0,
            xray: false,
            settled: false,
            grid_size,
        }
    }

    /// Rebuild only the water model (called whenever levels/particles change per frame; no need if terrain is unchanged)
    fn rebuild_water_model(&mut self) {
        self.water_model =
            render::build_water_model(&self.physics.water, &self.terrain, &self.particles);
    }

    /// Rebuild the terrain model (called only when the terrain is regenerated)
    fn rebuild_terrain_model(&mut self) {
        self.terrain_model = render::build_terrain_model(&self.terrain);
    }

    /// Generate random terrain
    ///
    /// Column heights scale with the grid size: max height = grid_size - 1,
    /// so a larger grid also gets taller terrain (the default 8 keeps the
    /// original 0..=7 range).
    fn generate_terrain(&mut self) {
        self.terrain
            .generate_random(self.grid_size.saturating_sub(1) as u8);
        self.particles.clear();
        self.physics.clear();
        self.trapped_water = 0;
        self.settled = false;
        self.rebuild_terrain_model();
        self.rebuild_water_model();
        self.dirty = true;
    }

    /// Toggle rain
    fn toggle_rain(&mut self) {
        self.raining = !self.raining;
        self.settled = false;
        self.dirty = true;
    }

    /// Toggle pause (pause/resume the whole physics simulation)
    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        self.dirty = true;
    }

    /// Reset the view: restore the default camera but keep the rotation center (container center)
    fn reset_view(&mut self) {
        self.view = ViewState::default();
        self.view.center = self.center;
        self.view.fit_to(view::model_extent(&self.terrain_model));
        self.dirty = true;
    }

    /// Clear particles
    fn clear_particles(&mut self) {
        self.particles.clear();
        self.physics.clear();
        self.trapped_water = 0;
        self.settled = false;
        self.rebuild_water_model();
        self.dirty = true;
    }

    /// Rain: accumulate rainfall by real dt (independent of frame rate; random position is the allowed random source)
    fn spawn_particles(&mut self, dt: f64) {
        if !self.raining || self.paused || self.settled {
            return;
        }
        self.physics
            .add_rain(dt, &mut self.particles, &self.terrain);
    }

    /// Update physics with real dt
    fn update_physics(&mut self, dt: f64) {
        if self.paused || self.settled {
            return;
        }
        self.physics.update(&mut self.particles, &self.terrain, dt);
        self.particles
            .remove_out_of_bounds(self.grid_size as f64, self.grid_size as f64, -5.0);
        // live trapped-water count (units: number of 1x1x1 cubes)
        self.trapped_water = self.physics.total_water_units().round() as usize;

        // after rain stops: wait until flow and drainage mostly stop and all particles settle -> converge and freeze
        // flow rules are deterministic and the converged value depends only on the terrain -> same container, same result
        if !self.raining {
            // guard: never converge when no rain has fallen / no water exists (avoid misjudging dry land)
            let has_water = self.physics.water.total_water() > 0.0;
            let no_more_flow = self.physics.residual() < 0.05;
            let all_settled = !self
                .particles
                .particles
                .iter()
                .any(|p| (p.z - p.target_z).abs() > 0.01);
            if has_water && no_more_flow && all_settled {
                let total = self.physics.total_water_units();
                let answer = self.physics.equilibrium_units(&self.terrain);
                if (total - answer as f64).abs() <= 1.0 {
                    // saturated: snapshot exactly to the unique deterministic equilibrium (result independent of the rain path)
                    self.trapped_water = self.physics.finalize(&mut self.particles, &self.terrain);
                } else {
                    // not saturated: freeze at the current converged value
                    self.trapped_water = total.round() as usize;
                }
                self.settled = true;
            }
        }
    }

    /// Handle input event; returns true to quit
    fn handle_input(&mut self, ev: Event) -> bool {
        if let Event::Key(key) = ev {
            let shift = key.modifiers.contains(KeyModifiers::SHIFT);
            let now = Instant::now();
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return true,
                KeyCode::Char('?') => {
                    self.hud = match self.hud {
                        Hud::Collapsed => Hud::Expanded,
                        Hud::Expanded => Hud::Collapsed,
                    };
                    self.dirty = true;
                }
                KeyCode::Tab => {
                    self.show_axes = !self.show_axes;
                    self.dirty = true;
                }
                KeyCode::Char(' ') => {
                    self.auto_spin = !self.auto_spin;
                    self.dirty = true;
                }
                KeyCode::Char('f') | KeyCode::Char('F') if shift => {
                    self.view.fit_to(view::model_extent(&self.terrain_model));
                    self.dirty = true;
                }
                KeyCode::Char('f') | KeyCode::Char('F') => {
                    self.view.center_origin();
                    self.dirty = true;
                }
                KeyCode::Char('0') => {
                    self.reset_view();
                }
                // Project-specific keys
                KeyCode::Char('g') => {
                    self.generate_terrain();
                }
                KeyCode::Char('w') => {
                    self.toggle_rain();
                }
                KeyCode::Char('p') => {
                    self.toggle_pause();
                }
                KeyCode::Char('c') => {
                    self.clear_particles();
                }
                KeyCode::Char('x') => {
                    self.xray = !self.xray;
                    self.dirty = true;
                }
                // Motion keys
                _ => {
                    if let Some(m) = motion_for(key.code, shift) {
                        let ms = view::model_extent(&self.terrain_model);
                        if let Some(st) = self.held.get_mut(&m) {
                            st.steady = true;
                            st.last = now;
                        } else {
                            press_step(&mut self.view, m, ms);
                            self.held.insert(
                                m,
                                MotionInput {
                                    steady: false,
                                    last: now,
                                },
                            );
                        }
                        self.dirty = true;
                    }
                }
            }
        }
        false
    }

    /// Update held motion state
    fn update_held(&mut self, now: Instant, dt: f64) {
        let ms = view::model_extent(&self.terrain_model);
        let mut stopped: Vec<Motion> = Vec::new();
        for (m, st) in self.held.iter_mut() {
            let timeout = if st.steady { HOLD_TIMEOUT } else { TAP_TIMEOUT };
            if now.duration_since(st.last) > timeout {
                stopped.push(*m);
            } else if st.steady {
                continuous_step(&mut self.view, *m, ms, dt);
            }
        }
        for m in stopped {
            self.held.remove(&m);
        }
    }
}

/// Single press step
fn press_step(v: &mut ViewState, m: Motion, scale: f64) {
    apply_motion_step(v, m, PRESS_ROT_STEP, scale * PRESS_MOVE_FRACTION);
}

/// Continuous held step
fn continuous_step(v: &mut ViewState, m: Motion, scale: f64, dt: f64) {
    apply_motion_step(v, m, ROT_RATE * dt, scale * MOVE_RATE * dt);
}

/// Apply motion step
fn apply_motion_step(v: &mut ViewState, m: Motion, rot: f64, mv: f64) {
    match m {
        Motion::YawLeft => v.add_yaw(rot),
        Motion::YawRight => v.add_yaw(-rot),
        Motion::PitchUp => v.add_pitch(-rot),
        Motion::PitchDown => v.add_pitch(rot),
        Motion::RollPlus => v.roll -= rot,
        Motion::RollMinus => v.roll += rot,
        Motion::MoveLeft => v.pan_x -= mv,
        Motion::MoveRight => v.pan_x += mv,
        Motion::MoveUp => v.pan_y += mv,
        Motion::MoveDown => v.pan_y -= mv,
        Motion::MoveForward => v.add_dist_delta(-mv),
        Motion::MoveBack => v.add_dist_delta(mv),
    }
    v.normalize();
}

/// HUD layout helper: produce the status lines
fn hud_layout(app: &App) -> (String, String) {
    let mut row0 = format!(
        "TRW | yaw={:.2} pitch={:.2} roll={:.2} dist={:.2} pan=({:.2},{:.2})",
        app.view.yaw, app.view.pitch, app.view.roll, app.view.dist, app.view.pan_x, app.view.pan_y
    );

    // Wireforge style: show the question-mark hint only when collapsed
    if app.hud == Hud::Collapsed {
        row0.push_str("   [?] keys");
    }

    let paused_str = if app.paused { "YES" } else { "NO" };
    let xray_str = if app.xray { "YES" } else { "NO" };
    let row1 = format!(
        "Water: {} particles | paused: {} | xray: {} | trapped: {} units",
        app.particles.count(),
        paused_str,
        xray_str,
        app.trapped_water
    );

    (row0, row1)
}

/// Copy rows [y0, y1) of the HUD buffer into the screen
fn blit_hud_rows(screen: &mut render_impl::Screen, hud: &Buffer, y0: u16, y1: u16) {
    let (w, _) = screen.size();
    for y in y0..y1 {
        for x in 0..w {
            let cell = &hud[(x as u16, y)];
            let ch = cell.symbol().chars().next().unwrap_or(' ');
            screen.set(x, y as usize, ch, render_impl::color_idx(cell.fg));
        }
    }
}

/// The help overlay text
const HELP: &[&str] = &[
    "=== keys ===",
    "",
    "Rotate:",
    "  yaw left  <- / h        yaw right  -> / l",
    "  pitch up  ^ / k         pitch down v / j",
    "  roll      r / e",
    "",
    "Move:",
    "  left      Shift+<- / h  right      Shift+-> / l",
    "  up        Shift+^ / k   down       Shift+v / j",
    "  nearer    =             farther    -",
    "",
    "Keys:",
    "  center    f             fit        Shift+f",
    "  reset     0             spin       Space",
    "  keys      ?             axes       Tab",
    "  generate  g             water      w",
    "  pause     p             xray       x",
    "  clear     c             quit       q / Esc",
    "",
    "[?] close help",
];

/// Render one frame
fn render_frame(
    app: &mut App,
    engine: &mut Engine,
    stdout: &mut io::Stdout,
) -> Result<(), Box<dyn Error>> {
    let (w, h) = engine.screen.size();
    if w == 0 || h == 0 {
        return Ok(());
    }
    let w16 = w as u16;
    let h16 = h as u16;

    let (row0, row1) = hud_layout(app);

    // HUD rows: row0 + row1 = 2 rows, or more if help is shown
    let overlay: Option<Vec<String>> = if app.hud == Hud::Expanded {
        Some(HELP.iter().map(|s| s.to_string()).collect())
    } else {
        None
    };

    // Reallocate HUD buffer if needed
    let needs_realloc = engine
        .hud_buf
        .as_ref()
        .is_none_or(|b| b.area.width != w16 || b.area.height != h16);
    if needs_realloc {
        engine.hud_buf = Some(Buffer::empty(Rect::new(0, 0, w16, h16)));
    } else if let Some(b) = engine.hud_buf.as_mut() {
        b.reset();
    }

    // Draw HUD rows
    let canvas_top: u16;
    {
        let hud = engine.hud_buf.as_mut().unwrap();
        Paragraph::new(row0).render(Rect::new(0, 0, w16, 1), hud);
        Paragraph::new(row1).render(Rect::new(0, 1, w16, 1), hud);
        canvas_top = 2;
    }

    // Draw overlay or model
    let canvas_area = if let Some(lines) = overlay {
        let top = canvas_top;
        let height = h16.saturating_sub(top);
        let area = Rect::new(0, top, w16, height);
        let block = Block::default().borders(Borders::ALL);
        let inner = block.inner(area);
        let hud = engine.hud_buf.as_mut().unwrap();
        block.render(area, hud);
        for (i, line) in lines.iter().enumerate() {
            if i as u16 >= inner.height {
                break;
            }
            Paragraph::new(line.as_str())
                .render(Rect::new(inner.x, inner.y + i as u16, inner.width, 1), hud);
        }
        None
    } else {
        Some(Rect::new(
            0,
            canvas_top,
            w16,
            h16.saturating_sub(canvas_top),
        ))
    };

    // Blit HUD to screen
    {
        let screen = &mut engine.screen;
        let hud = engine.hud_buf.as_ref().unwrap();
        blit_hud_rows(screen, hud, 0, canvas_top);
    }

    if let Some(area) = canvas_area {
        let cw = area.width as usize;
        let ch = area.height as usize;
        if cw > 0 && ch > 0 {
            engine.raster.resize(cw, ch);

            // render the terrain (uses the terminal default foreground; resetting the depth buffer = a new frame)
            engine.raster.render_colored(
                &app.terrain_model,
                &app.view,
                render_impl::RenderOpts {
                    region: (0, canvas_top as usize, cw, ch),
                    show_axes: app.show_axes,
                    edge_color: render_impl::Fg::Default as u8, // terminal default foreground
                    keep_depth: false,
                },
                &mut engine.screen,
            );

            // render the water (blue columns + falling drops):
            // - normal mode: keep the terrain depth buffer -> water columns are occluded by the terrain
            // - X-Ray mode: reset the depth buffer -> see-through (drawn above the terrain)
            engine.raster.render_colored(
                &app.water_model,
                &app.view,
                render_impl::RenderOpts {
                    region: (0, canvas_top as usize, cw, ch),
                    show_axes: false, // no axes (shown only when rendering the terrain)
                    edge_color: render_impl::Fg::LightBlue as u8, // blue water columns/drops
                    keep_depth: !app.xray,
                },
                &mut engine.screen,
            );
        }
    } else {
        // Overlay: blit the rest
        let screen = &mut engine.screen;
        let hud = engine.hud_buf.as_ref().unwrap();
        blit_hud_rows(screen, hud, canvas_top, h16);
    }

    // Present
    engine.screen.present(stdout)?;
    Ok(())
}

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(
    name = "trw",
    author,
    version,
    about = "3D Trapping Rain Water visualization demo"
)]
struct Args {
    /// Grid size
    #[arg(short, long, default_value = "8")]
    grid_size: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let grid_size = args.grid_size;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, Hide)?;

    let (tx, rx) = mpsc::channel::<Event>();

    // Input thread
    thread::spawn(move || {
        loop {
            if let Ok(ev) = event::read()
                && tx.send(ev).is_err()
            {
                break;
            }
        }
    });

    let mut app = App::new(grid_size);
    app.generate_terrain();
    app.view.fit_to(view::model_extent(&app.terrain_model));

    let (cols, rows) = crossterm::terminal::size()?;
    let mut engine = Engine::new(cols as usize, rows as usize);

    // Render initial frame
    render_frame(&mut app, &mut engine, &mut stdout)?;
    app.dirty = false;

    let mut last = Instant::now();

    loop {
        let now = Instant::now();
        let dt = (now - last).as_secs_f64().min(0.1);
        last = now;

        // Handle resize
        if let Ok((new_cols, new_rows)) = crossterm::terminal::size()
            && (new_cols as usize != engine.screen.size().0
                || new_rows as usize != engine.screen.size().1)
        {
            engine.resize(new_cols as usize, new_rows as usize);
            app.dirty = true;
        }

        // Drain input
        while let Ok(ev) = rx.try_recv() {
            if let Event::Resize(cols, rows) = ev {
                engine.resize(cols as usize, rows as usize);
                app.dirty = true;
                continue;
            }
            if app.handle_input(ev) {
                execute!(stdout, Show, LeaveAlternateScreen)?;
                disable_raw_mode()?;
                return Ok(());
            }
        }

        // Update held keys
        app.update_held(now, dt);

        // Auto spin
        if app.auto_spin {
            app.view.spin_local(SPIN_RATE * dt);
            app.view.normalize();
            app.dirty = true;
        }

        // Spawn rain particles
        if app.raining && !app.paused {
            app.spawn_particles(dt);
        }

        // Update physics
        let had_particles = !app.particles.particles.is_empty();
        let had_water = app.physics.water.total_water() > 0.0;
        app.update_physics(dt);
        if !app.particles.particles.is_empty()
            || had_particles
            || app.physics.water.total_water() > 0.0
            || had_water
        {
            app.rebuild_water_model();
            app.dirty = true;
        }

        // Render if dirty
        if app.dirty {
            render_frame(&mut app, &mut engine, &mut stdout)?;
            app.dirty = false;
        }

        thread::sleep(Duration::from_millis(16)); // ~60 FPS
    }
}

/// The Wireforge rendering engine (Screen, Rasterizer, etc.)
/// Adapted from /home/lenitain/.projects/wireforge/src/render.rs
mod render_impl {
    use ratatui::style::Color;
    use ratatui_wireframe::model::Model;
    use std::io::Write;

    use crate::view::{self, ViewState};

    const SPACE: u64 = (' ' as u64) << 8;

    #[inline(always)]
    fn pack(ch: char, fg: u8) -> u64 {
        (ch as u64) << 8 | fg as u64
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    #[repr(u8)]
    pub enum Fg {
        Default = 0,
        Cyan = 1,
        Red = 2,
        Yellow = 3,
        LightBlue = 4,
    }

    impl Fg {
        pub fn sgr(self) -> &'static [u8] {
            match self {
                Fg::Default => b"\x1b[0m",
                Fg::Cyan => b"\x1b[36m",
                Fg::Red => b"\x1b[31m",
                Fg::Yellow => b"\x1b[33m",
                Fg::LightBlue => b"\x1b[94m",
            }
        }

        pub fn from_u8(v: u8) -> Fg {
            match v {
                1 => Fg::Cyan,
                2 => Fg::Red,
                3 => Fg::Yellow,
                4 => Fg::LightBlue,
                _ => Fg::Default,
            }
        }
    }

    pub fn color_idx(c: Color) -> u8 {
        match c {
            Color::Cyan => Fg::Cyan as u8,
            Color::Red => Fg::Red as u8,
            Color::Yellow => Fg::Yellow as u8,
            Color::LightBlue => Fg::LightBlue as u8,
            _ => Fg::Default as u8,
        }
    }

    pub struct Screen {
        w: usize,
        h: usize,
        cur: Vec<u64>,
        prev: Vec<u64>,
        out: Vec<u8>,
        full_repaint: bool,
    }

    impl Screen {
        pub fn new(w: usize, h: usize) -> Self {
            let n = w * h;
            Screen {
                w,
                h,
                cur: vec![SPACE; n],
                prev: vec![SPACE; n],
                out: Vec::with_capacity(4096),
                full_repaint: false,
            }
        }

        pub fn resize(&mut self, w: usize, h: usize) {
            if w == self.w && h == self.h {
                return;
            }
            self.w = w;
            self.h = h;
            self.cur = vec![SPACE; w * h];
            self.prev = vec![SPACE; w * h];
            self.full_repaint = true;
        }

        pub fn size(&self) -> (usize, usize) {
            (self.w, self.h)
        }

        #[inline]
        pub fn set(&mut self, x: usize, y: usize, ch: char, fg: u8) {
            if x < self.w && y < self.h {
                self.cur[y * self.w + x] = pack(ch, fg);
            }
        }

        fn push_usize(out: &mut Vec<u8>, mut n: usize) {
            if n == 0 {
                out.push(b'0');
                return;
            }
            let mut tmp = [0u8; 20];
            let mut i = tmp.len();
            while n > 0 {
                i -= 1;
                tmp[i] = b'0' + (n % 10) as u8;
                n /= 10;
            }
            out.extend_from_slice(&tmp[i..]);
        }

        fn push_cursor(out: &mut Vec<u8>, row: usize, col: usize) {
            out.push(0x1b);
            out.push(b'[');
            Self::push_usize(out, row + 1);
            out.push(b';');
            Self::push_usize(out, col + 1);
            out.push(b'H');
        }

        pub fn present<W: Write>(&mut self, out: &mut W) -> std::io::Result<usize> {
            self.out.clear();
            if self.full_repaint {
                self.out.extend_from_slice(b"\x1b[2J");
                self.full_repaint = false;
            }
            let mut changed = 0;
            for y in 0..self.h {
                let base = y * self.w;
                let cur_row = &self.cur[base..base + self.w];
                let prev_row = &self.prev[base..base + self.w];
                if cur_row == prev_row {
                    continue;
                }
                let mut x = 0;
                while x < self.w {
                    if cur_row[x] == prev_row[x] {
                        x += 1;
                        continue;
                    }
                    let start = x;
                    while x < self.w && cur_row[x] != prev_row[x] {
                        x += 1;
                    }
                    let end = x;
                    Self::push_cursor(&mut self.out, y, start);
                    let mut last_fg = u8::MAX;
                    for &cell in &cur_row[start..end] {
                        let fg = (cell & 0xff) as u8;
                        if fg != last_fg {
                            self.out.extend_from_slice(Fg::from_u8(fg).sgr());
                            last_fg = fg;
                        }
                        let ch = char::from_u32((cell >> 8) as u32).unwrap_or(' ');
                        let mut b = [0u8; 4];
                        self.out
                            .extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                    }
                    changed += end - start;
                }
            }
            std::mem::swap(&mut self.cur, &mut self.prev);
            self.cur.fill(SPACE);
            out.write_all(&self.out)?;
            out.flush()?;
            Ok(changed)
        }
    }

    pub struct Rasterizer {
        cw: usize,
        ch: usize,
        dots: Vec<u8>,
        colors: Vec<u8>,
        braille: [char; 256],
        proj: Vec<[f64; 2]>,
        proj_ok: Vec<bool>,
        proj_depth: Vec<f32>,
        proj_view: ViewState,
        proj_px_w: usize,
        proj_px_h: usize,
        proj_len: usize,
        depth_buf: Vec<f32>,
    }

    impl Default for Rasterizer {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Parameters for a single model render (keeps `render_colored` arguments under 7)
    pub struct RenderOpts {
        /// Canvas region (x, y, width, height)
        pub region: (usize, usize, usize, usize),
        /// Whether to show the axes (shown only when rendering the terrain)
        pub show_axes: bool,
        /// Edge color
        pub edge_color: u8,
        /// Whether to keep the depth buffer from the previous call (true = depth-test against earlier models)
        pub keep_depth: bool,
    }

    impl Rasterizer {
        pub fn new() -> Self {
            Rasterizer {
                cw: 0,
                ch: 0,
                braille: ratatui::symbols::braille::BRAILLE,
                dots: Vec::new(),
                colors: Vec::new(),
                proj: Vec::new(),
                proj_ok: Vec::new(),
                proj_depth: Vec::new(),
                proj_view: ViewState::default(),
                proj_px_w: 0,
                proj_px_h: 0,
                proj_len: 0,
                depth_buf: Vec::new(),
            }
        }

        pub fn resize(&mut self, cw: usize, ch: usize) {
            self.cw = cw;
            self.ch = ch;
            self.dots = vec![0; cw * ch];
            self.colors = vec![0; cw * ch];
            self.depth_buf = vec![f32::MAX; cw * 2 * ch * 4];
            self.proj_px_w = 0;
            self.proj_px_h = 0;
        }

        fn project(&mut self, model: &Model, view: &ViewState, px_w: usize, px_h: usize) {
            let n = model.vertices.len();
            if self.proj_len == n
                && self.proj_view == *view
                && self.proj_px_w == px_w
                && self.proj_px_h == px_h
            {
                return;
            }
            self.proj.clear();
            self.proj.resize(n, [0.0; 2]);
            self.proj_ok.clear();
            self.proj_ok.resize(n, false);
            self.proj_depth.clear();
            self.proj_depth.resize(n, f32::MAX);
            view::project_batch(
                &model.vertices,
                view,
                px_h,
                &mut self.proj,
                &mut self.proj_ok,
                &mut self.proj_depth,
            );
            self.proj_view = *view;
            self.proj_px_w = px_w;
            self.proj_px_h = px_h;
            self.proj_len = n;
        }

        /// Render a model.
        ///
        /// `keep_depth` (in `opts`): when `false`, resets the depth buffer (rendered independently,
        /// always drawn on top); when `true`, keeps the previous depth buffer so this model
        /// depth-tests against earlier models (e.g. particles keep the terrain depth buffer -> occluded by the terrain).
        pub fn render_colored(
            &mut self,
            model: &Model,
            view: &ViewState,
            opts: RenderOpts,
            screen: &mut Screen,
        ) {
            let RenderOpts {
                region,
                show_axes,
                edge_color,
                keep_depth,
            } = opts;
            let (rx, ry, cw, ch) = region;
            debug_assert!(cw > 0 && ch > 0);
            let px_w = cw * 2;
            let px_h = ch * 4;
            self.project(model, view, px_w, px_h);

            self.dots.fill(0);
            self.colors.fill(0);
            if !keep_depth {
                self.depth_buf.fill(f32::MAX);
            }

            // Fill depth buffer with box face depths for occlusion
            fill_box_faces(
                &self.proj,
                &self.proj_ok,
                &self.proj_depth,
                model.vertices.len(),
                px_w,
                px_h,
                &mut self.depth_buf,
            );

            for &(a, b) in &model.edges {
                if self.proj_ok[a] && self.proj_ok[b] {
                    let [x1, y1] = self.proj[a];
                    let [x2, y2] = self.proj[b];
                    paint_line_into(
                        x1,
                        y1,
                        x2,
                        y2,
                        self.proj_depth[a],
                        self.proj_depth[b],
                        edge_color,
                        self.cw,
                        px_w,
                        px_h,
                        &mut self.dots,
                        &mut self.colors,
                        &mut self.depth_buf,
                        true,
                    );
                }
            }

            // Axes (no depth testing — always visible)
            // axes are rooted at the rotation center (view.center = container center)
            let mut labels: Vec<(f64, f64, &str, u8)> = Vec::new();
            if show_axes {
                let axis_len = view::model_extent(model) / 0.618;
                let (cx, cy, cz) = (view.center[0], view.center[1], view.center[2]);
                let origin = view::project_point((cx, cy, cz), view, px_h);
                let ends = [
                    (
                        view::project_point((cx + axis_len, cy, cz), view, px_h),
                        "X",
                        Fg::Red as u8,
                    ),
                    (
                        view::project_point((cx, cy + axis_len, cz), view, px_h),
                        "Y",
                        Fg::Yellow as u8,
                    ),
                    (
                        view::project_point((cx, cy, cz + axis_len), view, px_h),
                        "Z",
                        Fg::LightBlue as u8,
                    ),
                ];
                if let Some((ox, oy)) = origin {
                    for (end, label, color) in ends {
                        if let Some((ex, ey)) = end {
                            paint_line_into(
                                ox,
                                oy,
                                ex,
                                ey,
                                0.0,
                                0.0,
                                color,
                                self.cw,
                                px_w,
                                px_h,
                                &mut self.dots,
                                &mut self.colors,
                                &mut self.depth_buf,
                                false,
                            );
                            labels.push((ex, ey, label, color));
                        }
                    }
                }
            }

            // Encode dots into screen cells
            let braille = self.braille;
            for cy in 0..ch {
                for cx in 0..cw {
                    let i = cy * cw + cx;
                    let p = self.dots[i];
                    if p != 0 {
                        screen.set(rx + cx, ry + cy, braille[p as usize], self.colors[i]);
                    }
                }
            }

            // Labels
            for (ex, ey, label, color) in labels {
                self.place_label(ex, ey, label, color, px_w, px_h, rx, ry, screen);
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn place_label(
            &self,
            ex: f64,
            ey: f64,
            label: &str,
            color: u8,
            px_w: usize,
            px_h: usize,
            rx: usize,
            ry: usize,
            screen: &mut Screen,
        ) {
            if self.cw < 2 || self.ch < 2 {
                return;
            }
            let left = -(px_w as f64) / 2.0;
            let right = (px_w as f64) / 2.0;
            let top = (px_h as f64) / 2.0;
            let bottom = -(px_h as f64) / 2.0;
            let cell_w = (px_w as f64) / (self.cw - 1) as f64;
            let cell_h = (px_h as f64) / (self.ch - 1) as f64;
            let label_x = left + ((ex - left) / cell_w).round() * cell_w;
            let label_y = top - ((top - ey) / cell_h).round() * cell_h;
            if label_x >= left && label_x <= right && label_y <= top && label_y >= bottom {
                let x = ((label_x - left) * (self.cw - 1) as f64 / px_w as f64) as u16;
                let y = ((top - label_y) * (self.ch - 1) as f64 / px_h as f64) as u16;
                if let Some(ch) = label.chars().next() {
                    screen.set(rx + x as usize, ry + y as usize, ch, color);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn paint_line_into(
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        depth_a: f32,
        depth_b: f32,
        color: u8,
        cw: usize,
        px_w: usize,
        px_h: usize,
        dots: &mut [u8],
        colors: &mut [u8],
        depth_buf: &mut [f32],
        depth_test: bool,
    ) {
        let px_wf = px_w as f64;
        let px_hf = px_h as f64;
        let (left, right) = (-px_wf / 2.0, px_wf / 2.0);
        let (bottom, top) = (-px_hf / 2.0, px_hf / 2.0);
        let Some((cx1, cy1, cx2, cy2)) = clip_line(x1, y1, x2, y2, left, right, bottom, top) else {
            return;
        };
        let Some((dx1, dy1)) = get_point(cx1, cy1, px_w, px_h) else {
            return;
        };
        let Some((dx2, dy2)) = get_point(cx2, cy2, px_w, px_h) else {
            return;
        };

        // Compute depth at clipped endpoints via parametric interpolation
        let orig_len_sq = ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) as f32;
        let (cd1, cd2) = if orig_len_sq > 1e-12 {
            let t1 = (((cx1 - x1) * (x2 - x1) + (cy1 - y1) * (y2 - y1))
                / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))) as f32;
            let t2 = (((cx2 - x1) * (x2 - x1) + (cy2 - y1) * (y2 - y1))
                / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))) as f32;
            (
                depth_a + t1 * (depth_b - depth_a),
                depth_a + t2 * (depth_b - depth_a),
            )
        } else {
            (depth_a, depth_b)
        };

        let pix_len_sq = (dx2 as f32 - dx1 as f32).powi(2) + (dy2 as f32 - dy1 as f32).powi(2);
        bresenham(dx1, dy1, dx2, dy2, |x, y| {
            if !depth_test {
                paint_dot(x, y, color, cw, px_w, px_h, dots, colors);
                return;
            }
            // Interpolate depth at this pixel
            let depth = if pix_len_sq > 0.0 {
                let t = ((x as f32 - dx1 as f32) * (dx2 as f32 - dx1 as f32)
                    + (y as f32 - dy1 as f32) * (dy2 as f32 - dy1 as f32))
                    / pix_len_sq;
                cd1 + t * (cd2 - cd1)
            } else {
                cd1
            };
            let idx = y * px_w + x;
            if idx < depth_buf.len() && depth <= depth_buf[idx] {
                paint_dot(x, y, color, cw, px_w, px_h, dots, colors);
                depth_buf[idx] = depth;
            }
        });
    }

    /// Convert canvas coordinates to pixel coordinates for depth filling.
    fn canvas_to_pixel(x: f64, y: f64, px_w: usize, px_h: usize) -> (f32, f32) {
        let px_wf = px_w as f64;
        let px_hf = px_h as f64;
        let (left, _right) = (-px_wf / 2.0, px_wf / 2.0);
        let (_bottom, top) = (-px_hf / 2.0, px_hf / 2.0);
        let px = ((x - left) * (px_wf - 1.0) / px_wf) as f32;
        let py = ((top - y) * (px_hf - 1.0) / px_hf) as f32;
        (px, py)
    }

    /// Rasterize a triangle into the depth buffer only (no pixel output).
    #[allow(clippy::too_many_arguments)]
    fn fill_triangle_depth(
        x0: f32,
        y0: f32,
        z0: f32,
        x1: f32,
        y1: f32,
        z1: f32,
        x2: f32,
        y2: f32,
        z2: f32,
        px_w: usize,
        px_h: usize,
        depth_buf: &mut [f32],
    ) {
        // Sort vertices by y-coordinate
        let mut verts = [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2)];
        verts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let (ax, ay, az) = verts[0];
        let (bx, by, bz) = verts[1];
        let (cx, cy, cz) = verts[2];

        if (cy - ay).abs() < 0.001 {
            return; // degenerate triangle
        }

        let y_start = (ay.max(0.0)).ceil() as i32;
        let y_end = (cy.min(px_h as f32 - 1.0)).floor() as i32;

        for y in y_start..=y_end {
            let yf = y as f32;
            // Long edge AC
            let t_ac = (yf - ay) / (cy - ay);
            let x_ac = ax + t_ac * (cx - ax);
            let z_ac = az + t_ac * (cz - az);

            // Short edge: upper half (AB) or lower half (BC)
            let (x_short, z_short) = if yf <= by {
                let t = if (by - ay).abs() > 0.001 {
                    (yf - ay) / (by - ay)
                } else {
                    0.0
                };
                (ax + t * (bx - ax), az + t * (bz - az))
            } else {
                let t = if (cy - by).abs() > 0.001 {
                    (yf - by) / (cy - by)
                } else {
                    0.0
                };
                (bx + t * (cx - bx), bz + t * (cz - bz))
            };

            let (xl, xr, zl, zr) = if x_ac < x_short {
                (x_ac, x_short, z_ac, z_short)
            } else {
                (x_short, x_ac, z_short, z_ac)
            };

            let xi_start = (xl.max(0.0)).ceil() as i32;
            let xi_end = (xr.min(px_w as f32 - 1.0)).floor() as i32;
            let x_range = xr - xl;

            for x in xi_start..=xi_end {
                let t = if x_range > 0.001 {
                    (x as f32 - xl) / x_range
                } else {
                    0.0
                };
                let z = zl + t * (zr - zl);
                let idx = y as usize * px_w + x as usize;
                if idx < depth_buf.len() && z < depth_buf[idx] {
                    depth_buf[idx] = z;
                }
            }
        }
    }

    /// Fill the depth buffer with face depths for all boxes in the model.
    /// Each group of 8 consecutive vertices forms a box (box8 layout).
    fn fill_box_faces(
        proj: &[[f64; 2]],
        proj_ok: &[bool],
        proj_depth: &[f32],
        n_verts: usize,
        px_w: usize,
        px_h: usize,
        depth_buf: &mut [f32],
    ) {
        // Face vertex indices within each 8-vertex box
        const BOX_FACES: [[usize; 4]; 6] = [
            [0, 1, 2, 3], // bottom
            [4, 5, 6, 7], // top
            [0, 1, 5, 4], // front
            [1, 2, 6, 5], // right
            [2, 3, 7, 6], // back
            [3, 0, 4, 7], // left
        ];

        let mut offset = 0;
        while offset + 8 <= n_verts {
            let all_ok = (0..8).all(|i| proj_ok[offset + i]);
            if all_ok {
                for face in &BOX_FACES {
                    let idx: [usize; 4] = face.map(|i| offset + i);
                    // Get pixel coords and depths for each face vertex
                    let v: [(f32, f32, f32); 4] = std::array::from_fn(|i| {
                        let [cx, cy] = proj[idx[i]];
                        let (px, py) = canvas_to_pixel(cx, cy, px_w, px_h);
                        (px, py, proj_depth[idx[i]])
                    });
                    // Split quad into two triangles: (0,1,2) and (0,2,3)
                    fill_triangle_depth(
                        v[0].0, v[0].1, v[0].2, v[1].0, v[1].1, v[1].2, v[2].0, v[2].1, v[2].2,
                        px_w, px_h, depth_buf,
                    );
                    fill_triangle_depth(
                        v[0].0, v[0].1, v[0].2, v[2].0, v[2].1, v[2].2, v[3].0, v[3].1, v[3].2,
                        px_w, px_h, depth_buf,
                    );
                }
            }
            offset += 8;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn clip_line(
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        left: f64,
        right: f64,
        bottom: f64,
        top: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        let mut p1 = (x1, y1);
        let mut p2 = (x2, y2);
        let mut r1 = region_code(p1, left, right, bottom, top);
        let mut r2 = region_code(p2, left, right, bottom, top);
        loop {
            if r1 & r2 != 0 {
                return None;
            }
            if r1 != 0 {
                p1 = intersect(p1, p2, r1, left, right, bottom, top);
                r1 = region_code(p1, left, right, bottom, top);
            } else if r2 != 0 {
                p2 = intersect(p2, p1, r2, left, right, bottom, top);
                r2 = region_code(p2, left, right, bottom, top);
            } else {
                return Some((p1.0, p1.1, p2.0, p2.1));
            }
        }
    }

    fn region_code(p: (f64, f64), left: f64, right: f64, bottom: f64, top: f64) -> u8 {
        let mut r = 0u8;
        if p.0 < left {
            r |= 1;
        } else if p.0 > right {
            r |= 2;
        }
        if p.1 < bottom {
            r |= 4;
        } else if p.1 > top {
            r |= 8;
        }
        r
    }

    fn intersect(
        p1: (f64, f64),
        p2: (f64, f64),
        region: u8,
        left: f64,
        right: f64,
        bottom: f64,
        top: f64,
    ) -> (f64, f64) {
        let dx = p2.0 - p1.0;
        let dy = p2.1 - p1.1;
        if region & 1 != 0 {
            let y = p1.1 + (left - p1.0) * dy / dx;
            return (left, y);
        }
        if region & 2 != 0 {
            let y = p1.1 + (right - p1.0) * dy / dx;
            return (right, y);
        }
        if region & 4 != 0 {
            let x = p1.0 + (bottom - p1.1) * dx / dy;
            return (x, bottom);
        }
        debug_assert!(region & 8 != 0);
        let x = p1.0 + (top - p1.1) * dx / dy;
        (x, top)
    }

    fn get_point(x: f64, y: f64, px_w: usize, px_h: usize) -> Option<(usize, usize)> {
        let px_wf = px_w as f64;
        let px_hf = px_h as f64;
        let (left, right) = (-px_wf / 2.0, px_wf / 2.0);
        let (bottom, top) = (-px_hf / 2.0, px_hf / 2.0);
        if x < left || x > right || y < bottom || y > top {
            return None;
        }
        let xd = ((x - left) * (px_wf - 1.0) / (right - left)).round() as usize;
        let yd = ((top - y) * (px_hf - 1.0) / (top - bottom)).round() as usize;
        Some((xd, yd))
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn paint_dot(
        x: usize,
        y: usize,
        color: u8,
        cw: usize,
        px_w: usize,
        px_h: usize,
        dots: &mut [u8],
        colors: &mut [u8],
    ) {
        if x >= px_w || y >= px_h {
            return;
        }
        let cell = (y >> 2) * cw + (x >> 1);
        dots[cell] |= 1 << ((x & 1) + 2 * (y & 3));
        colors[cell] = color;
    }

    fn bresenham(x1: usize, y1: usize, x2: usize, y2: usize, mut f: impl FnMut(usize, usize)) {
        let dx = x2.abs_diff(x1);
        let dy = y2.abs_diff(y1);
        if dx == 0 {
            for y in y1.min(y2)..=y1.max(y2) {
                f(x1, y);
            }
        } else if dy == 0 {
            for x in x1.min(x2)..=x1.max(x2) {
                f(x, y1);
            }
        } else if dy < dx {
            if x1 > x2 {
                line_low(x2, y2, x1, y1, &mut f);
            } else {
                line_low(x1, y1, x2, y2, &mut f);
            }
        } else if y1 > y2 {
            line_high(x2, y2, x1, y1, &mut f);
        } else {
            line_high(x1, y1, x2, y2, &mut f);
        }
    }

    fn line_low(x1: usize, y1: usize, x2: usize, y2: usize, f: &mut impl FnMut(usize, usize)) {
        let dx = (x2 - x1) as isize;
        let dy = (y2 as isize - y1 as isize).abs();
        let mut d = 2 * dy - dx;
        let mut y = y1;
        for x in x1..=x2 {
            f(x, y);
            if d > 0 {
                y = if y1 > y2 {
                    y.saturating_sub(1)
                } else {
                    y.saturating_add(1)
                };
                d -= 2 * dx;
            }
            d += 2 * dy;
        }
    }

    fn line_high(x1: usize, y1: usize, x2: usize, y2: usize, f: &mut impl FnMut(usize, usize)) {
        let dx = (x2 as isize - x1 as isize).abs();
        let dy = (y2 - y1) as isize;
        let mut d = 2 * dx - dy;
        let mut x = x1;
        for y in y1..=y2 {
            f(x, y);
            if d > 0 {
                x = if x1 > x2 {
                    x.saturating_sub(1)
                } else {
                    x.saturating_add(1)
                };
                d -= 2 * dy;
            }
            d += 2 * dx;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};

    /// Build a plain key event
    fn key(code: KeyCode) -> Event {
        Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    #[test]
    fn toggles_rain_pause_xray() {
        let mut app = App::new(4);

        // w: start/stop rain (toggle back and forth)
        assert!(!app.raining);
        assert!(!app.handle_input(key(KeyCode::Char('w'))));
        assert!(app.raining);
        app.handle_input(key(KeyCode::Char('w')));
        assert!(!app.raining);

        // p: pause/resume physics
        assert!(!app.paused);
        app.handle_input(key(KeyCode::Char('p')));
        assert!(app.paused);
        app.handle_input(key(KeyCode::Char('p')));
        assert!(!app.paused);

        // x: X-Ray see-through
        assert!(!app.xray);
        app.handle_input(key(KeyCode::Char('x')));
        assert!(app.xray);
        app.handle_input(key(KeyCode::Char('x')));
        assert!(!app.xray);
    }

    #[test]
    fn generate_rain_and_clear() {
        let mut app = App::new(4);
        // g: generate random terrain; heights scale with grid size (max = grid_size - 1 = 3)
        app.handle_input(key(KeyCode::Char('g')));
        assert!(
            app.terrain
                .heights
                .iter()
                .flatten()
                .all(|&h| h <= app.grid_size.saturating_sub(1) as u8)
        );

        // rain for a while to spawn particles
        app.handle_input(key(KeyCode::Char('w')));
        app.spawn_particles(2.0);
        assert!(app.particles.count() > 0);

        // c: clear particles and water levels
        app.handle_input(key(KeyCode::Char('c')));
        assert_eq!(app.particles.count(), 0);
        assert_eq!(app.trapped_water, 0);
        assert_eq!(app.physics.water.total_water(), 0.0);
    }

    #[test]
    fn terrain_heights_scale_with_grid_size() {
        // larger grid -> taller possible columns
        let mut big = App::new(16);
        big.handle_input(key(KeyCode::Char('g')));
        assert_eq!(big.terrain.max_height, 15);
        assert!(big.terrain.heights.iter().flatten().all(|&h| h <= 15));
        assert!(big.terrain.heights.iter().flatten().any(|&h| h > 7));

        // smaller grid -> shorter possible columns
        let mut small = App::new(4);
        small.handle_input(key(KeyCode::Char('g')));
        assert_eq!(small.terrain.max_height, 3);
        assert!(small.terrain.heights.iter().flatten().all(|&h| h <= 3));
    }

    #[test]
    fn quit_keys_exit() {
        assert!(App::new(4).handle_input(key(KeyCode::Char('q'))));
        assert!(App::new(4).handle_input(key(KeyCode::Esc)));
    }

    #[test]
    fn reset_view_preserves_center() {
        let mut app = App::new(8);
        // rotate the view a bit
        app.view.add_yaw(1.0);
        app.view.add_pitch(0.5);
        app.handle_input(key(KeyCode::Char('0')));
        // after reset the view returns to default and the rotation center stays at the container center (4,4,4)
        assert_eq!(app.view.center, [4.0, 4.0, 4.0]);
        assert_eq!(app.view.yaw, 0.0);
        assert_eq!(app.view.pitch, 0.0);
    }

    #[test]
    fn hud_layout_reflects_state() {
        let mut app = App::new(4);
        let (row0, row1) = hud_layout(&app);
        assert!(row0.contains("TRW"));
        assert!(row1.contains("paused: NO"));
        assert!(row1.contains("xray: NO"));
        app.toggle_pause();
        app.xray = true;
        let (_, row1) = hud_layout(&app);
        assert!(row1.contains("paused: YES"));
        assert!(row1.contains("xray: YES"));
    }
}

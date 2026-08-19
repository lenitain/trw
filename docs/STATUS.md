# trw — development status (handoff notes)

- Date: 2026-08-19
- Purpose: record the current architecture, recent changes, and open items so
  a fresh agent can continue without re-deriving context.

## Project
3D "Trapping Rain Water" visualization in the terminal: random terrain, rain
particles, water flows downhill and pools, then converges to the exact TRW
equilibrium (deterministic, path-independent).

## Motion input (unified model, shared with ca3d/wireforge)
- Kitty keyboard protocol is enabled at startup via
  `PushKeyboardEnhancementFlags(REPORT_EVENT_TYPES | REPORT_ALL_KEYS_AS_ESCAPE_CODES | DISAMBIGUATE_ESCAPE_CODES)`.
- Press/Repeat = key held; `Release` = stop immediately. There is NO
  timeout-based tap/hold detection.
- Enabling is NON-FATAL (`let _ = execute!`): unsupported terminals (e.g. the
  legacy Windows console API) run without it; crossterm still reports
  Press/Repeat/Release natively on Windows, so the motion model still works.
- No frame-rate cap: the fixed 16ms sleep was removed (loop runs as fast as
  possible).
- `update_held` sets `dirty` while a key is held, otherwise the continuous
  motion is applied but never redrawn (this was the "stutter" symptom).

## Physics convergence (IMPORTANT — fixed bug)
- Water flow is dt-scaled: `flow = min(level, diff * flow_rate * dt * n)`.
- The old convergence check compared the ABSOLUTE per-frame residual
  (`residual() < 0.05`), which depends on frame pacing: after the 16ms sleep
  was removed, dt ~1ms made the per-frame residual tiny, so convergence fired
  prematurely and left a thin water layer on non-valley surfaces.
- Fix: `Physics::converged(dt)` normalizes residual to a per-second rate
  (`residual / dt < 3.125`, which equals 0.05 per 16ms frame). main.rs uses
  `self.physics.converged(dt)`.
- Regression test `convergence_is_frame_rate_independent` runs the same rain
  at dt = 0.016 and dt = 0.001 and asserts the identical equilibrium.

## Files
- `src/main.rs` — App, main loop, motion, convergence check
- `src/physics.rs` — particles, water, `converged(dt)`, cell sleep optimization
- `src/water.rs` — WaterSimulation (flow / drain / evaporation, all dt-scaled)
- `src/render.rs` / `src/view.rs` — rendering (rayon-parallel projection and bounds)

## Pending
- The Windows compatibility fix (keyboard-enhancement push/pop made non-fatal)
  is UNCOMMITTED in `src/main.rs` — commit it.

## Verify
```bash
cargo test --offline   # 41 tests pass; build has no warnings
```

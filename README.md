# trw

A 3D **Trapping Rain Water** visualization demo that runs in your terminal.

[![Crates.io](https://img.shields.io/crates/v/trw)](https://crates.io/crates/trw)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/lenitain/trw/actions/workflows/ci.yml/badge.svg)](https://github.com/lenitain/trw/actions/workflows/ci.yml)

`trw` generates a random-height terrain grid, rains on it, and lets the
water fall, flow downhill and pool in hollows. Everything is rendered as
braille wireframe in a standard terminal — no GPU required. When the rain
stops, the simulation converges to the exact Trapping Rain Water
equilibrium.

## Features

- **3D terrain + water** rendered with braille characters.
- **Particle-based rain**: drops fall, flow and stack.
- **Deterministic convergence**: the settled result matches the Trapping
  Rain Water algorithm exactly.
- **Interactive camera**: rotate, move, spin and zoom with the keyboard.
- **X-Ray mode**: see water hidden behind the terrain.

## Usage

### Install from crates.io

```bash
cargo install trw
```

### Build from source

Requires Rust toolchain.

```bash
git clone https://github.com/lenitain/trw.git
cd trw
cargo run --release
```

Optional: set the grid size with `--grid-size` (default 8). Terrain column
heights scale with the grid size (max height = grid size − 1), so a larger
grid also gets taller mountains.

```bash
cargo run --release -- --grid-size 12
```

### Keys

| Key                 | Action                        |
| :------------------ | :---------------------------- |
| `g`                 | Generate a new random terrain |
| `w`                 | Toggle rain                   |
| `p`                 | Pause / resume physics        |
| `c`                 | Clear water and particles     |
| `x`                 | Toggle X-Ray mode             |
| `Space`             | Auto-spin                     |
| `Tab`               | Toggle axes                   |
| `←` `→` `↑` `↓`     | Rotate (yaw / pitch)          |
| `r` / `e`           | Roll                          |
| `Shift` + arrows    | Move the view                 |
| `=` / `-`           | Nearer / farther              |
| `f` / `Shift` + `f` | Center / fit                  |
| `0`                 | Reset the view                |
| `?`                 | Show key help                 |
| `q` / `Esc`         | Quit                          |

## License

MIT — see [LICENSE](LICENSE).

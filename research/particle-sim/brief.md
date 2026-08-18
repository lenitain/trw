# Research Brief: Efficient Particle Simulation for TUI Renderer

## Question
What is the most efficient particle simulation approach for a TUI-based 3D Trapping Rain Water demo that needs to simulate 100-500 particles with flowing, stacking, and settling behavior at 30+ FPS on CPU-only hardware?

## Context
- Project: `trw` — Rust TUI app using ratatui + ratatui-wireframe for 3D wireframe rendering via braille characters
- Resolution: ~120x30 characters (240x120 braille dots)
- Current state: Particles only fall vertically, no horizontal movement, no particle-particle interaction, no flowing
- Grid: 8x8 terrain height matrix (expandable)
- Already using `rayon` for parallelism
- Each particle is rendered as a wireframe cube (8 vertices, 12 edges) — vertex count is the rendering bottleneck

## Requirements
1. Particles must flow into valleys (not just fall straight down)
2. Particles must stack on top of each other
3. Particles must flow out when container is inverted
4. Physically accurate simulation NOT required — visual plausibility suffices
5. Implementation must fit in a few hundred lines of Rust
6. Must run at 30+ FPS on modern CPU

## Scope boundaries
- IN: CPU-only particle simulation methods, optimization tricks, 2D-grid-based approaches, simplified physics
- OUT: GPU compute, CUDA/OpenCL, full SPH fluid simulation, ML-based approaches
- Language: Rust (no external physics libraries — must be self-contained)

## Assumptions
- We can sacrifice physical accuracy for visual quality
- The terrain grid is small (8x8 to maybe 16x16)
- Particle count is modest (100-500)
- The rendering bottleneck is vertex count (each particle = 8 vertices for a cube)
- We can use simplified 2D approaches projected to 3D

## Depth: standard
## Date: 2026-08-18

## Angles

1. **Position-Based Dynamics (PBD) for granular/particle simulation** — How PBD works for particles, its O(n) or O(n log n) characteristics, suitability for small particle counts, implementation complexity
2. **Grid-based cellular automata for water/fluid** — Cellular automata water simulation, falling sand games, how they handle flow and stacking, performance characteristics
3. **Simplified 2D fluid projected to 3D** — 2D height-field water simulation (like shallow water equations), how to project to 3D for visual effect, common in games
4. **Spatial hashing and neighbor search optimization** — Uniform grid, spatial hashing, how to make O(n²) collision detection tractable for 100-500 particles
5. **Game industry tricks for real-time particle water** — How games like Minecraft, Noita, Powder Game, falling sand simulations handle water particles efficiently on CPU

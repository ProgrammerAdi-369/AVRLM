# CLAUDE.md

Instructions for the Claude Code agent working on this repository. Read this
file in full before writing any code.

---

## 1. Project context (read-only background, for orientation)

This repo is one module of a 5-6 person hackathon project: **Adaptive
Variable Resolution 2.5D LiDAR Mapping** for a UGV (unmanned ground vehicle).
The full pipeline is:

```
Raw LiDAR points -> Spiking PointNet++ (Member 1) -> semantic labels + spike
counts -> THIS MODULE: variable-resolution 2.5D grid engine (Member 3) ->
Streamlit dashboard (Member 4)
```

Member 5 also consumes memory/latency metrics from this module for their
MAC-vs-AC benchmarking story.

**You do not need to build or understand the SNN model, the dashboard, or
the benchmarking scripts.** They are other people's modules with their own
repos/branches. Do not modify files outside this module's scope unless
explicitly asked. If a task seems to require touching another module's code,
stop and say so instead of doing it.

## 2. This module's job

Build the **Variable Resolution 2.5D Grid & Quadtree Engine**:

1. Radial filter: split points into a 0-10m zone (5cm cells) and a 10-100m
   zone (50cm cells).
2. A hierarchical grid structure that nests the 5cm cells exactly inside the
   50cm cells, with zero seam/alignment error at the 10m boundary.
3. Per-cell elevation (max height, height variance) and majority semantic
   class.
4. Event-driven updates: a cell only refreshes when it received points with
   a nonzero spike count this frame; otherwise it keeps its cached value.
5. A handoff function `generate_2_5d_grid(points, labels, spikes)` that
   returns the structured grid, plus a small memory-metrics helper for
   Member 5.

## 3. Confirmed interface contract — TREAT AS FIXED, DO NOT ALTER

This was negotiated with Member 1 and Member 2 and is a hard external
contract, not a design choice you get to revisit. If you think a value here
should change, stop and ask — never silently "improve" or reinterpret it.

| Field | Shape | dtype | Notes |
|---|---|---|---|
| `points` | `(N, 4)` | `float32` | X, Y, Z, Intensity |
| `labels` | `(N,)` | `int64` (or `uint8` in NumPy to save RAM) | class ID per point |
| `spikes` | `(N,)` | `uint8` or `int32` | **spike COUNT per point this frame, not a 0/1 flag** — event-driven logic must check `> 0`, not `== 1` |

- **Coordinate frame**: strictly ego-centric. UGV is always at `(0, 0, 0)`.
  Never assume or convert to a global/world frame — the whole radial
  10m/100m zoning logic depends on this.
- **Units**: meters, everywhere. No unit conversion needed anywhere in this
  module.
- **Class ID mapping** (fixed, do not renumber):
  `0` = drivable terrain, `1` = static obstacle, `2` = dynamic object.
- Intensity (`points[:, 3]`) is not used by this module — drop it after
  ingestion. Z is used only to compute per-cell elevation, not carried
  forward per-point into the output grid.
- **Z must never be NaN/Inf.** Member 1 still owns not producing NaN Z
  upstream, but this module now guards it defensively too:
  `aggregate_cells()` checks `points[:, 2]` for NaN before computing
  elevation stats and raises `ValueError` (naming the offending point
  indices) rather than silently propagating NaN into `elevation_max`/
  `elevation_var` (see Reports/AUDIT-v2.md §3.4 and its Phase 2 fix). Inf
  is not currently checked separately.

## 4. Design decisions already made — do not re-derive or silently change

These were worked out deliberately. Implement them as described; if you
think a different approach is better, say so explicitly and ask, per the
working principles below — don't just implement your own idea instead.

- **Grid structure is NOT a textbook 4-way (2x2) recursive quadtree.**
  Halving 50cm repeatedly never lands exactly on 5cm (10 is not a power of
  2). Instead: a base 50cm grid over the full bounding box, and any 50cm
  cell that falls inside the 10m radius is subdivided into a fixed **10x10
  array of 5cm sub-cells**. This is the primary, performance-critical
  implementation.
- **Alignment guarantee comes from shared origin, not clever rounding.**
  Sub-cell index must always be computed as: (1) find the parent 50cm cell
  index from the single shared grid origin, (2) compute the point's local
  offset *within that specific parent cell* (always in `[0, 0.5)`), (3)
  floor-divide that local offset by 0.05 to get the 0-9 sub-index. Never
  compute the fine grid's position independently from a separate origin —
  that reintroduces the seam bug this whole design exists to prevent.
- **10m boundary membership is decided per parent cell**, not per point
  (e.g. by the 50cm cell's center distance), so a single cell is never
  half-subdivided.
- A secondary, literal recursive `QuadNode` class (true 4-way subdivision)
  is an optional bonus/demo artifact for judge questions — only build it if
  explicitly asked. It is not the runtime path.
- Elevation and class aggregation must be vectorized (NumPy grouping /
  Numba), never a Python `for point in points` loop — this needs to run at
  real-time frame rates on hundreds of thousands of points.
- Output format: prefer a sparse structure (active cells only, with cell
  coords, resolution tag, elevation, height variance, class ID) over a
  dense array — this is what makes the memory-savings story to Member 5
  and the dashboard honest.

## 5. Tech stack

NumPy, SciPy (spatial/KDTree if genuinely needed — likely not, since this
is grid binning, not nearest-neighbor search), Numba for JIT-accelerating
the aggregation step. Don't introduce other heavy dependencies (pandas,
pytorch, etc.) into this module without asking — it should stay lightweight
and fast.

## 6. Testing

Use `synthetic_lidar_data.py` for all standalone testing — do not wait for
real model output or a real dataset. In particular:

- Always test against `generate_boundary_stress_ring()` first — a dense
  ring of points at exactly r=10m. It should render as flat terrain with no
  phantom step or gap. This is the single most important correctness check
  in this module.
- Use `build_moving_sequence()` to verify event-driven behavior across
  frames: after frame 0, cells not under the moving cluster must not
  refresh (their spike sum should be 0 on later frames).
- Turn every bug fix or new function into a runnable check (a quick script
  or assertion), not just a manual read-through. See "Goal-Driven
  Execution" below.

---

## 7. General working principles

### Think before coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Simplicity first

Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes,
simplify.

### Surgical changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### Goal-driven execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it
work") require constant clarification.

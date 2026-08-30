# Pipeline Audit Report — 2026-08-28

## 1. Summary

The core grid/aggregation logic is sound: the boundary-authority design (per-parent-cell-center
subdivision, shared-origin sub-cell alignment) is implemented exactly as specified, the boundary
stress ring shows no seam/gap, event-driven caching correctly gates on `spike_sum > 0` (never a
truthy flag), and the previously-reported "three-number mismatch" and "22,248 vs ~21,000 gap" are
now fully reconciled — both were measurement artifacts, not logic bugs (see §3). The single most
important finding this pass: **the real-time latency budget (10fps / 100ms per frame) is not just
occasionally breached but breached on every single frame tested**, including the fixed
`test_integration.py` 10-frame run (114–214ms/frame) and long-horizon runs — this module is
roughly 1.5–2x too slow for its own stated target at current (non-Numba) scene sizes, before even
reaching the 250k/2.5M-point stress scales. Two correctness bugs were also found that CLAUDE.md
explicitly asked to defend against: an out-of-range class ID crashes `aggregate_cells` instead of
degrading gracefully, and a `NaN` in a point's Z coordinate is not filtered by `assign_cells`'
`in_range` check (which only bounds X/Y) and silently corrupts that cell's elevation stats to
`NaN`.

## 2. Interface contract compliance

| Contract item | Enforced? | Evidence/test | Notes |
|---|---|---|---|
| `points` shape `(N,4)` | **N** | Fed `assign_cells()` an `(N,3)` array — no exception, ran silently (indexing `points[:,1]`/`[:,0]` still "worked" against the wrong columns) | Not asserted anywhere in `grid.py`, `aggregate.py`, or `grid_state.py`. Non-blocking but should fail loudly. |
| `points` dtype `float32` | **N** | Fed `assign_cells()` a `float64` `(N,4)` array — no exception, ran to completion | Never asserted; the code is dtype-agnostic in practice (NumPy upcasts happily), so this doesn't corrupt output, but it's silently trusted rather than checked. |
| `labels` dtype `int64`/`uint8` | **N** | Same as above — `aggregate.py` explicitly does `labels[mask].astype(np.int64)` itself, so a wrong input dtype is silently coerced, not validated | Coercion masks the absence of a real check. |
| `spikes` is a **count**, not a flag | **Y** | Grepped every `spikes`/`spike_sum` use across all 6 source files (`grid_state.py:62`, `aggregate.py:82`). Only gating condition found is `spike_sum > 0` (a summed float, from `np.bincount(..., weights=spk)`). Hand-test: two points with counts `[5, 3]` in the same cell → `spike_sum = 8.0`, not `2` or `1`. No `if spikes:` truthy usage found anywhere. | Clean — the exact bug class flagged in CLAUDE.md as "watch for siblings of" does not appear elsewhere. |
| Ego-centric frame, origin `(0,0,0)` | **Y** | Grepped for hardcoded offsets — none found. `parent_cell_center`/`sub_cell_center` in `grid.py` derive positions purely from `OUTER_RADIUS`/`OUTER_RES`/`INNER_RES`, no translation constant. Origin point `(0,0,0)` test: lands in `parent=(200,200)`, `is_fine=True`, `sub=(0,0)` — the exact center sub-cell, as expected for an ego-centric grid centered on the vehicle. | |
| Class IDs `0`/`1`/`2` | **N** (crashes on bad input) | Hand-test: fed `aggregate_cells` a label array containing `99` (with `NUM_CLASSES=3` hardcoded in `aggregate.py`). Result: `ValueError: cannot reshape array of size 100 into shape (1,3)` — the `class_flat_counts.reshape(n_cells, NUM_CLASSES)` step blows up because `np.bincount(group_id * NUM_CLASSES + lbl, ...)` produces a longer array than `NUM_CLASSES` accounts for. | **Blocker-adjacent** — CLAUDE.md explicitly asked for this to not crash on a bad upstream value; it does crash, ungracefully (opaque reshape error, not even a clear validation message). |
| Intensity dropped after ingestion | **Y** | `grep -n "intensity\|\[:, ?3\]" *.py` — the only hits for `points[:, 3]`-equivalent are inside `synthetic_lidar_data.py`'s own point *construction* (`np.stack([x, y, z, intensity], ...)`). No consumer file (`grid.py`, `aggregate.py`, `grid_state.py`, `handoff.py`) ever reads column 3. | Clean. |

## 3. Reconciled discrepancies (from prior sessions)

### Task 0 status — **resolved**
Single documented authority now exists: `grid.py`'s per-parent-cell-center rule. Quoting
`radial_filter.py:20-23`: *"NON-AUTHORITATIVE: this per-point split is a diagnostic/reporting
utility only. Which resolution a point's data is actually stored at is decided by grid.py's
per-parent-cell-center rule ... use prefilter_mask() below to gate what reaches grid.py."*
`grid_state.py:51` calls `prefilter_mask()` before `assign_cells()`, confirming the pipeline
actually follows this contract, not just the docstring. **Resolved, not just documented — verified
wired up.**

### `r > 100` out-of-range test — **exists and passes**
`test_grid.py:62-80` feeds `assign_cells()` a point at `r=150` directly and asserts `in_range=False`,
all index fields `-1`, and exclusion from the unique-parent-cell count. Re-ran it plus additional
sweep points at `r=100.0`, `r=100.0001`, `r=1000`: `r=100.0` → `in_range=True` (inclusive boundary,
matches `prefilter_mask`'s `r <= 100.0`); `r=100.0001` and `r=1000` → both `in_range=False`,
handled identically (no special-casing "just past" vs "far out"). **Confirmed working.**

### The three-number mismatch (21,066 / 20,765 / 20,780) — **resolved, was a measurement artifact, not a logic bug**
Traced what each number actually is:
- `21,066` = `test_handoff.py`'s `generate_2_5d_grid(build_scene())` cell count.
- `20,765` = `test_grid_state.py` frame-0 `touched` count.
- `20,780` = `test_grid_state.py`'s own independent "true ungated" recomputation via `aggregate_cells`.

All three are supposed to measure the *same* quantity (occupied-cell count for one `build_scene()`
frame) and, when run **inside one consistent process** where `build_scene()` is called only once,
they are in fact bit-for-bit identical — verified: `aggregate_cells` row count, `GridState`
frame-0 `touched` count, and a fresh `generate_2_5d_grid()` call all returned **21066 / 21066 /
21066** in the same run. The reason the three *test files* report slightly different numbers when
run separately is **not a pipeline bug**: `synthetic_lidar_data.py` uses a single **module-level**
RNG (`RNG = np.random.default_rng(42)`) shared across every generator function. Each test script
calls a different sequence of generators before its own `build_scene()` call (e.g. `test_grid.py`
calls `generate_boundary_stress_ring()` first; `test_grid_state.py` runs a whole 10-frame
`build_moving_sequence()` before its regression check's `build_scene()` call). That consumes the
shared RNG differently per script, so **each script's "the same scene" is actually a different
random draw** — confirmed directly: two successive `build_scene()` calls in one process, back to
back, produced non-identical point clouds (`np.allclose` → `False`) despite both nominally being
"`build_scene()` with seed 42." This is a **non-blocking but real reproducibility gap** in the test
harness, not in the production module — see §8.

### The 22,248 vs ~21,000 gap — **resolved, explained exactly, not a bug**
Traced one parent cell by hand through both code paths. `test_grid.py`'s reported "22,248" is
`len(unique parent cells touched) + len(unique sub-cells touched)`, where the parent-cell set is
built from **every** point regardless of whether its parent cell was subdivided. That means a
50cm parent cell that *was* subdivided gets counted once in the parent-cell set (even though no
data is actually stored at that resolution — its points live in the sub-cell set instead) **and**
its sub-cells get counted again in the sub-cell set — double-counting every subdivided parent.
Reproduced with fresh data in one run: `19,584` parent cells touched (incl. subdivided) + `2,373`
sub-cells touched = `21,957` (test_grid.py-style sum); of those 19,584 parent cells, `891` were
actually subdivided (and thus double-counted); true occupied count = `(19,584 − 891) + 2,373 =
21,066`, which matches `aggregate_cells`'s real row count (`21,066`) exactly. **The gap is 100%
explained by this double-counting artifact in the diagnostic print in `test_grid.py` — no cells
are actually being lost anywhere in the real pipeline.**

### Latency budget breach — **confirmed, reproduces on every frame, worse than previously characterized**
Re-ran `test_integration.py` fresh: all 10 frames breach the 100ms/10fps budget, range
**114.17ms – 214.37ms**, worst case frame 7 at 214.37ms (~4.7fps). This is not an occasional
breach — it is 10/10 frames, on a scene of only ~21,000-25,000 raw points. See §7 for the full
timing table.

### Unbounded cache growth (114,910 cells after 11 calls) — **confirmed to reproduce, and confirmed to be a synthetic-data artifact via a dedicated control experiment**
Reproduced on a fresh `GridState`: after 10 frames of `build_moving_sequence()`, cache =
`114,910` cells (matches prior report). Extended to 100 frames: growth continues to **219,865**
cells by frame 100, but the *rate* decelerates sharply — average cells added per frame across
checkpoint windows: `11,112` (frames 1-5) → `7,419` (5-10) → `3,612` (10-20) → `1,382` (20-50) →
`573` (50-100). So growth is **decelerating (superlinear-to-sublinear transition), not plateaued,
by frame 100**.

To isolate cause, ran the Phase-4-specified control: identical ground-plane points fed every
frame (no resampling) plus only the moving cluster changing position. Result: cache grew from
`21,105` (frame 1) to only `21,805` (frame 100) — **~700 new cells over 99 frames**, tracking the
cluster's own path almost exactly (58 cells/frame × ~12 net-new positions worth of overlap).
**This confirms the hypothesis from the prior session: the bulk of the unbounded growth in the
original `build_moving_sequence()`-based test is a synthetic-data artifact** — `generate_ground_plane()`
draws a *fresh* random point cloud every frame, so slightly different ground points touch
slightly different fine 5cm cells near the origin each frame, permanently growing the cache even
though the physical ground hasn't moved. See §8 for the residual real risk this doesn't rule out.

## 4. Per-cell correctness findings

**Coarse cell, hand-built ground truth** (4 points, one 50cm cell, `r>10` so no subdivision):
`elevation_max = 3.0` exactly (correct). `class_id = 0` (correct, 3-of-4 majority).

**Height variance formula** (doubt-rule test): 3 points with `z ∈ {1.0, 3.0, 2.0, 2.0}` →
hand-computed population variance (`ddof=0`) = `0.5`, sample variance (`ddof=1`) = `0.6667`.
Code (`aggregate.py`'s `sum_z2/counts - mean_z**2`) returned `0.5`. **Confirmed: population
variance, not sample variance.** This should be stated explicitly in any docs/dashboard tooltip
that surfaces "height variance," since it's currently implicit in the formula only.

**Majority-class tie-break** (doubt-rule test, 2-vs-2 split with labels `{0,0,1,1}` in one cell):
result was `class_id = 0` — the lower class ID. This is an artifact of `np.argmax` returning the
*first* maximal index when counts tie; it is not a deliberately chosen tie-break rule anywhere in
the code or docs. **Empirically confirmed and now documented here: ties resolve to the numerically
lowest class ID.** Worth deciding explicitly (e.g. "prefer higher-priority class on tie" might be
more sensible for obstacle-avoidance — a dynamic object tying with drivable terrain probably
shouldn't silently become "drivable").

**Subdivided (fine, `r<10`) cell**: 4 points inside one 5cm sub-cell all confirmed `is_fine=True`,
one cell produced, `elevation_max = 1.5` (max of the 4 z-values) — aggregation is identical at
fine resolution, no special-cased bug.

**Boundary-ring correctness**: sampled individual points near `r≈10.0` — confirms the design
behaves exactly as documented: membership is decided by the *parent cell's center*, not the
individual point's own radius. Concretely: a point at `r=9.987` fell into a **coarse** cell
(`parent=(183,211)`), while a different point also at `r=9.987` fell into a **fine (subdivided)**
cell (`parent=(181,206)`) — because those two points landed in different 50cm parent cells whose
*centers* are on opposite sides of the 10m radius. This is correct, intentional per-cell (not
per-point) behavior per CLAUDE.md §4, verified directly rather than assumed. A point placed at
exactly `r=10.0` on the +X axis landed in a **coarse** cell (`is_fine=False`) even though its own
radius is exactly at the nominal boundary — again correct, since its parent cell's center distance
is what's actually tested, and > 10.0 in this case.

## 5. Stress test results

| Scenario | Input size | Result | Time | Notes |
|---|---|---|---|---|
| Empty input | 0 points | Graceful | <1ms | Returns `touched={}`, `_cells` stays empty. No crash. |
| Single point | 1 point | Graceful | <1ms | Touched=1, cell created normally. |
| All points identical | 10 identical points | Graceful, correct | <1ms | `elevation_var = 0.0` exactly (no float noise). |
| Point at origin `(0,0,0)` | 1 point | Graceful, correct | <1ms | Lands in the exact center sub-cell `(200,200,0,0)`, `is_fine=True`. |
| NaN/Inf in X or Y | 3 points (NaN x, Inf y, NaN z) | **Silent-wrong (partial)** | <1ms | NaN-in-X and Inf-in-Y points are silently dropped (`in_range=False` via `abs(nan)<=100` evaluating False) — arguably acceptable, but undocumented. **The NaN-in-Z point is NOT filtered** (`in_range` only checks X/Y bounds) and passes straight through to aggregation, producing `elevation_max = NaN` for its cell — silent corruption of that cell's stats, not a crash and not a rejection. |
| Negative Z | 2 points, z=-3.0/-1.0 | Graceful, correct | <1ms | `elevation_max = -1.0` (the greater/less-negative value) — correctly not clipped to 0. |
| `r=10.0` exact | 1 point | Graceful, correct | <1ms | See §4 boundary discussion. |
| `r=100.0` exact | 1 point | Graceful, correct | <1ms | `in_range=True` — inclusive boundary. |
| `r=100.0001` | 1 point | Graceful, correct | <1ms | `in_range=False`. |
| `r=1000` (far out) | 1 point | Graceful, correct | <1ms | `in_range=False` — **identical handling** to the just-past-boundary case, as required. |
| Spikes at 0, 1, 255 (uint8 max) | small hand test | Graceful, correct | <1ms | No special-case bugs. |
| Spike count sum overflow (255+10 in one cell) | 2 points | Graceful, correct | <1ms | `aggregate.py` casts `spikes` to `float64` **before** summing (`spk = spikes[mask].astype(np.float64)`), so `spike_sum = 265.0` — no `uint8` wraparound. The *raw per-point* dtype is still capped at 255 by the interface contract itself (not this module's choice), but the module's own aggregation does not introduce an additional overflow bug. |
| Duplicate points, 2 frames | 1 point x2 frames | Graceful, correct | <1ms | Cache size stays 1 after both frames — cleanly overwritten, not duplicated or averaged. |
| Unknown class ID (99) | 3 points | **Crash** | n/a | `ValueError: cannot reshape array of size 100 into shape (1,3)` inside `aggregate_cells`'s `class_flat_counts.reshape(n_cells, NUM_CLASSES)`. **Not graceful** — CLAUDE.md explicitly asked for this not to crash. |
| `(N,3)` points (wrong shape) | 5 points | **Silent-wrong** | <1ms | No exception; ran to completion misinterpreting whatever the 3rd column happened to be. |
| `float64` points (wrong dtype) | 5 points | Silent, non-corrupting | <1ms | Ran fine — NumPy upcasts transparently, so this particular dtype deviation doesn't corrupt output, but it's still unvalidated. |
| 10x scale (~250,000 pts, uniform 0-100m) | 250,000 | Graceful, no crash | **1,069 ms** | 117,584 cells touched. ~10.7x slower than the 100ms budget at only 10x the nominal point count. |
| 100x scale (~2,500,000 pts) | 2,500,000 | Graceful, no crash/OOM | **10,413.5 ms** | 223,247 cells touched. Scales roughly linearly with point count (1,069ms→10,414ms for 10x more points), which is good news for the *algorithm*, bad news for the *budget* — confirms this is a throughput problem, not a pathological blowup. |
| All points crammed `r<10` (worst-case density) | 250,000, all inner | Graceful, no crash | **944.1 ms** | 96,649 cells touched — comparable timing to the uniformly-spread 250k case (1,069ms), so the fine-grid subdivision path isn't a disproportionate cost driver relative to raw point count. |
| Long-horizon: 100 frames, resampled ground | ~21-25k pts/frame x 100 | Graceful, no crash | see §7 | Cache: 20,995 → 219,865 cells (decelerating growth, not yet plateaued). Per-frame time stayed in the 80-146ms range throughout — still over budget at every checkpoint. |
| Long-horizon control: fixed ground + moving cluster only | ~21-25k pts/frame x 100 | Graceful, no crash | not separately timed | Cache: 21,105 → 21,805 cells — near-flat, confirming resampling (not a leak) drives the unconstrained-growth number above. |

## 6. Checked because uncertain

- **Q: Does a point exactly at `r=10.0` land in the inner or outer grid?**
  Test: single point at `(10.0, 0, 0)` fed to `assign_cells()`. **A: coarse (outer)** —
  `is_fine=False`, because the authoritative rule is the parent cell's *center* distance, and this
  point's parent cell center happened to be > 10.0m out. Matches documented design.
- **Q: Is height variance population or sample variance?**
  Test: 3 known z-values, hand-computed both formulas (`0.5` vs `0.6667`), compared to code output.
  **A: population variance (`ddof=0`)**, confirmed exactly.
- **Q: What does the majority-class aggregation do on an exact 2-vs-2 tie?**
  Test: 4 points, labels `{0,0,1,1}`, same cell. **A: resolves to the lower class ID** (an
  `np.argmax` first-max artifact, not a deliberate rule) — see §4.
- **Q: Does `spikes` ever get treated as a boolean flag (`if spikes:`) anywhere, echoing the
  previously-fixed cold-start bug?**
  Test: grepped every `spike`/`spike_sum` reference across all 6 source files and hand-tested
  `spike_sum` for counts `[5,3]` in one cell. **A: no — only gating condition anywhere is
  `spike_sum > 0`, and it correctly sums counts (`8.0`, not `2`), not a truthy check.** Clean.
- **Q: Does an out-of-range/unknown class ID (e.g. `99`) crash or degrade gracefully, per
  CLAUDE.md's explicit ask?**
  Test: label array containing `99` fed to `aggregate_cells`. **A: it crashes** with an opaque
  `ValueError` on a reshape, not a clear validation error and not graceful degradation. See §2, §5.
- **Q: Is the unbounded cache growth reported previously a real leak, or a synthetic-data
  resampling artifact?**
  Test: 100-frame run with a genuinely fixed (non-resampled) ground plane vs. the original
  freshly-resampled-every-frame version, side by side. **A: overwhelmingly a resampling artifact**
  — fixed-ground control grew ~700 cells over 99 frames vs. ~199,000 for the resampled version.
  See §3, §8 for the residual caveat.
- **Q: Does `NaN`/`Inf` in a point's coordinates get rejected, or does it corrupt a cell silently?**
  Test: 3 points with NaN-x, Inf-y, and NaN-z respectively, fed through `assign_cells` +
  `aggregate_cells`. **A: mixed** — NaN-x and Inf-y are incidentally dropped by the `abs(x)<=100`
  in-range check; NaN-z is **not** checked at all (the check only inspects X/Y) and corrupts
  `elevation_max`/`elevation_var` to `NaN` for its cell. See §5.
- **Q: Are `r=100.0001` (just past boundary) and `r=1000` (far out) handled the same way, or
  differently?**
  Test: both fed to `assign_cells()`/`prefilter_mask()`. **A: identically** — both get
  `in_range=False`, no special-casing. Confirmed, not assumed.

## 7. Timing & memory summary

| Scenario | Point count | Time | Meets 100ms/10fps budget? |
|---|---|---|---|
| `test_integration.py`, 10 frames of `build_moving_sequence()` | ~21-25k/frame | 114.17 – 214.37 ms (all 10 frames) | **No — 0/10 frames** |
| Long-horizon 100-frame run, checkpoint frames | ~21-25k/frame | 80.29 – 145.88 ms (checkpoints 1,5,10,20,50,100) | **No — 0/6 checkpoints** |
| 10x scale, single frame | 250,000 | 1,069.0 ms | No (10.7x over budget) |
| 100x scale, single frame | 2,500,000 | 10,413.5 ms | No (104x over budget) |
| All-inner (`r<10`) worst-case density, single frame | 250,000 | 944.1 ms | No (9.4x over budget) |

| Memory metric (from `memory_metrics()`, `build_scene()`) | Value |
|---|---|
| Active cell count | 21,066 |
| Estimated sparse bytes | 1,032,234 (~1.0 MB) |
| Naive dense-grid bytes (5cm, 200x200m, 3m Z band) | 960,000,000 (~960 MB) |
| Savings ratio | ~930x |
| After 2nd accumulated call (build_scene + 1 moving-sequence frame) | 38,661 cells, ~1.9MB, ~507x savings |

| Cache growth (100-frame long-horizon) | Cells cached |
|---|---|
| Frame 1 | 20,995 |
| Frame 5 | 76,553 |
| Frame 10 | 113,648 |
| Frame 20 | 149,764 |
| Frame 50 | 191,223 |
| Frame 100 | 219,865 |
| **Fixed-ground control, frame 100** | **21,805** (vs. 219,865 for resampled — isolates the artifact) |

## 8. Open blockers vs. non-blocking findings

**Blockers (should be resolved before demo/integration):**
1. **Real-time budget is not met at all — 0% of tested frames, at any scale, stayed under
   100ms.** Even the nominal ~21-25k-point moving-sequence scene (the scene size this was actually
   designed around) takes 80-214ms/frame. This needs either the Numba JIT pass CLAUDE.md already
   anticipated as a stretch task, or a scope/target-fps conversation with the team, before this
   module can be called real-time-ready.
2. **Unknown/out-of-range class ID crashes `aggregate_cells`** with an opaque `ValueError`
   instead of the graceful handling CLAUDE.md explicitly asked for. A single bad upstream label
   from Member 1's model would currently take down the whole grid update for that frame.
3. **NaN in a point's Z coordinate silently corrupts a cell's elevation stats to NaN**, undetected,
   because `assign_cells`'s `in_range` check only bounds X/Y, never Z. Real sensor data producing
   even one NaN Z value (not implausible) poisons that cell's `elevation_max`/`elevation_var`
   downstream, with no error or warning anywhere in the pipeline.

**Non-blocking (worth knowing, not urgent):**
1. No shape/dtype validation on `points`/`labels` — wrong-shape or wrong-dtype input is silently
   accepted rather than failing loudly (wrong-shape input actively misbehaves rather than raising).
2. `radial_filter.py` hardcodes its own `10.0`/`100.0` literals instead of importing
   `INNER_RADIUS`/`OUTER_RADIUS` from `grid.py`. Currently in sync, but a drift risk if either
   constant is ever changed in only one file.
3. The 2-vs-2 majority-class tie-break (lowest class ID wins) is an unintentional `np.argmax`
   artifact, not a documented design decision — worth a deliberate call, especially since ties
   currently favor "drivable terrain" (class 0) over "dynamic object" (class 2), which seems like
   the wrong direction to default toward for obstacle safety.
4. `synthetic_lidar_data.py`'s shared module-level RNG means `build_scene()` is **not**
   deterministic across repeated calls within one process — two back-to-back calls in the same
   script produce different point clouds despite the fixed seed. This is why prior sessions'
   "same scene" cell counts differed slightly across different test files; it's a test-harness
   reproducibility gap, not a production bug, but it will keep producing confusing
   number-mismatches in future audits unless each generator call explicitly reseeds or accepts an
   independent RNG.
5. The unbounded-cache-growth concern is now well-explained as a synthetic-data artifact rather
   than a leak, **but** the underlying cache still has no eviction/expiry policy at all — a real
   deployment driving through genuinely novel terrain for an extended period (not resampling the
   same geometry, but actually covering new ground) would still grow the dict unboundedly over a
   long mission. The 100-frame test isn't long enough to distinguish "artifact-driven growth that
   will plateau once resampling noise is exhausted" from "will keep growing forever on real novel
   terrain" — worth a longer/real-trajectory test before ruling this out entirely.

## 9. Recommended next actions

- **Decide the real-time budget question**: is 100ms/frame (10fps) still the actual target given
  current numbers, or should Numba JIT tuning (CLAUDE.md's flagged stretch task) be pulled forward
  from "optional" to "required," or should the target itself be renegotiated with the team given
  hackathon time constraints?
- **Decide how out-of-range class IDs should be handled**: clamp/ignore unexpected labels, raise a
  clear validation error at the boundary instead of an opaque reshape crash, or treat it as "should
  never happen, not worth guarding" — but currently it's an unguarded crash, which contradicts
  CLAUDE.md's explicit ask, so this needs an explicit answer either way.
- **Decide whether Z should be included in the `in_range`/validity check**, or whether NaN-guarding
  belongs further upstream (e.g. Member 1's model output should never produce NaN Z, so is a
  defensive check here even warranted?) — surfacing as a question rather than picking silently.
- **Decide whether the majority-class tie-break should be made explicit and safety-biased**
  (e.g. prefer higher class ID / prefer non-drivable on ties) rather than left as an `np.argmax`
  side effect.
- **Decide whether `synthetic_lidar_data.py`'s shared RNG should be fixed** (e.g. each generator
  function takes/creates its own local RNG instance) so future regression numbers across different
  test scripts are actually comparable apples-to-apples.
- **Decide whether shape/dtype input validation is worth adding** at the `assign_cells()` or
  `generate_2_5d_grid()` boundary, given Member 1's model output is the only real caller and is
  presumably contract-compliant by construction — or whether this is deliberately left as "trust
  the upstream contract."

## 10. Round 2 — Fixes Implemented & Re-Test Results

All 8 decisions from `implement_fixes_and_retest_guide.md` were implemented. Two things were kept
explicitly out of scope per the guide's ground rules: a cache eviction/decay policy, and any NaN-Z
code guard (Member 1's responsibility, documented not enforced).

### Blocker 1: Numba JIT

**Profiling first (guide Step 1) contradicted the guide's assumed hot path.** `cProfile` on
`test_integration.py` (10 frames, ~21k pts/frame) before any change showed:

| Cost center | tottime (10 frames) | ~ms/frame |
|---|---|---|
| `GridState.update()`'s own per-cell Python loop | 0.990s | ~99ms |
| `np.unique(..., axis=0)` (parent-pair dedup in `grid.py` + cell-key dedup in `aggregate.py`) | 0.511s | ~51ms |
| `assign_cells()`'s vectorized index math | ~0.03s | ~3ms |
| `aggregate_cells()`'s bincount/variance math | ~0.06s | ~6ms |

The guide's Step 2 named `assign_cells()`'s index math and `aggregate.py`'s aggregation math as JIT
targets — together only ~9ms/frame, a small fraction of the ~162ms/frame average. Flagged to the
user before proceeding; decided to implement the guide's literal targets first (isolated, reported
alone), then apply a second, separately-flagged fix for the two real bottlenecks — both staying in
plain Python/NumPy, not JIT, so the guide's explicit "leave the dict cache un-JIT'd" instruction is
still honored.

**1a — guide's literal JIT** (`grid.py`): extracted the per-point parent-origin/local-offset/floor-
divide math into `@njit(cache=True) _sub_index_loop()`, doubt-rule verified bit-identical to the
plain-numpy version on 5000 random points (`scratchpad/doubt_jit_subindex.py`) before wiring into
`assign_cells()`. `aggregate.py`'s aggregation math was **not** JIT'd — the guide's own condition
("if it's implemented as an explicit loop rather than already-vectorized NumPy calls") does not
apply, since it's already `bincount`/`np.maximum.at`-vectorized with no explicit loop. Warm-up call
added at module import time in `grid.py` (documented inline). Isolated benchmark (`test_integration.py`,
10 frames): **93–249ms/frame** (mixed, still 7/10 frames over budget) — a small, inconsistent
improvement over baseline's 114–214ms, confirming the profiling: this alone wasn't the fix.

**1b — separately-flagged fix (the actual win)**: packed the multi-column cell keys
(`parent_ix, parent_iy[, sub_ix, sub_iy]`) into single `int64` arrays (`_pack_pair`/`_unpack_pair`
in `grid.py`, `_pack_cell_key`/`_unpack_cell_key` in `aggregate.py`) so `np.unique` runs on a 1D
array instead of `axis=0` on a 2D array; doubt-rule verified the packing round-trips exactly and
gives identical unique rows + inverse mapping vs. the old `axis=0` call on real `build_scene()` data
(`scratchpad/doubt_pack_unique.py` — 21,066 cells / 19,584 parent cells, matching this audit's own
§3 baseline exactly). Also replaced `GridState.update()`'s per-scalar `int()`/`float()` numpy-scalar
casts with one bulk `.tolist()` per array before the loop — the dict/`CellRecord` commit logic
itself is untouched plain Python, per the guide's instruction. Re-profiling after 1b:
`GridState.update()` tottime dropped from 0.990s → 0.301s (10 frames), `np.unique` cumtime dropped
from 0.511s → 0.039s (20→30 calls, the extra 10 being `validate_labels`'s cheap check).

**1c — correctness regression**: `test_grid.py`, `test_aggregate.py`, `test_grid_state.py`,
`test_handoff.py` all re-run after each sub-step; cell counts, elevation stats, and cache growth
numbers are bit-identical to pre-JIT baseline throughout (e.g. `build_scene()` → 21,066 cells,
10-frame cache → 114,910 — unchanged at every step).

**1d — verdict, final numbers (1a+1b combined) vs. audit baseline:**

| Scenario | Baseline | Round 2 | Meets 100ms budget? |
|---|---|---|---|
| 10-frame integration (`test_integration.py`) | 114.17–214.37 ms | **32–58 ms** (one outlier run hit 91.67ms, still <100ms) | **Yes — 10/10 frames** |
| 100-frame long-horizon checkpoints | 80.29–145.88 ms | **27.50–39.64 ms** | **Yes — 6/6 checkpoints** |
| 10x scale (250,000 pts) | 1,069.0 ms | **440.5 ms** (2.4x faster) | No — still 4.4x over |
| 100x scale (2,500,000 pts) | 10,413.5 ms | **2,314.5 ms** (4.5x faster) | No — still ~23x over |
| All-inner worst-case (250,000 pts, r<10) | 944.1 ms | **263.8 ms** (3.6x faster) | No — still 2.6x over |

**The primary target is met**: the nominal ~21–25k-point moving-sequence scene (the scene size this
module was actually designed around) now clears the 100ms/10fps budget on every tested frame,
roughly 3–5x faster than baseline. The 10x/100x/all-inner stress scales are all 2.4–4.5x faster but
remain over budget — reported plainly as a **residual finding**, not folded into "fixed" (see below).

### Blocker 2: Invalid class ID

`validate_labels(labels)` added to `aggregate.py`, called at the top of `aggregate_cells()` before
any bincount/reshape math. Raises `ValueError: Invalid class ID(s) [99] found in labels - expected
values in [0, 3).` instead of the old opaque `cannot reshape array of size 100 into shape (1,3)`.
New test in `test_aggregate.py` confirms the clear message and confirms all three valid IDs (0, 1,
2), individually and combined in one cell, do not trip the check.

### Blocker 3: NaN-Z contract

`CLAUDE.md` (this repo's copy, section 3) now states explicitly that Z must never be NaN/Inf,
enforced upstream by Member 1, not validated here, and that a violation silently corrupts that
cell's elevation stats. No code guard was added, per the decision. A new test in `test_aggregate.py`
(`NaN-Z documented behavior confirmed...`) feeds a NaN-Z point through and asserts the current
accepted behavior (`elevation_max` becomes NaN, no exception) — its docstring/comment states this
documents accepted behavior, not an open bug. **This contract line still needs to reach Member 1
outside this repo** — it was only added to the copy of `CLAUDE.md` inside this module's directory;
the separate root-level copy at `C:\Users\Aditya\Downloads\CLAUDE.md` was left untouched since it's
outside this module's scope.

### Non-blocking fixes 1-5

**1. Input shape/dtype validation**: `validate_inputs()` added to `handoff.py`, called at the top of
`generate_2_5d_grid()` (the true outermost entry point). `(N,3)` points and `float64` points now
raise clear `ValueError`s (tested in `test_handoff.py`); a full `build_scene()` run still passes
through unaffected.

**2. `radial_filter.py` constant drift**: now imports `INNER_RADIUS`/`OUTER_RADIUS` from `grid.py`
instead of hardcoding `10.0`/`100.0`. Re-ran `test_radial_filter.py` and `test_grid.py` — confirmed
a true no-op (identical inner/outer counts and boundary-ring numbers before and after).

**3. Safety-biased tie-break**: `aggregate.py`'s majority-class `argmax` now resolves ties toward
the higher class ID (reverse-then-argmax-then-unreverse). Doubt-rule tested directly against the
real `aggregate_cells()` on hand-built single-cell inputs (`scratchpad/doubt_tiebreak.py`):
`{0,0,1,1}` → 1 (was 0), `{0,0,2,2}` → 2 (was 0), and the non-tie case `{0,0,1,1,1}` → 1, confirming
real majorities are untouched by the change. Full regression suite re-run after the change with no
other numbers affected (exact ties are rare in the synthetic scenes).

**4. Shared-RNG reproducibility gap**: removed `synthetic_lidar_data.py`'s module-level `RNG`;
every generator function (`generate_ground_plane`, `generate_pole`, `generate_dynamic_cluster`,
`generate_boundary_stress_ring`, `build_scene`, `build_moving_sequence`) now takes `rng=None` and
creates its own `np.random.default_rng(42)` locally when not given one, while still threading a
passed-in `rng` through to sub-generators exactly as before. Verified: two back-to-back
`build_scene()` calls now produce bit-identical points (`np.allclose` → `True`, was `False`);
confirmed unaffected by unrelated generator calls run beforehand. As a direct consequence, this
audit's own §3 "three-number mismatch" is now resolved at the source: `test_grid_state.py`'s
cold-start regression check, which previously reported `20,780` due to RNG drift from
`build_moving_sequence()` consuming the shared RNG before its own `build_scene()` call, now reports
`21,066` — matching `test_handoff.py` and a fresh `generate_2_5d_grid()` call exactly.

**5. Longer growth-curve data**: extended `build_moving_sequence()` to 500 frames with a larger
cluster velocity `(0.14, 0.05)` (vs. the original `(0.4, 0.15)` over only 10 frames) so the cluster
actually traverses new ground rather than jittering near its start position. Checkpoints (cache
size / cells-added-per-frame in that window):

| Frame | Cache size | Cells added since prior checkpoint | Per-frame rate in window |
|---|---|---|---|
| 1 | 20,765 | — | — |
| 5 | 76,355 | 55,590 | 13,897.5 |
| 10 | 114,855 | 38,500 | 7,700.0 |
| 20 | 151,203 | 36,348 | 3,634.8 |
| 50 | 193,253 | 42,050 | 1,401.7 |
| 100 | 221,734 | 28,481 | 569.6 |
| 150 | 235,597 | 13,863 | 277.3 |
| 200 | 242,902 | 7,305 | 146.1 |
| 300 | 248,495 | 5,593 | 55.9 |
| 500 | 251,063 | 2,568 | 12.8 |

Growth continues to **decelerate sharply** and is much closer to flattening by frame 500 (12.8
cells/frame in the 300→500 window, down from 569.6 in the 50→100 window) than the original
100-frame data suggested. This is consistent with the round-1 diagnosis that most growth is a
resampling artifact of `generate_ground_plane()`'s fresh random draw each frame, bounded by that
generator's own point density near the origin — not evidence of an unbounded leak. **No eviction
policy was proposed or implemented**, per the explicit decision; this is data only. Per-frame time
stayed in the 25–50ms range throughout this run (post-Blocker-1 fix), comfortably under budget.

### Checked because uncertain (Round 2)

- **Q: Which functions are actually the hot path for Blocker 1 — the guide's assumed
  `assign_cells()`/`aggregate.py` math, or something else?**
  Test: `cProfile` on `test_integration.py` before making any change. **A: something else** —
  `GridState.update()`'s Python loop (~99ms/frame) and `np.unique(axis=0)` calls (~51ms/frame)
  dominate; the guide's named targets are only ~9ms/frame combined. Flagged to the user before
  proceeding (see above).
- **Q: Does numba support the exact per-point sub-index loop pattern (floor-division inside a
  per-point loop, `np.floor` on scalars, int64 casts) and does the JIT'd version match plain numpy
  bit-for-bit?**
  Test: 5000 random points through both versions. **A: yes, exact match** —
  `scratchpad/doubt_jit_subindex.py`.
- **Q: Does packing the 2-column and 4-column cell keys into single `int64` values round-trip
  exactly, and does `np.unique` on the packed 1D array give identical unique rows + inverse mapping
  to `np.unique(..., axis=0)` on real `build_scene()` data?**
  Test: full `build_scene()` scene through both code paths. **A: yes, exact match** — 21,066 cells
  (4-col) and 19,584 parent cells (2-col), both matching this audit's own §3 baseline numbers
  exactly — `scratchpad/doubt_pack_unique.py`.
- **Q: Does the reversed-argmax tie-break resolve ties toward the higher class ID without disturbing
  real (non-tied) majorities?**
  Test: `{0,0,1,1}`, `{0,0,2,2}`, `{0,0,1,1,1}` through the real `aggregate_cells()`. **A: yes** —
  1, 2, and 1 respectively, matching the guide's expected cases exactly —
  `scratchpad/doubt_tiebreak.py`.
- **Q: Is `build_scene()` now actually deterministic across repeated in-process calls, and unaffected
  by unrelated generator calls run beforehand?**
  Test: two back-to-back `build_scene()` calls, then a third after an intervening
  `generate_boundary_stress_ring()` call. **A: yes to both** — `scratchpad/doubt_rng.py`.

### Residual findings

- **Stress-scale scenarios remain over budget.** 10x (440.5ms), 100x (2,314.5ms), and all-inner
  worst-case (263.8ms) are all 2.4–4.5x faster than baseline but still 2.6x–23x over the 100ms
  budget. Not fixed in this pass — the guide's 8 decisions targeted the nominal scene size; further
  work here (e.g. JIT-ing more of `GridState`'s commit path, or a different cache data structure)
  would be a new decision, not covered by this pass.
- **No eviction/decay policy exists for the cell cache** — explicitly out of scope per the guide's
  ground rules. Phase 8's extended 500-frame data shows strong deceleration on resampled-but-
  otherwise-static terrain, but genuinely novel terrain over a long real mission remains untested
  and could still grow the dict unboundedly — this was already flagged in round 1 and remains open.
- **`GridState.update()`'s per-cell loop is still O(touched cells) in pure Python** even after the
  bulk-`.tolist()` fix — the dict-commit logic itself (key construction, `in` check, `CellRecord`
  construction) is inherent Python-dict overhead that can't be JIT'd without restructuring the cache
  away from a dict-of-objects, which the guide explicitly asked to leave alone this pass.
- **`test_handoff.py`'s "state should accumulate across calls" assertion is loose (`>=`)** and did
  not itself catch the Phase 7 RNG-determinism bug in round 1 — worth tightening to an exact
  expected count in a future pass, not changed here since it wasn't part of the 8 decisions.
- **`_pack_pair`/`_pack_cell_key`'s bit-packing assumes parent index magnitude stays under `2**20`**
  (true for the current 200m/0.5m grid, max index ~400) — this is documented inline but would
  silently need revisiting if `OUTER_RADIUS`/`OUTER_RES` ever changed enough to exceed that margin.
  Not a bug against the current fixed contract, just a noted assumption.

## 11. Round 3 — Dashboard & Full-Pipeline Audit — 2026-08-30

Scope: a fresh full-repo static pass (including files rounds 1-2 didn't cover in depth —
`dashboard_pro.py`, `dashboard_driving.py`, `driving_sequence.py`, `spiking_model.py`,
`integration_pipeline.py`, `profiler.py`), a re-run of the full regression suite, and — new this
round — actually launching both Streamlit dashboards (`streamlit run dashboard_pro.py` /
`dashboard_driving.py`) and driving them from a real Chrome browser via the `claude-in-chrome`
tools, watching server stdout/stderr and browser console throughout. No `snn_weights.pth`
checkpoint exists anywhere in the repo, so both dashboards ran against a freshly-initialized,
**untrained** `SpikingPointNet` — this shapes several findings below and is flagged explicitly
where relevant.

### 11.1 Regression suite re-run

All 6 test scripts re-run via `py <file>.py` from the repo root: **6/6 pass**, no regressions
since Round 2.

| Script | Result | Notable numbers |
|---|---|---|
| `test_grid.py` | pass | `build_scene()`: 24,700 pts, 19,584 parent cells, 2,373 sub-cells |
| `test_grid_state.py` | pass | 10-frame cache growth 20,765 → 114,910 (matches Round 2 exactly) |
| `test_aggregate.py` | pass | invalid class ID (99) correctly raises; NaN-Z documented behavior confirmed |
| `test_handoff.py` | pass | `generate_2_5d_grid(build_scene())` → 21,066 cells; validation errors correct |
| `test_integration.py` | pass | **24.64–55.33 ms/frame — 10/10 frames now under the 100ms budget**, even better than Round 2's post-fix 32–58ms range (normal run-to-run variance, same fix) |
| `test_radial_filter.py` | pass | boundary ring: inner=202, outer=198, r∈[9.9803, 10.0197] |

No new correctness regressions found in the core module. The real-time budget fix from Round 2
continues to hold.

### 11.2 Static findings (new files this round)

**11.2.1 — Model predictions are unseeded and swing wildly run-to-run without a checkpoint.**
`spiking_model.py`'s `SpikingPointNet.__init__` never calls `torch.manual_seed(...)`, and neither
dashboard seeds it before `load_model()`. With no `snn_weights.pth` present (the actual state of
this repo), each fresh Streamlit process gets a different random weight init, and because the
conv/LIF stack is completely untrained, its output collapses to a single dominant class rather
than a mix — confirmed empirically in §11.3 below (one process run classified ~98% of active
cells "Static Obstacle", a separate process run on the same code classified almost the entire
scene "Dynamic Threat" instead). This is expected behavior for an untrained net, not a module bug,
but it means **the dashboards are not demo-safe out of the box** — anyone cloning this repo and
running either dashboard without first running `synthetic_train_loop_v5.py` will see an
unpredictable, usually-nonsensical single-class scene, not a "less accurate but reasonable" one.
Both dashboards do show a clear sidebar warning ("No checkpoint found — run
synthetic_train_loop_v5.py"), so the failure mode is at least surfaced to the user, just not
prevented.

**11.2.2 — Both dashboards carry a dead `sys.path.append`.** `dashboard_pro.py:11` and
`dashboard_driving.py:11` both run
`sys.path.append(os.path.join(os.path.dirname(__file__), 'AVRLM'))` before importing sibling
modules. This pattern makes sense in `integration_pipeline.py` (which expects an `AVRLM`
subfolder), but both dashboard files already live inside the `AVRLM` repo root, so this appends a
nonexistent `.../AVRLM/AVRLM` path. Confirmed harmless — imports resolve via the script's own
directory being on `sys.path` regardless — but it's leftover copy-paste with no effect. Not fixed
(non-blocking, cosmetic, out of scope per this round's read-only audit stance).

**11.2.3 — `spikes` arriving at `generate_2_5d_grid()` is `float`, not `uint8`/`int32`, from every
real caller.** Confirmed by reading all three integration points (`integration_pipeline.py:50`,
`dashboard_pro.py:159`, `dashboard_driving.py:210`): all three compute
`spikes_np = total_spikes.sum(dim=1)...cpu().numpy()` from a `torch.float32` tensor. The CLAUDE.md
contract nominally specifies `uint8`/`int32` for `spikes`. `handoff.validate_inputs()` (Round 2's
own fix) checks `points` shape/dtype and `labels`/`spikes` length, but never checks `spikes`
dtype. Functionally harmless — every consumer (`aggregate.py`'s `spike_sum > 0` gating,
`GridState`) works identically on floats — but it means the dtype half of the interface contract
is silently unenforced at the one boundary that does have a validator. **Not a bug, but worth a
decision**: either loosen the documented contract to say "numeric count, dtype not enforced," or
extend `validate_inputs()` to check it — currently neither has happened.

**11.2.4 — Dashboard `scene_to_tensor()` normalization is sound.** Checked whether the raw,
un-normalized intensity channel (`norm_tensor[3,:] = pts_sampled[:,3]`, unlike x/y/z which are
scaled to `[0,1]`) creates a scale mismatch feeding the model: `synthetic_lidar_data.py`'s
intensity generators all draw from ranges within `[0.1, 0.9]` (grep-verified across all 4
generators), which is already compatible with the `[0,1]` coordinate scale — no bug here, closing
out a lead raised during this round's static pass.

**11.2.5 — SNN output tensor indexing verified correct.** Checked `spk_rec` shape end-to-end since
a channel/point-dimension mixup would silently corrupt every prediction: `Conv1d` layers keep
tensors as `(batch, channels, num_points)`, so `spk_rec` (stacked over `num_steps`) is
`(num_steps, batch, 3, num_points)`; `total_spikes = spk_rec.sum(dim=0)` → `(batch, 3, num_points)`;
`torch.argmax(total_spikes, dim=1)` correctly reduces over the **class-channel** dimension, not
points or batch. Confirmed correct in both dashboards and `integration_pipeline.py`.

### 11.3 Live dashboard testing (Streamlit + Chrome)

Both apps launched with `streamlit run <file>.py --server.headless true` from the repo root (so
`dashboard_driving.py`'s absolute checkpoint path and `dashboard_pro.py`'s relative one both
resolve against the same directory) and driven from Chrome via `claude-in-chrome`. Neither process
printed a Python traceback or import error at startup.

**11.3.1 — Cold-start latency is real and non-trivial (~10-15s to first paint).** Both dashboards
show a fully blank page (just the Streamlit "Stop/Deploy" toolbar, spinner running) for roughly
10-15 seconds after `streamlit run` before any UI content renders — the `torch`/`snntorch` import
chain and `SpikingPointNet` construction inside `@st.cache_resource load_model()` dominate this.
Not a bug, but worth knowing for a live demo: don't launch the dashboard in front of judges and
expect instant content.

**11.3.2 — `dashboard_pro.py`, "Generate Scene" click, confirmed working end-to-end.** Sidebar
correctly showed "No checkpoint found — run synthetic_train_loop_v5.py" in red before the click.
After clicking (via the DOM element ref — a plain pixel-coordinate click on the first
mis-rendered layout pass did not register; see 11.3.4), the pipeline ran and rendered a 3D Plotly
scatter of the scene with **7,718 active cells (777 fine ≤10m, 6,941 coarse >10m)** — figures
matching this run's own `memory_metrics()`, i.e. internally consistent. However, with the
untrained model, **7,598 of 7,718 cells (98.4%) were classified "Static Obstacle"** (115
"Drivable", 5 "Dynamic Threat"), and the DBSCAN-based object panel listed **76 separate "Static
Obstacle" objects**, most with only a handful of points — a direct visual demonstration of 11.2.1.
Reported "Speed" metric: **2 FPS**. This number is computed by `profiler.evaluate_efficiency()`
from a timer that starts before `scene_to_tensor`/model inference/`generate_2_5d_grid` and stops
right after — it does **not** include the subsequent DBSCAN `cluster_objects()` call. This means
the dashboard's headline "Speed" metric measures model+grid latency only, while the actual
click-to-render wall time users experience is higher (clustering ~7,600 points into candidate
objects is not free). Not a bug, but the metric's scope should be described more precisely
in-app if it's meant to represent perceived responsiveness — currently the label "Speed" implies
end-to-end.

**11.3.3 — `dashboard_driving.py`, "Start Driving Sequence" click, ran correctly through visible
frames, but a websocket reconnect silently wiped the finished sequence back to the initial state
with no error shown.** Clicking "Start Driving Sequence" (default 40 frames, 0.2s delay)
progressed normally — frame counter, UGV world position, per-frame metrics (Speed, Grid Cells,
Memory savings ratio, Sparsity, Energy, Objects) and the 3D scatter all updated live and matched
expectations (e.g. frame 5/40 showed Memory "587.0x" savings, consistent with `memory_metrics()`'s
formula). In *this* process's random weight init, the class skew went the other way from
`dashboard_pro.py`'s run: at frame 5 the scene was almost entirely "Dynamic Threat" (75 filtered
objects, 46 total) rather than "Static Obstacle" — reinforcing 11.2.1 that the skew direction is
purely a function of unseeded weight init, not a deterministic bug in either file. Server stderr
during this run showed repeated `tornado.websocket.WebSocketClosedError` bursts (visible in
`driving_stderr.log` at the time), coinciding with the browser tab being reused for other
navigation during this audit session. After that reconnect, the page fully reset to its initial
"Start Driving Sequence" pre-click state — no frame counter, no metrics, no chart, and critically
**no "Sequence complete." success message** (which the code does emit via
`progress_placeholder.success(...)` at the end of a normal run) — meaning the reset was a mid-run
Streamlit script rerun triggered by the reconnect, not a natural completion. **This is a real
reliability finding, not just an audit-tooling artifact**: because none of the per-frame state
(frame index, accumulated `GridState`, displayed chart) is held in `st.session_state`, any
WebSocket disconnect/reconnect during a running sequence — which can happen from something as
mundane as a laptop sleeping, a tab losing focus long enough to be throttled, or a network hiccup,
not just automated browser control — silently discards the in-progress or just-finished sequence
and returns the user to square one with no error message explaining why. Worth a team decision:
either accept this as a known limitation of `st.empty()`-based live updates, or move key
progress/results into `st.session_state` so a reconnect mid-sequence doesn't lose the run.

**11.3.4 — Minor UI-testing note, not a dashboard bug**: an initial pixel-coordinate click on
"Generate Scene" (`dashboard_pro.py`) computed from a screenshot did not register — the
accessibility-tree viewport (2048px wide) and the screenshot's raster width (1560px) were on
different scales at that moment, so the translated click coordinate landed off-element. Clicking
via the element's DOM reference instead worked immediately. Not a finding about the app itself,
noted only so a future audit doesn't waste time suspecting a frontend bug when it's a coordinate-
scaling mismatch in the browser-automation step.

**11.3.5 — `use_container_width` deprecation warning floods dashboard stderr on every frame.**
Both dashboards call `st.plotly_chart(fig, use_container_width=True)`. The installed Streamlit
version prints `Please replace 'use_container_width' with 'width'... will be removed after
2025-12-31` to stderr on **every single call** — for `dashboard_driving.py`'s 40-frame default
sequence that's 40+ repeated multi-line warnings per run, and it fires on every `st.button`/
`use_container_width=True` call elsewhere too. Purely cosmetic today (the call still works), but
the stated removal date has already passed relative to this session's clock, meaning a Streamlit
upgrade could break both dashboards outright. Simple fix whenever the team wants it:
`use_container_width=True` → `width="stretch"` in both files (3 call sites total: 1 in
`dashboard_pro.py`, 2 in `dashboard_driving.py`... plus the `st.button(...,
use_container_width=True)` calls). Not fixed here per this round's read-only audit stance — flagged
for a decision.

**11.3.6 — No crashes, no unhandled exceptions, no blank-page failures observed in either
dashboard under normal interaction.** Aside from 11.3.3's reconnect-reset behavior, both apps
handled button clicks, slider adjustments, checkbox toggles, and the full render pipeline
(model inference → `generate_2_5d_grid` → DBSCAN clustering → Plotly 3D scatter → sidebar
detected-objects panel) without a single server-side traceback or browser console error across
the whole session.

### 11.4 Summary — new items this round

**Blockers:** none. The core grid module remains correct and now consistently meets its own
real-time budget (§11.1); no new crashes or data-corruption bugs were found in either dashboard.

**Non-blocking, worth a team decision:**
1. Sequence state loss on WebSocket reconnect mid-animation in `dashboard_driving.py` (§11.3.3) —
   the most concrete new finding this round; could visibly bite a live demo.
2. Dashboards are not demo-safe without first running `synthetic_train_loop_v5.py` to produce
   `snn_weights.pth` — no checkpoint currently exists in the repo, and predictions are unseeded and
   unstable run-to-run without one (§11.2.1, §11.3.2, §11.3.3).
3. `use_container_width` deprecation spam (§11.3.5) — trivial fix, not yet applied.
4. Dead `sys.path.append(.../AVRLM)` in both dashboards (§11.2.2) — cosmetic, harmless.
5. `spikes` dtype (`float`, not `uint8`/`int32`) is unchecked by `validate_inputs()` at the one
   real entry point that validates anything (§11.2.3) — decide whether to enforce or to formally
   loosen the documented contract.
6. Dashboard "Speed" metric label doesn't reflect that DBSCAN clustering cost is excluded from the
   timed region (§11.3.2) — minor labeling/accuracy issue, not a functional bug.

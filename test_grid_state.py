import numpy as np

from radial_filter import prefilter_mask
from grid import VariableResolutionGrid
from grid_state import GridState
from synthetic_lidar_data import build_moving_sequence, LABEL_DYNAMIC

START_XY = (6.0, -3.0)
VELOCITY = (0.4, 0.15)

probe_grid = VariableResolutionGrid()

CAP = 100  # shared cap for the exact-eviction test group below


def cluster_cell_keys(points, labels):
    """Ground truth: which cell keys the frame's LABEL_DYNAMIC points fall
    into, computed independently of GridState's internal bookkeeping."""
    cluster_pts = points[labels == LABEL_DYNAMIC]
    mask = prefilter_mask(cluster_pts)
    cluster_pts = cluster_pts[mask]
    assignment = probe_grid.assign_cells(cluster_pts)
    keys = set(zip(
        assignment.parent_ix.tolist(), assignment.parent_iy.tolist(),
        assignment.sub_ix.tolist(), assignment.sub_iy.tolist(),
    ))
    return keys


# Deterministic index -> (x, y) mesh, coarse zone only (10m < r <= 100m),
# one 0.5m coarse cell per grid step so distinct indices always land in
# distinct cells. Domain: i in [0, 1600) via a 40x40 grid based at
# (50.0, 50.0), max r = sqrt(69.5^2+69.5^2) ~= 98.3m, min r = sqrt(50^2+50^2)
# ~= 70.7m - safely inside (10, 100] for every i in range.
_MESH_SIDE = 40


def _mesh_xy(i):
    row, col = divmod(i, _MESH_SIDE)
    return 50.001 + 0.5 * col, 50.001 + 0.5 * row


def _unique_frame(i):
    """One point, one unique cell (coarse zone, isolated), spike_sum>0 so it
    always gets committed - used to control exactly which cells GridState
    sees, for the exact-eviction tests below."""
    x, y = _mesh_xy(i)
    points = np.array([[x, y, 0.1, 0.1]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    spikes = np.array([1], dtype=np.uint8)
    return points, labels, spikes


STRESS_MESH_SIDE = 60  # separate, larger mesh so the long-horizon test doesn't run out of fresh indices


def _stress_xy(i):
    row, col = divmod(i, STRESS_MESH_SIDE)
    return 50.001 + 0.5 * col, 50.001 + 0.5 * row


def _stress_frame(i):
    x, y = _stress_xy(i)
    points = np.array([[x, y, 0.1, 0.1]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    spikes = np.array([1], dtype=np.uint8)
    return points, labels, spikes


def test_event_driven_update_across_moving_sequence():
    frames = build_moving_sequence(n_frames=10, start_xy=START_XY, velocity=VELOCITY)
    state = GridState()

    for t, (pts, lbl, spk) in enumerate(frames):
        before = dict(state._cells)  # key -> CellRecord object refs, pre-update
        touched = state.update(pts, lbl, spk)

        # --- Core invariant (post cold-start fix): a zero-spike cell that was
        # ALREADY cached keeps its exact pre-update value (robust to
        # ground-plane's own random spiking and pole's always-nonzero spiking -
        # see plan Task 2 note). A zero-spike cell seen for the FIRST time is
        # now committed as a baseline record instead of staying absent. ---
        for key, spike_sum in touched.items():
            if spike_sum == 0:
                if key in before:
                    assert state._cells[key] is before[key], \
                        f"frame {t}: zero-spike cached cell {key} was overwritten"
                else:
                    assert key in state._cells, \
                        f"frame {t}: zero-spike first-sight cell {key} was not baselined"

        # --- Regression check for the cold-start gating bug: frame 0 must
        # ingest EVERY touched cell as baseline, not just the spiking subset. ---
        if t == 0:
            assert len(state._cells) == len(touched), \
                f"frame 0: expected full baseline ({len(touched)} cells), got {len(state._cells)}"

        # --- Cells the moving cluster's points actually land in this frame
        # must update (dynamic cluster spikes are always >0 by construction). ---
        cluster_keys = cluster_cell_keys(pts, lbl)
        assert len(cluster_keys) > 0, f"frame {t}: expected cluster to touch at least one cell"
        for key in cluster_keys:
            assert touched.get(key, 0) > 0, f"frame {t}: cluster cell {key} failed to update"

        n_changed = sum(1 for s in touched.values() if s > 0)
        n_elsewhere_changed = n_changed - len(cluster_keys)
        print(f"frame {t}: touched={len(touched)}, changed={n_changed}, "
              f"cluster_cells={len(cluster_keys)}, other_changed={n_elsewhere_changed}, "
              f"total_cached={len(state._cells)}")


def test_cold_start_recovers_true_ungated_count():
    """Dedicated regression check for the reported bug: build_scene() through
    a fresh GridState must recover the true ungated occupied-cell count, not
    just the ~3% of cells that happened to spike (see
    debug_guide_cell_count_discrepancy.md)."""
    from aggregate import aggregate_cells
    from synthetic_lidar_data import build_scene

    scene_points, scene_labels, scene_spikes = build_scene()
    scene_mask = prefilter_mask(scene_points)
    f_scene_points = scene_points[scene_mask]
    f_scene_labels = scene_labels[scene_mask]
    f_scene_spikes = scene_spikes[scene_mask]
    true_assignment = probe_grid.assign_cells(f_scene_points)
    true_stats = aggregate_cells(f_scene_points, f_scene_labels, f_scene_spikes, true_assignment)
    true_ungated_count = len(true_stats)

    fresh_state = GridState()
    fresh_state.update(scene_points, scene_labels, scene_spikes)
    print(f"build_scene() cold-start regression check: true_ungated={true_ungated_count}, "
          f"GridState_after_one_update={len(fresh_state._cells)}")
    assert len(fresh_state._cells) == true_ungated_count, \
        f"cold-start baseline mismatch: expected {true_ungated_count}, got {len(fresh_state._cells)}"


def test_lru_eviction_plateaus_at_cap():
    """Cache plateaus at max_cells, never evicts a cell touched this frame
    (AUDIT-v2 §2/§5.2, Phase 8)."""
    MAX_CELLS = 30000  # above the ~21,000 cells touched per frame in this scene,
                        # below the ~115,000 the sequence accumulates unbounded
    bounded_frames = build_moving_sequence(n_frames=20, start_xy=START_XY, velocity=VELOCITY)
    bounded_state = GridState(max_cells=MAX_CELLS)
    plateaued = False
    for t, (pts, lbl, spk) in enumerate(bounded_frames):
        touched = bounded_state.update(pts, lbl, spk)
        assert len(bounded_state._cells) <= MAX_CELLS, \
            f"frame {t}: cache grew to {len(bounded_state._cells)}, exceeding max_cells={MAX_CELLS}"
        # every cell touched this frame must still be present (never evicted
        # mid-frame, even though eviction runs after this frame's touches)
        for key in touched:
            assert key in bounded_state._cells, \
                f"frame {t}: cell {key} touched this frame was evicted"
        if t >= 5:
            plateaued = plateaued or len(bounded_state._cells) == MAX_CELLS
    print(f"LRU eviction: cache size after 20 frames = {len(bounded_state._cells)} "
          f"(max_cells={MAX_CELLS}), plateaued at cap: {plateaued}")
    assert plateaued, "expected cache to reach and hold the max_cells cap over 20 frames"


def test_lru_eviction_exact_fill_plus_one():
    """Fill the cache to exactly max_cells with brand-new unique cells,
    then add one more - exactly one eviction should occur, and it must be
    the least-recently-touched (oldest) cell (AUDIT-v2 test-rebuild
    Phase 4)."""
    exact_state = GridState(max_cells=CAP)
    first_key = None
    for i in range(CAP):
        pts, lbl, spk = _unique_frame(i)
        touched = exact_state.update(pts, lbl, spk)
        if i == 0:
            first_key = next(iter(touched))
    assert len(exact_state._cells) == CAP, \
        f"expected exactly {CAP} cells after filling to capacity, got {len(exact_state._cells)}"
    assert first_key in exact_state._cells, "first-created cell should still be present before overflow"

    pts, lbl, spk = _unique_frame(CAP)  # one brand-new cell beyond capacity
    exact_state.update(pts, lbl, spk)
    assert len(exact_state._cells) == CAP, \
        f"expected size to stay capped at {CAP} after overflow, got {len(exact_state._cells)}"
    assert first_key not in exact_state._cells, \
        "expected the least-recently-touched (first-created) cell to be evicted"
    print(f"exact fill+1 eviction: cache held at {CAP}, oldest cell {first_key} correctly evicted")


def test_lru_vs_fifo_touch_before_evict():
    """Distinguishes LRU from FIFO/random eviction (AUDIT-v2 test-rebuild
    Phase 4). Fill to capacity, then re-touch the oldest cell right before
    forcing an eviction - it must survive, and a DIFFERENT (truly stale)
    cell must be evicted instead."""
    lru_state = GridState(max_cells=CAP)
    keys_in_order = []
    for i in range(CAP):
        pts, lbl, spk = _unique_frame(i)
        touched = lru_state.update(pts, lbl, spk)
        keys_in_order.append(next(iter(touched)))
    oldest_key = keys_in_order[0]
    second_oldest_key = keys_in_order[1]

    # Re-touch the oldest cell (same coordinates -> same key, spike_sum>0 so
    # it's treated as touched and moved to the end of the LRU order).
    re_touch_pts, re_touch_lbl, re_touch_spk = _unique_frame(0)
    lru_state.update(re_touch_pts, re_touch_lbl, re_touch_spk)

    # Now force one eviction with a brand-new cell.
    overflow_pts, overflow_lbl, overflow_spk = _unique_frame(CAP)
    lru_state.update(overflow_pts, overflow_lbl, overflow_spk)

    assert oldest_key in lru_state._cells, \
        "the re-touched (originally oldest) cell should survive - this is what LRU means, not FIFO"
    assert second_oldest_key not in lru_state._cells, \
        "expected the next-oldest UNTOUCHED cell to be evicted instead, confirming true LRU order"
    print(f"LRU vs FIFO: re-touched oldest cell {oldest_key} survived, "
          f"next-oldest untouched cell {second_oldest_key} evicted instead")


def test_lru_eviction_under_pressure_same_call_batch():
    """Pre-fill from PRIOR calls (60 cells, under cap), then touch a NEW
    batch of 80 cells in a single update() call. 60+80=140 exceeds the cap
    of 100, so eviction must run - but every cell touched in that single
    call must survive; only older, prior-call cells may be evicted."""
    pressure_state = GridState(max_cells=CAP)
    PREFILL_N = 60
    for i in range(PREFILL_N):
        pts, lbl, spk = _unique_frame(i)
        pressure_state.update(pts, lbl, spk)
    assert len(pressure_state._cells) == PREFILL_N

    BATCH_N = 80
    batch_xy = [_mesh_xy(PREFILL_N + i) for i in range(BATCH_N)]
    batch_x = np.array([xy[0] for xy in batch_xy], dtype=np.float32)
    batch_y = np.array([xy[1] for xy in batch_xy], dtype=np.float32)
    batch_z = np.full(BATCH_N, 0.1, dtype=np.float32)
    batch_i = np.full(BATCH_N, 0.1, dtype=np.float32)
    batch_points = np.stack([batch_x, batch_y, batch_z, batch_i], axis=1).astype(np.float32)
    batch_labels = np.zeros(BATCH_N, dtype=np.int64)
    batch_spikes = np.ones(BATCH_N, dtype=np.uint8)
    batch_touched = pressure_state.update(batch_points, batch_labels, batch_spikes)
    assert len(batch_touched) == BATCH_N, \
        f"test setup issue: expected the batch to touch {BATCH_N} unique new cells, got {len(batch_touched)}"
    for key in batch_touched:
        assert key in pressure_state._cells, \
            f"cell {key} touched within this single update() call was evicted despite fitting under the cap"
    assert len(pressure_state._cells) == CAP, \
        f"expected cache trimmed to exactly {CAP} after the batch, got {len(pressure_state._cells)}"
    print(f"eviction under pressure: prefilled {PREFILL_N} + single-call batch of {BATCH_N} "
          f"(total {PREFILL_N + BATCH_N} > cap={CAP}); every batch-touched cell survived, "
          f"only older prior-call cells were evicted, cache trimmed to {CAP}")


def test_lru_single_call_batch_exceeds_cap_outright():
    """Explicit confirmation of a mathematical edge case: if a SINGLE
    update() call's own batch exceeds max_cells outright, not every cell
    touched in that call can survive (there's no smaller/older set left to
    evict instead) - this is an inherent limit of the cap, not a bug."""
    overflow_state = GridState(max_cells=CAP)
    n_overflow_batch = 5 * CAP
    overflow_xy = [_mesh_xy(i) for i in range(n_overflow_batch)]
    overflow_x = np.array([xy[0] for xy in overflow_xy], dtype=np.float32)
    overflow_y = np.array([xy[1] for xy in overflow_xy], dtype=np.float32)
    overflow_z = np.full(n_overflow_batch, 0.1, dtype=np.float32)
    overflow_i = np.full(n_overflow_batch, 0.1, dtype=np.float32)
    overflow_points = np.stack([overflow_x, overflow_y, overflow_z, overflow_i], axis=1).astype(np.float32)
    overflow_labels = np.zeros(n_overflow_batch, dtype=np.int64)
    overflow_spikes = np.ones(n_overflow_batch, dtype=np.uint8)
    overflow_touched = overflow_state.update(overflow_points, overflow_labels, overflow_spikes)
    assert len(overflow_touched) == n_overflow_batch
    assert len(overflow_state._cells) == CAP, \
        f"cache must still be trimmed to the cap ({CAP}) even when a single call's batch exceeds it, " \
        f"got {len(overflow_state._cells)}"
    survivors = sum(1 for key in overflow_touched if key in overflow_state._cells)
    assert survivors == CAP, \
        f"expected exactly {CAP} of this call's own {n_overflow_batch} touched cells to survive, got {survivors}"
    print(f"single-call batch ({n_overflow_batch}) exceeding the cap ({CAP}) on its own: "
          f"cache still trimmed to {CAP}, exactly {survivors} of that call's own cells survive "
          f"(the rest are evicted within the same call - expected, not a bug)")


def test_lru_long_horizon_plateau_800_frames():
    """Many eviction cycles over hundreds of frames must plateau at the
    cap, not oscillate or drift (AUDIT-v2 test-rebuild Phase 4). Mixes
    brand-new cells with re-touches of recently-seen cells to simulate
    locality, similar to a real moving-sensor workload."""
    STRESS_CAP = 500
    stress_state = GridState(max_cells=STRESS_CAP)
    N_STRESS_FRAMES = 800
    PLATEAU_CHECK_START = 700  # comfortably after ~640 new cells (4/5 of frames) should exceed the cap
    next_new_idx = 0
    recent_new = []  # last few brand-new indices, for locality re-touches
    sizes_after_fill = []
    for f in range(N_STRESS_FRAMES):
        if f % 5 == 4 and recent_new:
            idx = recent_new[-1]  # re-touch the most recently created cell (locality)
        else:
            idx = next_new_idx
            next_new_idx += 1
            recent_new.append(idx)
            if len(recent_new) > 5:
                recent_new.pop(0)
        pts, lbl, spk = _stress_frame(idx)
        stress_state.update(pts, lbl, spk)
        assert len(stress_state._cells) <= STRESS_CAP, \
            f"frame {f}: cache grew to {len(stress_state._cells)}, exceeding max_cells={STRESS_CAP}"
        if f >= PLATEAU_CHECK_START:
            sizes_after_fill.append(len(stress_state._cells))

    assert next_new_idx > STRESS_CAP, \
        f"test setup issue: only {next_new_idx} distinct new cells were ever created, need > {STRESS_CAP} to force a real cap"
    assert min(sizes_after_fill) == STRESS_CAP == max(sizes_after_fill), (
        f"expected a true plateau at {STRESS_CAP} after the fill window, "
        f"got range [{min(sizes_after_fill)}, {max(sizes_after_fill)}] over "
        f"{len(sizes_after_fill)} frames"
    )
    print(f"long-horizon stress: {N_STRESS_FRAMES} frames ({next_new_idx} distinct new cells), "
          f"cap={STRESS_CAP}, held a true plateau (no oscillation/drift) "
          f"from frame {PLATEAU_CHECK_START} onward")


if __name__ == "__main__":
    test_event_driven_update_across_moving_sequence()
    test_cold_start_recovers_true_ungated_count()
    test_lru_eviction_plateaus_at_cap()
    test_lru_eviction_exact_fill_plus_one()
    test_lru_vs_fifo_touch_before_evict()
    test_lru_eviction_under_pressure_same_call_batch()
    test_lru_single_call_batch_exceeds_cap_outright()
    test_lru_long_horizon_plateau_800_frames()
    print("test_grid_state.py: all assertions passed")

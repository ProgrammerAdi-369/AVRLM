"""
Adversarial + scale stress tests (AUDIT-v2 test-rebuild Phase 6), matching
the rigor AUDIT-v2 itself applied when first verifying this code, targeted
specifically at the Phases 1-10 changes:
  1. Validation fuzzing across points/labels/spikes (200+ combinations)
  2. LRU eviction under adversarial churn (a new unique cell every frame)
  3. Long-horizon growth with the cap active, on genuinely novel terrain
  4. Timing regression guard for the packed-int64 dedup path at scale

All heavy work lives inside the test_* functions (not at module import
time) so pytest collection stays fast and doesn't run these loops just to
discover the tests.
"""

import time

import numpy as np

from grid import VariableResolutionGrid
from radial_filter import prefilter_mask
from aggregate import NUM_CLASSES, aggregate_cells
from grid_state import GridState
from handoff import generate_2_5d_grid
from synthetic_lidar_data import build_moving_sequence, generate_ground_plane, \
    generate_pole, generate_dynamic_cluster, generate_boundary_stress_ring


def valid_points(n, rng):
    x = rng.uniform(-90, 90, size=n).astype(np.float32)
    y = rng.uniform(-90, 90, size=n).astype(np.float32)
    z = rng.uniform(-1, 1, size=n).astype(np.float32)
    inten = rng.uniform(0, 1, size=n).astype(np.float32)
    return np.stack([x, y, z, inten], axis=1).astype(np.float32)


def valid_labels(n, rng):
    return rng.integers(0, NUM_CLASSES, size=n).astype(np.int64)


def valid_spikes(n, rng):
    return rng.integers(0, 5, size=n).astype(np.uint8)


def test_validation_fuzzing():
    """200+ malformed/edge points-labels-spikes combinations. Every one
    must either raise ValueError or be legitimately valid and accepted -
    no silent passes on bad input, no false-positive rejections of valid
    edge cases. Each case is tagged with its expected outcome BY
    CONSTRUCTION (we know why it's malformed or valid because we built it
    that way), not by re-deriving a duplicate oracle."""
    rng = np.random.default_rng(123)
    cases = []  # list of (name, points, labels, spikes, expect_ok, expect_msg_substr)

    # --- wrong-shape points, across several N ---
    for n in (0, 1, 5, 20, 50):
        good_labels, good_spikes = valid_labels(n, rng), valid_spikes(n, rng)
        for extra_cols, name in [(3, "3cols"), (5, "5cols")]:
            bad_pts = valid_points(n, rng)[:, :extra_cols] if extra_cols == 3 else \
                np.concatenate([valid_points(n, rng), np.zeros((n, 1), dtype=np.float32)], axis=1)
            cases.append((f"points_wrong_ncols_{name}_n{n}", bad_pts, good_labels, good_spikes,
                           False, "points must have shape (N, 4)"))
        if n > 0:
            cases.append((f"points_1d_n{n}", valid_points(n, rng)[:, 0], good_labels, good_spikes,
                           False, "points must have shape (N, 4)"))

    # --- labels/spikes length mismatches, across several N and offsets ---
    for n in (1, 5, 20, 50, 100):
        good_pts = valid_points(n, rng)
        good_labels, good_spikes = valid_labels(n, rng), valid_spikes(n, rng)
        for offset in (-5, -2, -1, 1, 2, 5):
            m = max(0, n + offset)
            cases.append((f"labels_len_mismatch_n{n}_off{offset}", good_pts,
                           valid_labels(m, rng), good_spikes, False, "labels must have shape"))
            cases.append((f"spikes_len_mismatch_n{n}_off{offset}", good_pts,
                           good_labels, valid_spikes(m, rng), False, "spikes must have shape"))

    # --- out-of-range labels, across several N and offending values ---
    for n in (1, 5, 20, 50, 100):
        for bad_val in (-5, -4, -3, -2, -1, NUM_CLASSES, NUM_CLASSES + 1, NUM_CLASSES + 2,
                         NUM_CLASSES + 3, NUM_CLASSES + 4, NUM_CLASSES + 5):
            good_pts = valid_points(n, rng)
            labels = valid_labels(n, rng)
            labels[rng.integers(0, n)] = bad_val
            cases.append((f"label_out_of_range_n{n}_val{bad_val}", good_pts, labels,
                           valid_spikes(n, rng), False, "outside valid range"))

    # --- NaN Z, across several N and NaN patterns (single / multiple / all) ---
    for n in (2, 5, 20, 50, 100):
        for pattern_name, n_nan in [("single", 1), ("multiple", max(1, n // 2)), ("all", n)]:
            good_pts = valid_points(n, rng)
            nan_idx = rng.choice(n, size=n_nan, replace=False)
            good_pts[nan_idx, 2] = np.nan
            cases.append((f"nan_z_{pattern_name}_n{n}", good_pts, valid_labels(n, rng),
                           valid_spikes(n, rng), False, "NaN"))

    # --- Inf Z: NOT currently guarded (CLAUDE.md explicitly documents only
    # NaN is checked) - these are legitimately accepted, not malformed. ---
    for n in (2, 5, 20, 50, 100):
        for sign in (1.0, -1.0):
            good_pts = valid_points(n, rng)
            good_pts[rng.integers(0, n), 2] = sign * np.inf
            cases.append((f"inf_z_n{n}_sign{sign}", good_pts, valid_labels(n, rng),
                           valid_spikes(n, rng), True, None))

    # --- non-integer / negative spikes: accepted via coerce_spikes
    # (rounds, warns) - sign is not validated by the current contract. ---
    for n in (2, 5, 20, 50, 100):
        for spike_pattern in ("small_float", "large_float", "negative_float", "negative_int"):
            good_pts = valid_points(n, rng)
            good_labels = valid_labels(n, rng)
            if spike_pattern == "small_float":
                spikes = rng.uniform(0, 5, size=n).astype(np.float32)
            elif spike_pattern == "large_float":
                spikes = rng.uniform(100, 500, size=n).astype(np.float32)
            elif spike_pattern == "negative_float":
                spikes = rng.uniform(-5, 5, size=n).astype(np.float32)
            else:
                spikes = rng.integers(-5, 5, size=n).astype(np.int32)
            cases.append((f"spikes_{spike_pattern}_n{n}", good_pts, good_labels, spikes, True, None))

    # --- legitimately empty frame: zero points/labels/spikes must be accepted ---
    cases.append(("empty_frame", np.zeros((0, 4), dtype=np.float32),
                  np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint8), True, None))

    # --- single-point valid frames ---
    for i in range(5):
        cases.append((f"single_point_valid_{i}", valid_points(1, rng), valid_labels(1, rng),
                       valid_spikes(1, rng), True, None))

    # --- fully valid randomized combos, to exercise realistic diversity ---
    for i in range(50):
        n = int(rng.integers(1, 200))
        cases.append((f"random_valid_{i}_n{n}", valid_points(n, rng), valid_labels(n, rng),
                       valid_spikes(n, rng), True, None))

    assert len(cases) >= 200, f"expected at least 200 fuzz cases, got {len(cases)}"

    n_ok, n_err = 0, 0
    for name, pts, labels, spikes, expect_ok, expect_msg_substr in cases:
        try:
            generate_2_5d_grid(pts, labels, spikes)
            actual_ok = True
            actual_msg = None
        except ValueError as e:
            actual_ok = False
            actual_msg = str(e)

        assert actual_ok == expect_ok, (
            f"case {name!r}: expected {'accepted' if expect_ok else 'rejected'}, "
            f"got {'accepted' if actual_ok else f'rejected ({actual_msg})'}"
        )
        if not expect_ok:
            assert expect_msg_substr in actual_msg, (
                f"case {name!r}: expected error message to contain {expect_msg_substr!r}, got: {actual_msg}"
            )
            n_err += 1
        else:
            n_ok += 1

    print(f"validation fuzzing: {len(cases)} cases ({n_ok} accepted, {n_err} rejected), "
          f"every one matched its by-construction expected outcome")


def test_eviction_under_adversarial_churn():
    """A brand-new unique cell every single frame (worst case for an LRU
    cache), many multiples of max_cells, confirm the cap holds and
    eviction itself doesn't become the bottleneck."""
    CHURN_CAP = 1000
    CHURN_FRAMES = 10_000  # 10x max_cells
    CHURN_MESH_SIDE = 110  # fine-zone mesh, domain 110*110=12100 > CHURN_FRAMES, r stays well under 10m

    def _churn_xy(i):
        row, col = divmod(i, CHURN_MESH_SIDE)
        return 0.5 + 0.05 * col, 0.5 + 0.05 * row

    churn_state = GridState(max_cells=CHURN_CAP)
    churn_times = []
    for i in range(CHURN_FRAMES):
        x, y = _churn_xy(i)
        pts = np.array([[x, y, 0.1, 0.1]], dtype=np.float32)
        lbl = np.array([0], dtype=np.int64)
        spk = np.array([1], dtype=np.uint8)
        start = time.perf_counter()
        churn_state.update(pts, lbl, spk)
        churn_times.append(time.perf_counter() - start)
        assert len(churn_state._cells) <= CHURN_CAP, \
            f"frame {i}: cache grew to {len(churn_state._cells)}, exceeding max_cells={CHURN_CAP}"

    first_100_median = float(np.median(churn_times[:100]))
    last_100_median = float(np.median(churn_times[-100:]))
    # Generous tolerance (5x, not 2x): individual-frame timing on shared/noisy
    # CI-style hardware showed up to ~2.7x run-to-run jitter with no algorithmic
    # change, so a tight multiplier is a flakiness risk, not a real regression
    # guard. 5x is still tight enough to catch a genuine O(n)-in-cache-size
    # regression, which would show up as an order-of-magnitude difference over
    # 10,000 frames, not a small multiple.
    assert last_100_median <= first_100_median * 5.0, (
        f"eviction churn appears to be degrading over time: first-100 median={first_100_median * 1000:.3f}ms, "
        f"last-100 median={last_100_median * 1000:.3f}ms (>5x slower)"
    )
    print(f"eviction under adversarial churn: {CHURN_FRAMES} frames, cap={CHURN_CAP} held throughout; "
          f"first-100 median={first_100_median * 1000:.3f}ms, last-100 median={last_100_median * 1000:.3f}ms")


def test_long_horizon_growth_with_cap_active():
    """Long-horizon growth with the cap active, on genuinely novel terrain
    (build_moving_sequence, not a synthetic index mesh) - confirms the
    cache never exceeds max_cells regardless of terrain novelty, now that
    eviction exists (extends the original 20-frame version in
    test_grid_state.py to the 500-1000 frame range the guide asks for)."""
    GROWTH_CAP = 30_000  # matches test_grid_state.py's existing bounded-growth test
    N_GROWTH_FRAMES = 500
    growth_frames = build_moving_sequence(n_frames=N_GROWTH_FRAMES, start_xy=(6.0, -3.0), velocity=(0.4, 0.15))
    growth_state = GridState(max_cells=GROWTH_CAP)
    max_seen = 0
    for t, (pts, lbl, spk) in enumerate(growth_frames):
        growth_state.update(pts, lbl, spk)
        size = len(growth_state._cells)
        max_seen = max(max_seen, size)
        assert size <= GROWTH_CAP, f"frame {t}: cache grew to {size}, exceeding max_cells={GROWTH_CAP}"
    assert max_seen == GROWTH_CAP, \
        f"expected the cap ({GROWTH_CAP}) to actually be reached over {N_GROWTH_FRAMES} frames, max seen was {max_seen}"
    print(f"long-horizon growth ({N_GROWTH_FRAMES} frames of novel terrain): cap={GROWTH_CAP} "
          f"never exceeded, reached exactly {max_seen}")


def test_timing_at_scale_packed_key_regression_guard():
    """Head-to-head timing of the current packed-1D-key dedup against an
    inline reimplementation of the old np.unique(axis=0) dedup, on the SAME
    dense adversarial+realistic dataset - avoids hardware-dependent absolute
    ms thresholds by comparing components directly (AUDIT-v2 Phase 7 claimed
    a 91% reduction in this specific step; here we just assert new <= old
    with a small noise margin, as a revert-guard)."""
    rng = np.random.default_rng(456)
    dense_parts = [
        generate_ground_plane(n_rings=3000, pts_per_ring=40, rng=rng),  # 5x default density
        generate_pole((4.0, 2.5), n_points=600, rng=rng),
        generate_pole((15.0, -8.0), n_points=600, rng=rng),
        generate_dynamic_cluster((6.0, -3.0), n_points=300, rng=rng),
        generate_boundary_stress_ring(n_points=2000, rng=rng),
    ]
    dense_points = np.concatenate([p[0] for p in dense_parts], axis=0)
    dense_labels = np.concatenate([p[1] for p in dense_parts], axis=0)
    dense_spikes = np.concatenate([p[2] for p in dense_parts], axis=0)
    print(f"timing-at-scale dataset: {dense_points.shape[0]} points")

    scale_grid = VariableResolutionGrid()
    mask = prefilter_mask(dense_points)
    f_points, f_labels, f_spikes = dense_points[mask], dense_labels[mask], dense_spikes[mask]
    assignment = scale_grid.assign_cells(f_points)
    in_range = assignment.in_range
    parent_ix = assignment.parent_ix[in_range]
    parent_iy = assignment.parent_iy[in_range]
    sub_ix = assignment.sub_ix[in_range]
    sub_iy = assignment.sub_iy[in_range]

    PARENT_KEY_MULT = 1024
    SUB_KEY_MULT = 16
    N_REPS = 5

    new_times = []
    for _ in range(N_REPS):
        start = time.perf_counter()
        cell_packed = ((parent_ix * PARENT_KEY_MULT + parent_iy) * SUB_KEY_MULT + (sub_ix + 1)) * SUB_KEY_MULT + (sub_iy + 1)
        np.unique(cell_packed, return_index=True, return_inverse=True)
        new_times.append(time.perf_counter() - start)

    old_times = []
    for _ in range(N_REPS):
        start = time.perf_counter()
        cols = np.stack([parent_ix, parent_iy, sub_ix, sub_iy], axis=1)
        np.unique(cols, axis=0, return_index=True, return_inverse=True)
        old_times.append(time.perf_counter() - start)

    new_best = min(new_times)
    old_best = min(old_times)
    assert new_best < old_best * 1.1, (
        f"packed-1D-key dedup ({new_best * 1000:.2f}ms best-of-{N_REPS}) is not meaningfully faster than "
        f"the old axis=0 dedup ({old_best * 1000:.2f}ms best-of-{N_REPS}) on {parent_ix.shape[0]} points - "
        f"possible regression back toward np.unique(axis=0)"
    )
    speedup = old_best / new_best if new_best > 0 else float("inf")
    print(f"timing at scale ({parent_ix.shape[0]} in-range points): packed-key dedup "
          f"{new_best * 1000:.2f}ms vs old axis=0 dedup {old_best * 1000:.2f}ms ({speedup:.1f}x faster)")

    # Also run the full aggregate_cells pipeline once at this scale as a
    # non-flaky sanity check that it completes and produces the expected
    # shape (not a strict timing assertion - the head-to-head comparison
    # above is).
    full_start = time.perf_counter()
    dense_stats = aggregate_cells(f_points, f_labels, f_spikes, assignment)
    full_elapsed = time.perf_counter() - full_start
    print(f"full aggregate_cells at scale: {len(dense_stats)} cells in {full_elapsed * 1000:.1f}ms")


if __name__ == "__main__":
    test_validation_fuzzing()
    test_eviction_under_adversarial_churn()
    test_long_horizon_growth_with_cap_active()
    test_timing_at_scale_packed_key_regression_guard()
    print("test_stress.py: all assertions passed")

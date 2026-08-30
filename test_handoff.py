import numpy as np
import warnings

import handoff
from handoff import generate_2_5d_grid, memory_metrics, validate_inputs
from aggregate import coerce_spikes
from grid_state import GridState
from synthetic_lidar_data import build_scene, build_moving_sequence


def test_generate_2_5d_grid_accumulates_across_calls():
    handoff._state = GridState()  # isolate from other tests sharing the module singleton

    points, labels, spikes = build_scene()
    grid_out = generate_2_5d_grid(points, labels, spikes)

    print(f"generate_2_5d_grid(build_scene()) -> {len(grid_out)} cells")
    key, record = grid_out[0]
    print(f"schema: key={key} (parent_ix, parent_iy, sub_ix, sub_iy)")
    print(f"record fields: is_fine={record.is_fine}, center_x={record.center_x:.3f}, "
          f"center_y={record.center_y:.3f}, elevation_max={record.elevation_max:.3f}, "
          f"elevation_var={record.elevation_var:.5f}, class_id={record.class_id}, "
          f"point_count={record.point_count}")

    metrics = memory_metrics()
    print(f"memory_metrics after build_scene(): {metrics}")

    frames = build_moving_sequence(n_frames=1)
    fpoints, flabels, fspikes = frames[0]
    grid_out2 = generate_2_5d_grid(fpoints, flabels, fspikes)
    print(f"generate_2_5d_grid(one moving-sequence frame) -> {len(grid_out2)} cells "
          f"(accumulated on top of build_scene() call above)")

    metrics2 = memory_metrics()
    print(f"memory_metrics after second call: {metrics2}")
    assert metrics2["active_cell_count"] >= metrics["active_cell_count"], \
        "state should accumulate across calls, not reset"


def test_float_spikes_coercion_accepted():
    """AUDIT-v2 §3.5: accepted, rounded, >0 gating intact."""
    handoff._state = GridState()

    float_pts = np.array([[50.0, 50.0, 0.1, 0.1]], dtype=np.float32)  # coarse zone, isolated cell
    float_labels = np.array([0], dtype=np.int64)
    float_spikes = np.array([2.7], dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grid_out3 = generate_2_5d_grid(float_pts, float_labels, float_spikes)
        assert any("non-integer" in str(w.message) for w in caught), \
            "expected a non-integer spikes warning"
    key3, record3 = grid_out3[-1]
    assert record3.point_count >= 1
    print(f"float spikes [2.7] accepted with warning, cell committed (point_count={record3.point_count})")


def test_validate_inputs_shape_length_mismatches():
    """AUDIT-v2 §1/§8."""
    good_pts = np.array([[1.0, 1.0, 0.1, 0.1], [2.0, 2.0, 0.1, 0.1]], dtype=np.float32)
    try:
        validate_inputs(good_pts, np.zeros((2, 2), dtype=np.int64), np.zeros(2, dtype=np.uint8))
        raise AssertionError("expected ValueError for wrong-shaped labels")
    except ValueError as e:
        print(f"wrong-shaped labels (N,2) correctly raised: {e}")

    try:
        validate_inputs(good_pts, np.zeros(3, dtype=np.int64), np.zeros(2, dtype=np.uint8))
        raise AssertionError("expected ValueError for length-mismatched labels")
    except ValueError as e:
        print(f"length-mismatched labels correctly raised: {e}")

    try:
        validate_inputs(good_pts, np.zeros(2, dtype=np.int64), np.zeros(5, dtype=np.uint8))
        raise AssertionError("expected ValueError for length-mismatched spikes")
    except ValueError as e:
        print(f"length-mismatched spikes correctly raised: {e}")


def test_validate_inputs_points_and_all_three_wrong():
    """Points malformed on its own, and all three simultaneously wrong
    (confirms the short-circuit order: points is checked first)."""
    try:
        validate_inputs(np.zeros((2, 3), dtype=np.float32), np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.uint8))
        raise AssertionError("expected ValueError for wrong-shaped points")
    except ValueError as e:
        assert "points must have shape (N, 4)" in str(e), f"expected points-shape message, got: {e}"
        print(f"wrong-shaped points (N,3) correctly raised: {e}")

    try:
        validate_inputs(np.zeros((2, 3), dtype=np.float32), np.zeros(3, dtype=np.int64), np.zeros(5, dtype=np.uint8))
        raise AssertionError("expected ValueError when points/labels/spikes are all wrong at once")
    except ValueError as e:
        assert "points must have shape (N, 4)" in str(e), \
            f"expected the points check to surface first (short-circuit order), got: {e}"
        print(f"all-three-wrong-at-once correctly raised the points-first message: {e}")


def test_coerce_spikes_rounding_edge_cases():
    """np.round is round-half-to-even, and the warning threshold is a
    strict 1e-6 on the actual float64 diff, not a guessed boundary."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = coerce_spikes(np.array([2.1], dtype=np.float32))
        assert result[0] == 2, f"expected 2.1 to round to 2, got {result[0]}"
        assert len(caught) == 1, "expected exactly one warning for 2.1"
    print("coerce_spikes(2.1): rounds to 2, warns once")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        val = 2.0000001
        result = coerce_spikes(np.array([val], dtype=np.float64))
        assert result[0] == 2, f"expected {val} to round to 2, got {result[0]}"
        diff = abs(val - round(val))
        expect_warning = diff > 1e-6
        assert (len(caught) == 1) == expect_warning, \
            f"diff={diff}, expected warning={expect_warning}, got {len(caught)} warnings"
    print(f"coerce_spikes(2.0000001): rounds to 2, warn-expected={diff > 1e-6} matched actual")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        val = 2.999999
        result = coerce_spikes(np.array([val], dtype=np.float64))
        assert result[0] == 3, f"expected {val} to round to 3, got {result[0]}"
        diff = abs(val - round(val))
        expect_warning = diff > 1e-6
        assert (len(caught) == 1) == expect_warning, \
            f"diff={diff}, expected warning={expect_warning}, got {len(caught)} warnings"
    print(f"coerce_spikes(2.999999): rounds to 3, warn-expected={diff > 1e-6} matched actual")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = coerce_spikes(np.array([2.5], dtype=np.float64))
        # np.round uses round-half-to-even (banker's rounding): 2.5 -> 2, not 3.
        assert result[0] == 2, f"expected round-half-to-even(2.5)==2, got {result[0]}"
        assert len(caught) == 1, "expected a warning for 2.5 (diff=0.5 > 1e-6)"
    print("coerce_spikes(2.5): round-half-to-even gives 2 (not 3), warns once")


def test_coerce_spikes_warns_once_per_call():
    """Warns on every call with non-integer input, not just the first
    ever call."""
    with warnings.catch_warnings(record=True) as caught1:
        warnings.simplefilter("always")
        coerce_spikes(np.array([1.5], dtype=np.float64))
        assert len(caught1) == 1, "expected a warning on first call"
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        coerce_spikes(np.array([1.5], dtype=np.float64))
        assert len(caught2) == 1, "expected a warning on second call too (once-per-call, not once-ever)"
    print("coerce_spikes: warns once-per-call, confirmed across two separate calls")


if __name__ == "__main__":
    test_generate_2_5d_grid_accumulates_across_calls()
    test_float_spikes_coercion_accepted()
    test_validate_inputs_shape_length_mismatches()
    test_validate_inputs_points_and_all_three_wrong()
    test_coerce_spikes_rounding_edge_cases()
    test_coerce_spikes_warns_once_per_call()
    print("test_handoff.py: all assertions passed")

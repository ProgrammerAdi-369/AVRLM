import numpy as np

from radial_filter import prefilter_mask
from grid import VariableResolutionGrid
from aggregate import aggregate_cells, validate_labels, NUM_CLASSES
from synthetic_lidar_data import generate_pole, generate_ground_plane

grid = VariableResolutionGrid()


def run_pipeline(points, labels, spikes):
    mask = prefilter_mask(points)
    points, labels, spikes = points[mask], labels[mask], spikes[mask]
    assignment = grid.assign_cells(points)
    return aggregate_cells(points, labels, spikes, assignment)


def test_pole_scene():
    """Static obstacle, height 2.0m."""
    pole_center = (4.0, 2.5)
    pole_height = 2.0
    pts, lbl, spk = generate_pole(pole_center, height=pole_height)
    stats = run_pipeline(pts, lbl, spk)

    dist_to_pole = np.sqrt((stats.center_x - pole_center[0]) ** 2 + (stats.center_y - pole_center[1]) ** 2)
    pole_cells = dist_to_pole <= 0.15  # pole radius 0.08m + a little slack

    n_pole_cells = int(pole_cells.sum())
    print(f"pole scene: {len(stats)} occupied cells total, {n_pole_cells} near the pole")
    assert n_pole_cells > 0, "expected at least one occupied cell near the pole"

    max_elev_near_pole = stats.elevation_max[pole_cells].max()
    print(f"pole scene: max elevation near pole = {max_elev_near_pole:.3f} (pole height = {pole_height})")
    assert abs(max_elev_near_pole - pole_height) < 0.3, "pole max elevation should be close to pole height"
    assert np.all(stats.class_id[pole_cells] == 1), "pole cells should be class_id 1 (static obstacle)"


def test_ground_plane_scene():
    """Flat drivable terrain near z=0."""
    gpts, glbl, gspk = generate_ground_plane(z_noise=0.02)
    gstats = run_pipeline(gpts, glbl, gspk)

    sample = np.arange(min(20, len(gstats)))
    print("ground plane sample cells (elevation_max, elevation_var, class_id, point_count):")
    for i in sample:
        print(f"  {gstats.elevation_max[i]:.4f}, {gstats.elevation_var[i]:.6f}, "
              f"{gstats.class_id[i]}, {gstats.point_count[i]}")

    assert np.all(np.abs(gstats.elevation_max) < 0.2), "ground elevation should stay near 0"
    assert np.all(gstats.elevation_var < 0.01), "ground height variance should be low"
    assert np.all(gstats.class_id == 0), "ground cells should be class_id 0 (drivable)"


def test_validate_labels_boundaries():
    """Out-of-range label rejected, boundary values accepted (AUDIT-v2 §3.2)."""
    try:
        validate_labels(np.array([99], dtype=np.int64))
        raise AssertionError("expected ValueError for label=99")
    except ValueError as e:
        assert "99" in str(e), f"expected message to name the offending value 99, got: {e}"
        assert f"[0, {NUM_CLASSES})" in str(e), f"expected message to state the valid range, got: {e}"
        print(f"validate_labels(99) correctly raised: {e}")

    validate_labels(np.array([NUM_CLASSES - 1], dtype=np.int64))  # 2, valid boundary - must not raise
    print(f"validate_labels(NUM_CLASSES-1={NUM_CLASSES - 1}): accepted, as expected")

    try:
        validate_labels(np.array([NUM_CLASSES], dtype=np.int64))  # 3, just out of range
        raise AssertionError("expected ValueError for label=NUM_CLASSES")
    except ValueError as e:
        assert str(NUM_CLASSES) in str(e), f"expected message to name the offending value {NUM_CLASSES}, got: {e}"
        assert f"[0, {NUM_CLASSES})" in str(e), f"expected message to state the valid range, got: {e}"
        print(f"validate_labels(NUM_CLASSES={NUM_CLASSES}) correctly raised: {e}")


def test_validate_labels_edge_cases():
    """Empty array, realistic-size all-valid array, negative value."""
    validate_labels(np.array([], dtype=np.int64))  # empty -> must not raise
    print("validate_labels(empty array): accepted, as expected")

    realistic_labels = np.random.default_rng(7).integers(0, NUM_CLASSES, size=5000)
    validate_labels(realistic_labels.astype(np.int64))  # all in-range -> must not raise
    print(f"validate_labels({realistic_labels.size} realistic in-range labels): accepted, as expected")

    try:
        validate_labels(np.array([-1], dtype=np.int64))
        raise AssertionError("expected ValueError for label=-1")
    except ValueError as e:
        assert "-1" in str(e), f"expected message to name the offending value -1, got: {e}"
        print(f"validate_labels(-1) correctly raised: {e}")


def test_nan_z_guard():
    """NaN Z guard: raises instead of silently corrupting elevation stats (AUDIT-v2 §3.4)."""
    nan_pts = np.array([[1.0, 1.0, np.nan, 0.1],
                         [1.0, 1.0, 0.5, 0.1]], dtype=np.float32)
    nan_labels = np.array([0, 0], dtype=np.int64)
    nan_spikes = np.array([1, 1], dtype=np.uint8)
    nan_assignment = grid.assign_cells(nan_pts)
    try:
        aggregate_cells(nan_pts, nan_labels, nan_spikes, nan_assignment)
        raise AssertionError("expected ValueError for NaN Z")
    except ValueError as e:
        assert "NaN" in str(e), f"expected message to mention NaN, got: {e}"
        print(f"NaN-Z correctly raised: {e}")


def test_nan_z_guard_companions():
    """An all-NaN-Z cell, and a NaN-Z point isolated in its own cell with
    otherwise-valid fields, confirming the guard doesn't depend on a
    companion point and doesn't mangle other fields before raising."""
    all_nan_pts = np.array([[2.0, 2.0, np.nan, 0.1],
                             [2.01, 2.01, np.nan, 0.1]], dtype=np.float32)
    all_nan_labels = np.array([0, 0], dtype=np.int64)
    all_nan_spikes = np.array([1, 1], dtype=np.uint8)
    all_nan_assignment = grid.assign_cells(all_nan_pts)
    try:
        aggregate_cells(all_nan_pts, all_nan_labels, all_nan_spikes, all_nan_assignment)
        raise AssertionError("expected ValueError for all-NaN-Z cell")
    except ValueError as e:
        print(f"all-NaN-Z cell correctly raised: {e}")

    isolated_nan_pts = np.array([[80.0, 80.0, np.nan, 0.1]], dtype=np.float32)  # coarse zone, own cell
    isolated_nan_labels = np.array([1], dtype=np.int64)  # valid label
    isolated_nan_spikes = np.array([3], dtype=np.uint8)  # valid spike count
    isolated_nan_assignment = grid.assign_cells(isolated_nan_pts)
    try:
        aggregate_cells(isolated_nan_pts, isolated_nan_labels, isolated_nan_spikes, isolated_nan_assignment)
        raise AssertionError("expected ValueError for isolated NaN-Z point with otherwise-valid fields")
    except ValueError as e:
        assert "0" in str(e), f"expected message to name point index 0, got: {e}"
        print(f"isolated NaN-Z point (valid x/y/label/spike) correctly raised: {e}")


def test_tie_break():
    """Equal class-0/class-2 counts in one cell must favor class 2
    (dynamic object), not class 0 (drivable) (AUDIT-v2 §3.3)."""
    tie_pts = np.array([[1.0, 1.0, 0.1, 0.1],
                         [1.02, 1.02, 0.1, 0.1]], dtype=np.float32)
    tie_labels = np.array([0, 2], dtype=np.int64)
    tie_spikes = np.array([1, 1], dtype=np.uint8)
    tie_assignment = grid.assign_cells(tie_pts)
    tie_stats = aggregate_cells(tie_pts, tie_labels, tie_spikes, tie_assignment)
    assert len(tie_stats) == 1, "expected both points to land in the same cell"
    assert tie_stats.class_id[0] == 2, \
        f"tie-break should favor class 2 (dynamic), got {tie_stats.class_id[0]}"
    print(f"tie-break (class 0 vs class 2, equal counts): resolved to class_id={tie_stats.class_id[0]}")


def test_tie_break_three_class_companion():
    """A 3-class cell where class 1 has a strictly higher count than a
    class-0/class-2 tie must still favor class 1 (count wins over class
    ID) - confirms the reversed argmax didn't just special-case a 2-class
    tie into "class 2 always wins" (AUDIT-v2 test-rebuild Phase 1)."""
    three_way_pts = np.array([[1.0, 1.0, 0.1, 0.1],
                               [1.01, 1.01, 0.1, 0.1],
                               [1.02, 1.02, 0.1, 0.1],
                               [1.03, 1.03, 0.1, 0.1]], dtype=np.float32)
    three_way_labels = np.array([0, 2, 1, 1], dtype=np.int64)  # class1 count=2, class0=1, class2=1
    three_way_spikes = np.array([1, 1, 1, 1], dtype=np.uint8)
    three_way_assignment = grid.assign_cells(three_way_pts)
    three_way_stats = aggregate_cells(three_way_pts, three_way_labels, three_way_spikes, three_way_assignment)
    assert len(three_way_stats) == 1, "expected all four points to land in the same cell"
    assert three_way_stats.class_id[0] == 1, \
        f"count-majority (class 1, count=2) should win over the class0/class2 tie, got {three_way_stats.class_id[0]}"
    print(f"tie-break companion (class1 count=2 beats class0/class2 tie): resolved to class_id={three_way_stats.class_id[0]}")


if __name__ == "__main__":
    test_pole_scene()
    test_ground_plane_scene()
    test_validate_labels_boundaries()
    test_validate_labels_edge_cases()
    test_nan_z_guard()
    test_nan_z_guard_companions()
    test_tie_break()
    test_tie_break_three_class_companion()
    print("test_aggregate.py: all assertions passed")

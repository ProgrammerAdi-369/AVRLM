"""
Regression-parity tests for the packed-1D-int64 dedup optimization
(AUDIT-v2 Phase 7) that replaced np.unique(axis=0) in two places:
  - grid.py:79-90        parent-cell dedup (2 columns -> parent_ix*1024+parent_iy)
  - aggregate.py:101-110 per-point cell dedup (4 columns -> nested *1024/*16 packing)

Both changes are meant to be behavior-invariant performance optimizations.
This file independently reimplements the OLD np.unique(axis=0) logic
inline (not imported from production) and diffs it against the current
packed-key output, for both a realistic scene and the adversarial
boundary-stress-ring. It also directly stress-tests the packing schemes'
injectivity at domain-boundary values.
"""

import numpy as np

from grid import VariableResolutionGrid, OUTER_RADIUS, OUTER_RES, SUBDIV_FACTOR
from radial_filter import prefilter_mask
from aggregate import aggregate_cells
from synthetic_lidar_data import build_scene, generate_boundary_stress_ring

PARENT_KEY_MULT = 1024
SUB_KEY_MULT = 16


def old_parent_dedup(x, y):
    """Reimplements the pre-Phase-7 np.unique(axis=0) parent-cell dedup
    independently of grid.py, using the same floor-division formula
    (grid.py:76-77) but the old row-wise unique instead of the packed key."""
    r_parent_ix = np.floor((x + OUTER_RADIUS) / OUTER_RES).astype(np.int64)
    r_parent_iy = np.floor((y + OUTER_RADIUS) / OUTER_RES).astype(np.int64)
    pairs = np.unique(np.stack([r_parent_ix, r_parent_iy], axis=1), axis=0)
    return set(map(tuple, pairs.tolist()))


def old_cell_dedup(parent_ix, parent_iy, sub_ix, sub_iy):
    """Reimplements the pre-Phase-7 np.unique(axis=0) 4-column cell dedup."""
    cols = np.stack([parent_ix, parent_iy, sub_ix, sub_iy], axis=1)
    rows = np.unique(cols, axis=0)
    return set(map(tuple, rows.tolist()))


def test_packed_key_dedup_parity():
    grid = VariableResolutionGrid()
    for scene_name, (points, labels, spikes) in [
        ("build_scene()", build_scene()),
        ("boundary_stress_ring()", generate_boundary_stress_ring()),
    ]:
        mask = prefilter_mask(points)
        f_points, f_labels, f_spikes = points[mask], labels[mask], spikes[mask]

        # --- Differential test A: grid.py parent-cell dedup ---
        assignment = grid.assign_cells(f_points)
        in_range = assignment.in_range
        new_parent_set = set(zip(
            assignment.parent_ix[in_range].tolist(),
            assignment.parent_iy[in_range].tolist(),
        ))
        old_parent_set = old_parent_dedup(f_points[in_range, 0], f_points[in_range, 1])
        assert new_parent_set == old_parent_set, (
            f"{scene_name}: packed-key parent dedup differs from old np.unique(axis=0) dedup "
            f"(new-only: {new_parent_set - old_parent_set}, old-only: {old_parent_set - new_parent_set})"
        )
        print(f"{scene_name}: parent-cell dedup parity confirmed ({len(new_parent_set)} unique parent cells)")

        # --- Differential test B: aggregate.py per-point cell dedup ---
        stats = aggregate_cells(f_points, f_labels, f_spikes, assignment)
        new_cell_set = set(zip(
            stats.parent_ix.tolist(), stats.parent_iy.tolist(),
            stats.sub_ix.tolist(), stats.sub_iy.tolist(),
        ))
        old_cell_set = old_cell_dedup(
            assignment.parent_ix[in_range], assignment.parent_iy[in_range],
            assignment.sub_ix[in_range], assignment.sub_iy[in_range],
        )
        assert new_cell_set == old_cell_set, (
            f"{scene_name}: packed-key cell dedup differs from old np.unique(axis=0) dedup "
            f"(new-only: {new_cell_set - old_cell_set}, old-only: {old_cell_set - new_cell_set})"
        )
        print(f"{scene_name}: per-point cell dedup parity confirmed ({len(new_cell_set)} unique cells)")


def test_grid_parent_key_packing_injectivity():
    """grid.py parent packing (parent_ix*1024+parent_iy) injectivity at
    the domain boundary. Max parent index along one axis is
    2*OUTER_RADIUS/OUTER_RES = 400 (grid.py:85-86 comment); construct
    pairs at that boundary and confirm round-trip + pairwise
    distinctness."""
    boundary_pairs = [(0, 0), (0, 399), (399, 0), (399, 399), (200, 200)]
    packed_values = []
    for ix, iy in boundary_pairs:
        packed = ix * PARENT_KEY_MULT + iy
        rt_ix, rt_iy = packed // PARENT_KEY_MULT, packed % PARENT_KEY_MULT
        assert (rt_ix, rt_iy) == (ix, iy), f"round-trip failed for ({ix},{iy}): got ({rt_ix},{rt_iy})"
        packed_values.append(packed)
    assert len(set(packed_values)) == len(packed_values), \
        f"parent-key collision among boundary pairs: {list(zip(boundary_pairs, packed_values))}"
    print(f"grid.py parent-key packing: round-trip + distinctness confirmed for {boundary_pairs}")


def test_aggregate_cell_key_packing_injectivity():
    """aggregate.py cell packing
    (((parent_ix*1024+parent_iy)*16+(sub_ix+1))*16+(sub_iy+1)) injectivity,
    covering the sentinel case (sub_ix=sub_iy=-1, coarse cells) and the
    max sub-range (0..SUBDIV_FACTOR-1), at adjacent parent-cell
    boundaries."""
    max_sub = SUBDIV_FACTOR - 1  # 9
    cell_tuples = [
        (0, 0, -1, -1),               # coarse cell at parent (0,0)
        (0, 0, max_sub, max_sub),     # fine sub-cell at the far corner of parent (0,0)
        (0, 1, -1, -1),               # coarse cell at the adjacent parent (0,1)
        (0, 1, 0, 0),                 # fine sub-cell at the near corner of parent (0,1)
        (1, 0, -1, -1),               # coarse cell at adjacent parent (1,0)
    ]
    cell_packed_values = []
    for parent_ix, parent_iy, sub_ix, sub_iy in cell_tuples:
        packed = ((parent_ix * PARENT_KEY_MULT + parent_iy) * SUB_KEY_MULT + (sub_ix + 1)) * SUB_KEY_MULT + (sub_iy + 1)
        cell_packed_values.append(packed)
    assert len(set(cell_packed_values)) == len(cell_packed_values), \
        f"cell-key collision among boundary tuples: {list(zip(cell_tuples, cell_packed_values))}"
    print(f"aggregate.py cell-key packing: distinctness confirmed for {cell_tuples}")


if __name__ == "__main__":
    test_packed_key_dedup_parity()
    test_grid_parent_key_packing_injectivity()
    test_aggregate_cell_key_packing_injectivity()
    print("test_packed_keys.py: all assertions passed")

"""
Team-facing handoff. generate_2_5d_grid(points, labels, spikes) is the
fixed-signature function other modules integrate against (CLAUDE.md
section 2). It takes no explicit state argument, so event-driven
persistence (Task 2) is kept in a module-level GridState singleton here -
see implementation_plan_next_tasks.md Task 3 for the reasoning.
"""

import numpy as np

from grid_state import GridState
from aggregate import validate_labels, coerce_spikes

_state = GridState()


def validate_inputs(points, labels, spikes) -> None:
    """Shape/length sanity check at the true pipeline entry point, run
    before the narrower per-field checks (validate_labels/coerce_spikes)
    so a shape mismatch fails here with a clear message instead of a
    confusing error deeper in the stack."""
    points = np.asarray(points)
    labels = np.asarray(labels)
    spikes = np.asarray(spikes)

    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(
            f"points must have shape (N, 4), got {points.shape}"
        )
    n = points.shape[0]
    if labels.shape != (n,):
        raise ValueError(
            f"labels must have shape ({n},) to match points, got {labels.shape}"
        )
    if spikes.shape != (n,):
        raise ValueError(
            f"spikes must have shape ({n},) to match points, got {spikes.shape}"
        )

# Bytes-per-record estimate for memory_metrics, from CellRecord's actual
# stored field dtypes (not a guessed constant): is_fine(bool,1) +
# center_x/y(float64,8 each) + elevation_max/var(float64,8 each) +
# class_id(int64,8) + point_count(int64,8).
_BYTES_PER_RECORD = 1 + 8 + 8 + 8 + 8 + 8 + 8

_DENSE_Z_RANGE_M = 3.0  # assumed UGV operational height band, see plan


def generate_2_5d_grid(points, labels, spikes):
    """Returns the structured 2.5D grid: a list of
    (cell_key, CellRecord) pairs for every currently-known cell
    (accumulated across all frames seen so far via the internal
    event-driven GridState), reflecting this frame's update."""
    validate_inputs(points, labels, spikes)
    validate_labels(labels)
    spikes = coerce_spikes(spikes)
    _state.update(points, labels, spikes)
    return _state.snapshot()


def memory_metrics(state: GridState = None) -> dict:
    """Active cell count, estimated sparse bytes, and a comparison against
    a naive dense 3D voxel grid at 5cm resolution over the full 200m x 200m
    footprint (assumed 3m height band - see plan Task 3)."""
    if state is None:
        state = _state

    active_cell_count = len(state._cells)
    estimated_sparse_bytes = active_cell_count * _BYTES_PER_RECORD

    from grid import OUTER_RADIUS, INNER_RES
    span = 2 * OUTER_RADIUS
    voxels_xy = (span / INNER_RES) ** 2
    voxels_z = _DENSE_Z_RANGE_M / INNER_RES
    naive_dense_bytes = voxels_xy * voxels_z * 1  # 1 byte/voxel, minimal occupancy

    savings_ratio = naive_dense_bytes / estimated_sparse_bytes if estimated_sparse_bytes > 0 else float("inf")

    return {
        "active_cell_count": active_cell_count,
        "estimated_sparse_bytes": estimated_sparse_bytes,
        "naive_dense_bytes": naive_dense_bytes,
        "savings_ratio": savings_ratio,
    }

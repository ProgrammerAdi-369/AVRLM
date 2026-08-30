"""
Per-cell elevation and semantic-class aggregation. Consumes a
grid.CellAssignment (per-point cell indices) plus the raw points/labels/
spikes and produces one row per occupied cell. This is also the schema
Task 3's generate_2_5d_grid() hands back to teammates - see CellStats.

Vectorized only (NumPy grouping via np.unique/np.bincount), per CLAUDE.md -
never a Python loop over points.
"""

import warnings

import numpy as np

from grid import parent_cell_center, sub_cell_center

NUM_CLASSES = 3  # fixed: 0=drivable, 1=static obstacle, 2=dynamic object


def validate_labels(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> None:
    """Raises ValueError if any label falls outside [0, num_classes)."""
    if labels.size == 0:
        return
    lo, hi = int(labels.min()), int(labels.max())
    if lo < 0 or hi >= num_classes:
        bad = labels[(labels < 0) | (labels >= num_classes)]
        raise ValueError(
            f"labels contains value {int(bad[0])}, outside valid range "
            f"[0, {num_classes})"
        )


def coerce_spikes(spikes: np.ndarray) -> np.ndarray:
    """Coerces spikes to the documented integer-count contract
    (uint8/int32). Real SNN output (spk_rec.sum(dim=0)) legitimately
    arrives as float; round to the nearest integer and warn once (not
    raise) when rounding actually changes a value by more than a tiny
    epsilon - this is expected input shape, not malformed input."""
    spikes = np.asarray(spikes)
    if np.issubdtype(spikes.dtype, np.integer):
        return spikes
    rounded = np.round(spikes)
    if np.any(np.abs(spikes - rounded) > 1e-6):
        warnings.warn(
            "spikes contains non-integer values; rounding to nearest "
            "integer count per interface contract (uint8/int32)",
            stacklevel=2,
        )
    return rounded.astype(np.int64)


class CellStats:
    """One row per occupied cell (fine or coarse), parallel arrays."""

    def __init__(self, parent_ix, parent_iy, sub_ix, sub_iy, is_fine,
                 center_x, center_y, elevation_max, elevation_var,
                 class_id, point_count, spike_sum):
        self.parent_ix = parent_ix
        self.parent_iy = parent_iy
        self.sub_ix = sub_ix
        self.sub_iy = sub_iy
        self.is_fine = is_fine
        self.center_x = center_x
        self.center_y = center_y
        self.elevation_max = elevation_max
        self.elevation_var = elevation_var
        self.class_id = class_id
        self.point_count = point_count
        self.spike_sum = spike_sum

    def __len__(self):
        return self.parent_ix.shape[0]


def aggregate_cells(points: np.ndarray, labels: np.ndarray, spikes: np.ndarray, assignment) -> CellStats:
    mask = assignment.in_range
    z = points[mask, 2]
    nan_z = np.isnan(z)
    if np.any(nan_z):
        bad_indices = np.where(mask)[0][nan_z]
        raise ValueError(
            f"points[:, 2] (Z) contains NaN at point indices {bad_indices.tolist()}"
        )
    lbl = labels[mask].astype(np.int64)
    spk = spikes[mask].astype(np.float64)

    parent_ix = assignment.parent_ix[mask]
    parent_iy = assignment.parent_iy[mask]
    sub_ix = assignment.sub_ix[mask]
    sub_iy = assignment.sub_iy[mask]
    is_fine = assignment.is_fine[mask]

    n_points = z.shape[0]
    if n_points == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        empty_b = np.empty(0, dtype=bool)
        return CellStats(empty_i, empty_i, empty_i, empty_i, empty_b,
                          empty_f, empty_f, empty_f, empty_f, empty_i, empty_i, empty_f)

    # Dedup via a packed 1D int64 key instead of np.unique(axis=0) on 4
    # columns - a 1D sort is materially cheaper than the row-wise lexsort
    # axis=0 takes (profiled: np.unique(axis=0)'s argsort was the second-
    # largest cost in a full grid_state.update(), see
    # Reports/AUDIT-v2.md Phase 7). PARENT_KEY_MULT (>400, grid.py's
    # parent-index range) and SUB_KEY_MULT (>10, sub_ix/sub_iy's -1..9
    # range shifted to 0..10) both use power-of-2 margins.
    PARENT_KEY_MULT = 1024
    SUB_KEY_MULT = 16
    cell_packed = ((parent_ix * PARENT_KEY_MULT + parent_iy) * SUB_KEY_MULT + (sub_ix + 1)) * SUB_KEY_MULT + (sub_iy + 1)
    _, first_idx, group_id = np.unique(cell_packed, return_index=True, return_inverse=True)
    group_id = group_id.reshape(-1)
    n_cells = first_idx.shape[0]

    # elevation_max: vectorized scatter-max
    elevation_max = np.full(n_cells, -np.inf, dtype=np.float64)
    np.maximum.at(elevation_max, group_id, z)

    # elevation_var via sum/sum-of-squares per group (fully vectorized bincount)
    counts = np.bincount(group_id, minlength=n_cells)
    sum_z = np.bincount(group_id, weights=z, minlength=n_cells)
    sum_z2 = np.bincount(group_id, weights=z ** 2, minlength=n_cells)
    mean_z = sum_z / counts
    elevation_var = np.maximum(sum_z2 / counts - mean_z ** 2, 0.0)

    # majority class vote via bincount over (group, class) flattened index.
    # Ties favor the HIGHEST class ID (dynamic object over static obstacle
    # over drivable terrain) - reverse columns before argmax so np.argmax's
    # first-match tie-break lands on the highest original class index, then
    # map the reversed index back. See Reports/AUDIT-v2.md §3.3/§8.3.
    class_flat_counts = np.bincount(group_id * NUM_CLASSES + lbl, minlength=n_cells * NUM_CLASSES)
    class_counts = class_flat_counts.reshape(n_cells, NUM_CLASSES)
    class_id = NUM_CLASSES - 1 - np.argmax(class_counts[:, ::-1], axis=1)

    spike_sum = np.bincount(group_id, weights=spk, minlength=n_cells)

    cell_parent_ix = parent_ix[first_idx]
    cell_parent_iy = parent_iy[first_idx]
    cell_sub_ix = sub_ix[first_idx]
    cell_sub_iy = sub_iy[first_idx]
    cell_is_fine = cell_sub_ix >= 0

    center_x = np.empty(n_cells, dtype=np.float64)
    center_y = np.empty(n_cells, dtype=np.float64)

    coarse = ~cell_is_fine
    center_x[coarse], center_y[coarse] = parent_cell_center(cell_parent_ix[coarse], cell_parent_iy[coarse])
    center_x[cell_is_fine], center_y[cell_is_fine] = sub_cell_center(
        cell_parent_ix[cell_is_fine], cell_parent_iy[cell_is_fine],
        cell_sub_ix[cell_is_fine], cell_sub_iy[cell_is_fine],
    )

    return CellStats(cell_parent_ix, cell_parent_iy, cell_sub_ix, cell_sub_iy, cell_is_fine,
                      center_x, center_y, elevation_max, elevation_var,
                      class_id, counts, spike_sum)

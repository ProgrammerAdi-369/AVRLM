"""
Per-cell elevation and semantic-class aggregation. Consumes a
grid.CellAssignment (per-point cell indices) plus the raw points/labels/
spikes and produces one row per occupied cell. This is also the schema
Task 3's generate_2_5d_grid() hands back to teammates - see CellStats.

Vectorized only (NumPy grouping via np.unique/np.bincount), per CLAUDE.md -
never a Python loop over points.
"""

import numpy as np

from grid import parent_cell_center, sub_cell_center

NUM_CLASSES = 3  # fixed: 0=drivable, 1=static obstacle, 2=dynamic object


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

    cell_key = np.stack([parent_ix, parent_iy, sub_ix, sub_iy], axis=1)
    unique_cells, group_id = np.unique(cell_key, axis=0, return_inverse=True)
    group_id = group_id.reshape(-1)
    n_cells = unique_cells.shape[0]

    # elevation_max: vectorized scatter-max
    elevation_max = np.full(n_cells, -np.inf, dtype=np.float64)
    np.maximum.at(elevation_max, group_id, z)

    # elevation_var via sum/sum-of-squares per group (fully vectorized bincount)
    counts = np.bincount(group_id, minlength=n_cells)
    sum_z = np.bincount(group_id, weights=z, minlength=n_cells)
    sum_z2 = np.bincount(group_id, weights=z ** 2, minlength=n_cells)
    mean_z = sum_z / counts
    elevation_var = np.maximum(sum_z2 / counts - mean_z ** 2, 0.0)

    # majority class vote via bincount over (group, class) flattened index
    class_flat_counts = np.bincount(group_id * NUM_CLASSES + lbl, minlength=n_cells * NUM_CLASSES)
    class_counts = class_flat_counts.reshape(n_cells, NUM_CLASSES)
    class_id = np.argmax(class_counts, axis=1)

    spike_sum = np.bincount(group_id, weights=spk, minlength=n_cells)

    cell_parent_ix = unique_cells[:, 0]
    cell_parent_iy = unique_cells[:, 1]
    cell_sub_ix = unique_cells[:, 2]
    cell_sub_iy = unique_cells[:, 3]
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

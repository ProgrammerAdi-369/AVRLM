"""
Two-level (50cm / 5cm) hierarchical grid. See CLAUDE.md section 4 for the
design rationale (why this isn't a textbook 4-way quadtree) and the exact
alignment formulas this implements.
"""

import numpy as np

INNER_RES = 0.05  # 5cm
OUTER_RES = 0.50  # 50cm
INNER_RADIUS = 10.0
OUTER_RADIUS = 100.0
SUBDIV_FACTOR = int(round(OUTER_RES / INNER_RES))  # 10, computed not hardcoded


class CellAssignment:
    """Per-point cell assignment result. Sub-cell fields use -1 as the
    sentinel for points whose parent cell was not subdivided. `in_range`
    is False for points outside the [-100m, 100m] grid extent - those
    points get -1 for every index field and are excluded from cell
    grouping. Real callers are expected to pre-filter with
    radial_filter.prefilter_mask() before calling assign_cells(); the
    in_range check here is a defensive fallback, not the primary contract."""

    def __init__(self, parent_ix, parent_iy, is_fine, sub_ix, sub_iy, in_range):
        self.parent_ix = parent_ix
        self.parent_iy = parent_iy
        self.is_fine = is_fine
        self.sub_ix = sub_ix
        self.sub_iy = sub_iy
        self.in_range = in_range


def parent_cell_center(parent_ix, parent_iy):
    """World-space (x, y) center of a 50cm parent cell from its index."""
    center_x = (parent_ix + 0.5) * OUTER_RES - OUTER_RADIUS
    center_y = (parent_iy + 0.5) * OUTER_RES - OUTER_RADIUS
    return center_x, center_y


def sub_cell_center(parent_ix, parent_iy, sub_ix, sub_iy):
    """World-space (x, y) center of a 5cm sub-cell from its parent + sub index."""
    parent_origin_x = parent_ix * OUTER_RES - OUTER_RADIUS
    parent_origin_y = parent_iy * OUTER_RES - OUTER_RADIUS
    center_x = parent_origin_x + (sub_ix + 0.5) * INNER_RES
    center_y = parent_origin_y + (sub_iy + 0.5) * INNER_RES
    return center_x, center_y


class VariableResolutionGrid:
    """Base 50cm grid over [-100m, 100m]; any 50cm cell whose center falls
    within 10m of the origin is treated as subdivided into a 10x10 array
    of 5cm sub-cells."""

    def assign_cells(self, points: np.ndarray) -> CellAssignment:
        x = points[:, 0]
        y = points[:, 1]
        n = x.shape[0]

        # Defensive fallback: points outside the grid's own [-100, 100]
        # extent get no valid parent index. Primary contract is still that
        # the caller pre-filters with radial_filter.prefilter_mask() first.
        in_range = (np.abs(x) <= OUTER_RADIUS) & (np.abs(y) <= OUTER_RADIUS)

        parent_ix = np.full(n, -1, dtype=np.int64)
        parent_iy = np.full(n, -1, dtype=np.int64)
        is_fine = np.zeros(n, dtype=bool)
        sub_ix = np.full(n, -1, dtype=np.int64)
        sub_iy = np.full(n, -1, dtype=np.int64)

        if not np.any(in_range):
            return CellAssignment(parent_ix, parent_iy, is_fine, sub_ix, sub_iy, in_range)

        rx = x[in_range]
        ry = y[in_range]
        r_parent_ix = np.floor((rx + OUTER_RADIUS) / OUTER_RES).astype(np.int64)
        r_parent_iy = np.floor((ry + OUTER_RADIUS) / OUTER_RES).astype(np.int64)

        # Decide subdivision once per unique parent cell (never per point),
        # so a single cell can't be half-subdivided.
        parent_pairs = np.stack([r_parent_ix, r_parent_iy], axis=1)
        unique_pairs, inverse = np.unique(parent_pairs, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)

        unique_center_x, unique_center_y = parent_cell_center(unique_pairs[:, 0], unique_pairs[:, 1])
        unique_subdivide = np.sqrt(unique_center_x ** 2 + unique_center_y ** 2) <= INNER_RADIUS

        r_is_fine = unique_subdivide[inverse]

        r_sub_ix = np.full(rx.shape[0], -1, dtype=np.int64)
        r_sub_iy = np.full(rx.shape[0], -1, dtype=np.int64)

        parent_origin_x = r_parent_ix[r_is_fine] * OUTER_RES - OUTER_RADIUS
        parent_origin_y = r_parent_iy[r_is_fine] * OUTER_RES - OUTER_RADIUS
        local_x = rx[r_is_fine] - parent_origin_x
        local_y = ry[r_is_fine] - parent_origin_y

        r_sub_ix[r_is_fine] = np.floor(local_x / INNER_RES).astype(np.int64)
        r_sub_iy[r_is_fine] = np.floor(local_y / INNER_RES).astype(np.int64)

        parent_ix[in_range] = r_parent_ix
        parent_iy[in_range] = r_parent_iy
        is_fine[in_range] = r_is_fine
        sub_ix[in_range] = r_sub_ix
        sub_iy[in_range] = r_sub_iy

        return CellAssignment(parent_ix, parent_iy, is_fine, sub_ix, sub_iy, in_range)

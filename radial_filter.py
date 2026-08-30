"""
Radial filter: splits ego-centric LiDAR points into the 0-10m (fine, 5cm
grid) and 10-100m (coarse, 50cm grid) zones. See CLAUDE.md section 3-4 for
the interface contract and boundary convention this implements.
"""

import numpy as np

from grid import INNER_RADIUS, OUTER_RADIUS


def radial_filter(points: np.ndarray):
    """
    points: (N, 4) float32, ego-centric meters (X, Y, Z, Intensity).
    Returns boolean masks (not copies), so labels/spikes can be indexed
    the same way by the caller.

    inner_mask: r <= 10.0
    outer_mask: 10.0 < r <= 100.0
    Points with r > 100.0 satisfy neither mask and are effectively dropped.

    NON-AUTHORITATIVE: this per-point split is a diagnostic/reporting
    utility only. Which resolution a point's data is actually stored at is
    decided by grid.py's per-parent-cell-center rule (see CLAUDE.md
    section 4) - use prefilter_mask() below to gate what reaches grid.py.
    """
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    inner_mask = r <= INNER_RADIUS
    outer_mask = (r > INNER_RADIUS) & (r <= OUTER_RADIUS)
    return inner_mask, outer_mask


def prefilter_mask(points: np.ndarray) -> np.ndarray:
    """
    The actual pre-filter for grid ingestion: True for points with
    r <= 100.0. grid.assign_cells() expects to be called with points
    already reduced by this mask (see CLAUDE.md/implementation plan Task 0).
    """
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    return r <= OUTER_RADIUS

"""
Event-driven persistent grid state. A cell's cached record is overwritten
when either (a) this frame's points for that cell have spike_sum > 0, or
(b) the cell has never been recorded before (first sight - establishes a
baseline regardless of spike status, so a cold-start frame doesn't drop
the parts of the scene that simply weren't spiking yet). Once a cell has a
cached record, later frames only update it when it spikes again. See
CLAUDE.md section 2/4, implementation_plan_next_tasks.md Task 2, and
debug_guide_cell_count_discrepancy.md (the investigation that found the
original spike-only condition silently discarded ~97% of a cold-start
scene).

Storage: dict keyed by (parent_ix, parent_iy, sub_ix, sub_iy) -> CellRecord.
Chosen over a dense array because this module's output format is already
sparse/active-cells-only (CLAUDE.md section 4), and a dense array sized for
the full 200m x 200m extent at 5cm would be enormous and almost entirely
empty (fine cells only exist inside the 10m radius).
"""

from radial_filter import prefilter_mask
from grid import VariableResolutionGrid
from aggregate import aggregate_cells


class CellRecord:
    __slots__ = ("is_fine", "center_x", "center_y", "elevation_max",
                 "elevation_var", "class_id", "point_count")

    def __init__(self, is_fine, center_x, center_y, elevation_max, elevation_var, class_id, point_count):
        self.is_fine = is_fine
        self.center_x = center_x
        self.center_y = center_y
        self.elevation_max = elevation_max
        self.elevation_var = elevation_var
        self.class_id = class_id
        self.point_count = point_count


class GridState:
    def __init__(self):
        self._cells = {}
        self._grid = VariableResolutionGrid()

    def update(self, points, labels, spikes):
        """Runs the prefilter -> assign_cells -> aggregate_cells pipeline
        on this frame's points, then commits cells whose spike_sum > 0
        this frame, plus any cell seen for the first time (cold-start
        baseline - see module docstring). Returns the set of cell keys
        touched this frame plus their spike_sum, for event-driven
        bookkeeping."""
        mask = prefilter_mask(points)
        f_points, f_labels, f_spikes = points[mask], labels[mask], spikes[mask]
        assignment = self._grid.assign_cells(f_points)
        stats = aggregate_cells(f_points, f_labels, f_spikes, assignment)

        touched = {}
        for i in range(len(stats)):
            key = (int(stats.parent_ix[i]), int(stats.parent_iy[i]),
                   int(stats.sub_ix[i]), int(stats.sub_iy[i]))
            spike_sum = float(stats.spike_sum[i])
            touched[key] = spike_sum
            if spike_sum > 0 or key not in self._cells:
                self._cells[key] = CellRecord(
                    bool(stats.is_fine[i]), float(stats.center_x[i]), float(stats.center_y[i]),
                    float(stats.elevation_max[i]), float(stats.elevation_var[i]),
                    int(stats.class_id[i]), int(stats.point_count[i]),
                )
        return touched

    def snapshot(self):
        """Returns the full current map (every cached cell, old and
        freshly updated this frame) as a list of (key, CellRecord) pairs."""
        return list(self._cells.items())

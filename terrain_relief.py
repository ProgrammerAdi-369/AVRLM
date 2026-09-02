"""Near-field terrain relief: heightfield synthesis + colorization for the
radar app's UI. Plain functions only (no PyQt6) except `to_qimage`, so the
math here is testable without a Qt event loop.

`sample_heightfield` is a PLACEHOLDER, not real sensor data -- same spirit
as `RadarView2D._make_ground_clutter` in radar_view_2d.py. engine_adapter's
Frame carries no elevation/point data (TrackedObject.z is hardcoded 0.0
everywhere it's constructed), so there is nothing real to sample yet. If
this needs to become real, the cheapest path is wiring grid.py/aggregate.py/
grid_state.py's per-cell `elevation_max` into engine_adapter.py and reading
that here instead.
"""
import math
import warnings

import numpy as np
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QRadialGradient
from PyQt6.QtWidgets import QWidget

from radar_view_2d import ACCENT, LABEL_COLORS, RED, GREY

_ANCHORS = np.array([
    [0, 128, 128],    # teal
    [60, 170, 90],    # green
    [230, 175, 60],   # amber
    [200, 60, 40],    # red
], dtype=np.float64)


def sample_heightfield(ugv_x, ugv_y, heading_deg=0.0, radius_m=10.0, resolution_m=0.10):
    """Synthetic near-field heightmap, PLACEHOLDER data (see module docstring).

    Returns an (H, W) float64 array covering a 2*radius_m x 2*radius_m box
    centered on (ugv_x, ugv_y). Row 0 is the panel's forward/top edge,
    column 0 is its left edge -- matching polar_to_px's "0 deg/up = forward"
    convention in radar_view_2d.py. At heading_deg=0 this is plain north-up
    (row 0 = north, column 0 = west). heading_deg rotates the sampled box
    to match radar_view_2d.py's heading-up display (_draw_tracks subtracts
    ugv_heading_deg from bearing; this applies the inverse, adding it to the
    local offsets, so the panel's "up" always samples what's ahead of the
    UGV). Sampling is done in world coordinates offset by the UGV's real
    (dead-reckoned) position, so the terrain visibly scrolls as the UGV
    moves.
    """
    n = int(round(2 * radius_m / resolution_m))
    cols = np.arange(n)
    rows = np.arange(n)
    local_dx = (cols - n / 2 + 0.5) * resolution_m
    local_dy = (n / 2 - rows - 0.5) * resolution_m
    ldx, ldy = np.meshgrid(local_dx, local_dy)  # ldx varies along columns, ldy along rows

    hdg = math.radians(heading_deg)
    cos_h, sin_h = math.cos(hdg), math.sin(hdg)
    dx_world = ldx * cos_h + ldy * sin_h
    dy_world = -ldx * sin_h + ldy * cos_h
    wx, wy = ugv_x + dx_world, ugv_y + dy_world

    h = (0.6 * np.sin(wx * 0.15 + wy * 0.10)
         + 0.3 * np.cos(wx * 0.08 - wy * 0.20)
         + 0.10 * np.sin(wx * 1.3 + wy * 0.7)
         + 0.05 * np.cos(wx * 0.4 - wy * 1.1))
    return h


def world_offset_to_panel_px(local_dx, local_dy, radius_m, panel_w_px, panel_h_px):
    """Inverse of sample_heightfield's row/col mapping: a display-frame
    offset (already rotated by heading, same frame as local_dx/local_dy
    above) -> (px, py) panel pixel coordinates."""
    frac_x = (local_dx + radius_m) / (2 * radius_m)
    frac_y = (radius_m - local_dy) / (2 * radius_m)
    return frac_x * panel_w_px, frac_y * panel_h_px


def bin_cells_to_heightfield(cells, heading_deg=0.0, radius_m=10.0, resolution_m=0.10):
    """Real grid-engine cells (list of (cell_key, CellRecord) pairs from
    generate_2_5d_grid) -> (H, W) float64 heightfield, NaN where no cell
    lands. Uses every cell regardless of class -- ground elevation is
    exactly what a terrain heightmap wants, unlike the object-clustering
    path which excludes drivable cells.

    CellRecord.center_x/y are already ego-centric (grid.py's UGV-at-origin
    contract), so unlike sample_heightfield this needs no ugv_x/y offset --
    only the same heading-up rotation _draw_markers already applies to
    track positions (the inverse of sample_heightfield's forward rotation),
    then the inverse of sample_heightfield's row/col index formulas.
    """
    n = int(round(2 * radius_m / resolution_m))
    height = np.full((n, n), np.nan, dtype=np.float64)
    if not cells:
        return height

    ex = np.array([c.center_x for _, c in cells])
    ey = np.array([c.center_y for _, c in cells])
    ez = np.array([c.elevation_max for _, c in cells])

    in_range = np.hypot(ex, ey) <= radius_m
    if not np.any(in_range):
        return height
    ex, ey, ez = ex[in_range], ey[in_range], ez[in_range]

    hdg = math.radians(heading_deg)
    cos_h, sin_h = math.cos(hdg), math.sin(hdg)
    local_dx = ex * cos_h - ey * sin_h
    local_dy = ex * sin_h + ey * cos_h

    col = np.round(local_dx / resolution_m + n / 2 - 0.5).astype(np.int64)
    row = np.round(n / 2 - local_dy / resolution_m - 0.5).astype(np.int64)
    in_bounds = (row >= 0) & (row < n) & (col >= 0) & (col < n)
    row, col, ez = row[in_bounds], col[in_bounds], ez[in_bounds]

    np.fmax.at(height, (row, col), ez)   # fmax ignores NaN, unlike maximum
    return height


def fill_gaps(height_array):
    """Iterative nearest-neighbor-ish gap fill via nanmean over shifted
    copies -- needed once real (sparse) cell data replaces the synthetic
    source, which never produced NaN. No-op if there's nothing to fill.
    """
    if not np.any(np.isnan(height_array)):
        return height_array
    filled = height_array.copy()
    for _ in range(4):
        nan_mask = np.isnan(filled)
        if not np.any(nan_mask):
            break
        neighbors = [filled]
        for shift in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbors.append(np.roll(filled, shift=shift, axis=(0, 1)))
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN neighborhood in an early pass
            neighbor_mean = np.nanmean(np.stack(neighbors, axis=0), axis=0)
        filled = np.where(nan_mask, neighbor_mean, filled)
    remaining = np.isnan(filled)
    if np.any(remaining):
        fallback = np.nanmean(filled)
        filled = np.where(remaining, 0.0 if np.isnan(fallback) else fallback, filled)
    return filled


def colorize(height_array, vmin=-1.2, vmax=1.2):
    """(H, W) height array -> (H, W, 3) uint8 RGB via fixed anchor palette.

    vmin/vmax are placeholders (headroom over the synthetic function's
    ~+/-1.05 range) to revisit once real sensor elevation ranges are known.
    """
    t = np.clip((height_array - vmin) / (vmax - vmin), 0.0, 1.0)
    n_segs = len(_ANCHORS) - 1
    scaled = t * n_segs
    seg = np.clip(scaled.astype(np.int64), 0, n_segs - 1)
    frac = (scaled - seg)[..., None]
    c0 = _ANCHORS[seg]
    c1 = _ANCHORS[seg + 1]
    rgb = c0 + (c1 - c0) * frac
    return np.clip(rgb, 0, 255).astype(np.uint8)


def apply_hillshade(rgb_array, height_array, resolution_m,
                     light_azimuth_deg=315, light_altitude_deg=45):
    """Multiply a directional-lighting brightness factor into rgb_array."""
    d_row, d_col = np.gradient(height_array, resolution_m)
    dzdx = d_col
    dzdy = -d_row

    normal_len = np.sqrt(dzdx ** 2 + dzdy ** 2 + 1.0)
    nx, ny, nz = -dzdx / normal_len, -dzdy / normal_len, 1.0 / normal_len

    az = np.radians(light_azimuth_deg)
    alt = np.radians(light_altitude_deg)
    lx = np.sin(az) * np.cos(alt)
    ly = np.cos(az) * np.cos(alt)
    lz = np.sin(alt)

    brightness = nx * lx + ny * ly + nz * lz
    brightness = np.clip(brightness + 0.5, 0.5, 1.15)

    shaded = rgb_array.astype(np.float64) * brightness[..., None]
    return np.clip(shaded, 0, 255).astype(np.uint8)


def to_qimage(rgb_array):
    """(H, W, 3) uint8 -> QImage.Format_RGB888, safe to use after this returns."""
    rgb_array = np.ascontiguousarray(rgb_array)
    h, w, _ = rgb_array.shape
    image = QImage(rgb_array.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return image.copy()


_BG = QColor(5, 7, 13)  # matches radar_view_2d.py's BG constant


class TerrainReliefView(QWidget):
    """Near-field (0-10m) terrain relief panel. See sample_heightfield's
    docstring: the underlying heightmap is placeholder/demo data."""

    def __init__(self, radius_m=10.0, parent=None):
        super().__init__(parent)
        self.radius_m = radius_m
        self.setFixedSize(280, 280)
        self._frame = None
        self._cached_image = None
        self._tick_count = 0

    def update_frame(self, frame):
        self._frame = frame
        self._tick_count += 1
        # Recompute at ~6Hz (30Hz tick / 5) instead of every tick -- the
        # heightfield/color/hillshade pipeline is too heavy to redo at 30Hz.
        if self._tick_count % 5 == 0:
            if frame.live_cells is not None:
                height = bin_cells_to_heightfield(frame.live_cells,
                                                   heading_deg=frame.ugv_heading_deg,
                                                   radius_m=self.radius_m)
                height = fill_gaps(height)
                # Real elevations are non-negative (~0-3m for this scene),
                # unlike the old +/-1.2m sinusoidal placeholder range --
                # placeholder range to revisit once more scenes are profiled.
                rgb = colorize(height, vmin=0.0, vmax=3.0)
            else:
                height = sample_heightfield(frame.ugv_x, frame.ugv_y,
                                             heading_deg=frame.ugv_heading_deg,
                                             radius_m=self.radius_m)
                rgb = colorize(height, vmin=-1.2, vmax=1.2)
            shaded = apply_hillshade(rgb, height, resolution_m=0.10)
            self._cached_image = to_qimage(shaded)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        if self._cached_image is None:
            p.fillRect(self.rect(), _BG)
            p.end()
            return
        p.drawImage(self.rect(), self._cached_image)
        self._draw_markers(p)
        self._draw_ugv_marker(p)
        p.end()

    def _draw_ugv_marker(self, p):
        # The panel is heading-up (same convention as radar_view_2d.py's
        # _draw_tracks, which subtracts ugv_heading_deg from bearing), so
        # -- exactly like RadarView2D._draw_ugv -- the UGV glyph stays
        # FIXED at the panel's center, pointing up; it is the terrain/
        # markers that rotate around it, not the glyph itself.
        cx, cy = self.width() / 2, self.height() / 2

        glow = QRadialGradient(cx, cy, 14)
        glow.setColorAt(0.0, QColor(0, 255, 200, 130))
        glow.setColorAt(1.0, QColor(0, 255, 200, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(QPointF(cx, cy), 14, 14)

        path = QPainterPath()
        s = 6
        path.moveTo(cx, cy - s)
        path.lineTo(cx - s * 0.8, cy + s * 0.7)
        path.lineTo(cx + s * 0.8, cy + s * 0.7)
        path.closeSubpath()
        p.setPen(QPen(ACCENT, 1.5))
        p.setBrush(QBrush(QColor(0, 40, 35)))
        p.drawPath(path)

    def _draw_markers(self, p):
        # Runs every paint call (unthrottled), independent of the
        # background image's ~6Hz recompute rate, so markers track object
        # motion at full frame rate.
        frame = self._frame
        hdg = math.radians(frame.ugv_heading_deg)
        cos_h, sin_h = math.cos(hdg), math.sin(hdg)
        w, h = self.width(), self.height()
        size = max(10, min(14, w * 0.045))

        for trk in frame.tracks:
            dx_world = trk.x - frame.ugv_x
            dy_world = trk.y - frame.ugv_y
            if math.hypot(dx_world, dy_world) > self.radius_m:
                continue

            local_dx = dx_world * cos_h - dy_world * sin_h
            local_dy = dx_world * sin_h + dy_world * cos_h
            px, py = world_offset_to_panel_px(local_dx, local_dy, self.radius_m, w, h)

            base = LABEL_COLORS.get(trk.label, GREY)
            color = RED if trk.threat else base

            p.setPen(QPen(color.darker(150), 1))
            p.setBrush(color)
            p.drawEllipse(QPointF(px, py), size / 2, size / 2)

            if trk.threat:
                p.setPen(QPen(RED, 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(px, py), size / 2 + 3, size / 2 + 3)

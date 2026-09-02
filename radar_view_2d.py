"""
radar_view_2d.py
------------------------------------------------------------------
Flat, top-down 2D "PPI" (Plan Position Indicator) radar view -- the
classic circular sweep-radar screen, not a tilted 3D perspective.

This is a drop-in replacement for RadarView3D in avrlm_radar_app.py.
It exposes the same public surface (`update_frame(frame)`, same
constructor signature) so swapping it in is a two-line change in
avrlm_radar_app.py:

    from radar_view_2d import RadarView2D
    ...
    self.radar = RadarView2D()

Everything is drawn with plain QPainter in an overridden paintEvent
-- no OpenGL. That's the deliberate choice: the target look (compass
bezel, radial sweep glow, neon bloom on the object markers) is a
2D-canvas effect, not a 3D-perspective effect. Chasing it inside a
tilted GLViewWidget fights the rendering model the whole way; a flat
QPainter scene gets you there directly, and it's the standard
approach real PPI-style radar UIs use.

Coordinate convention (unchanged from RadarView3D): heading-up
display, UGV fixed at the center of the widget, world points rotated
into the UGV's local frame every frame via to_local().
"""

from __future__ import annotations
import math

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QRadialGradient, QConicalGradient,
    QFont, QPainterPath,
)

BG = QColor(5, 7, 13)
GRID = QColor(0, 255, 200, 40)
GRID_TEXT = QColor(0, 255, 200, 150)
ACCENT = QColor(0, 255, 200)
AMBER = QColor(255, 176, 32)
RED = QColor(255, 70, 70)
BLUE = QColor(60, 160, 255)
GREEN = QColor(60, 210, 120)
GREY = QColor(150, 165, 180)

LABEL_COLORS = {
    "obstacle": RED,
    "vehicle": BLUE,
    "vegetation": GREEN,
    "drivable": GREY,
}

# Rings are DERIVED from the display range, never hardcoded: DESIGN-APP.md
# flags that the old fixed (20,40,60,80,100) tuple agreed with the bezel only
# because SENSOR_RANGE_M happened to be exactly 100.0, so any rescale would
# silently draw a 100m ring outside the dial. Picks the step that yields 3-6
# rings; at 100m this still produces exactly (20,40,60,80,100).
_RING_STEPS = (2, 5, 10, 20, 25, 50)


def _range_rings(display_range_m):
    for step in _RING_STEPS:
        if 3 <= display_range_m / step <= 6:
            break
    n = int(display_range_m / step)
    return tuple(step * (k + 1) for k in range(n))
SWEEP_PERIOD_S = 4.0        # seconds for one full sweep rotation
SWEEP_ARC_DEG = 55          # width of the fading sweep wedge

# What the dial spans, which is deliberately NOT the 100m sensor range.
# Measured detection ranges: median 13.3m, 87% within 40m, 95% within 50m.
# At 100m the dial resolved only 3.8 px/m, which left the 2m navigation
# corridor a ~15px sliver and pushed every real object into a knot at the
# centre. 40m gives 9.6 px/m; the few detections beyond it clip at the rim.
DISPLAY_RANGE_M = 40.0

ELEVATION_PX_PER_M = 8          # screen px per meter of height (fixed visual scale, not range-scaled)
MIN_ELEVATION_FOR_PIN_M = 0.5   # below this: render exactly as before (icon at ground point)
MAX_STEM_PX = 40                # clamp so a very tall object's pin can't fly off-panel

# Measured track range distribution (real clustered detections): median
# 26.1m, 46.7% within 25m -- full glyph treatment only within this range so
# the display reads clearly instead of every track competing for attention
# regardless of distance; beyond it, a plain dot.
DETAIL_RANGE_M = 25.0
FAR_DOT_RADIUS_PX = 3.5

DENSITY_DOT_ALPHA_NEAR = 70     # dimmer than ground-clutter dots (alpha 90) -- reads as
DENSITY_DOT_ALPHA_FAR = 45      # "sensor return" texture, not discrete objects
DENSITY_DOT_RADIUS_NEAR = 1.4
DENSITY_DOT_RADIUS_FAR = 0.9

# Cap on the DRAWN length of a velocity arrow. The underlying velocity is
# untouched (is_dynamic and threat evaluation still use it in full); this is
# purely legibility. Measured arrow length (|v| * 2.5s) over 120 live frames:
# median 26.5m, p90 37.6m, max 53.0m -- on a 40m dial the median arrow reached
# two thirds of the way to the rim and 6% overshot it entirely, turning the
# display into a web of crossing lines. The lengths come from re-clustering
# jitter, not real motion (AUDIT-V3 §3.3), so nothing meaningful is lost.
VELOCITY_ARROW_MAX_M = 8.0


class RadarView2D(QWidget):
    def __init__(self, sensor_range_m=DISPLAY_RANGE_M, parent=None):
        super().__init__(parent)
        self.sensor_range_m = sensor_range_m
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background-color: transparent;")

        self._frame = None
        self._ground_dots = self._make_ground_clutter()

        self._sweep_deg = 0.0
        self._sweep_timer = QTimer(self)
        self._sweep_timer.timeout.connect(self._advance_sweep)
        self._sweep_timer.start(33)  # ~30fps sweep animation, independent of engine fps

    # ------------------------------------------------------------------
    def _make_ground_clutter(self):
        """Decorative low-intensity 'ground return' speckle, purely for
        visual texture -- this is NOT real LiDAR data. engine_adapter.py's
        Frame only carries clustered track objects, not raw points, so
        there is nothing real to plot here yet. If you want this to be
        real, the cheapest path is to have Engine.step() also return a
        down-sampled array of classified-drivable points and swap this
        method out for a real one."""
        import random
        rng = random.Random(7)
        dots = []
        for _ in range(420):
            ang = rng.uniform(0, 2 * math.pi)
            # bias toward two loose bands rather than a uniform disc,
            # just so it doesn't look like a flat random fill
            band = rng.choice([-1, 1])
            r = rng.uniform(4, DISPLAY_RANGE_M * 0.98)
            jitter = rng.uniform(-18, 18) * band
            dots.append((ang, r, jitter))
        return dots

    def _advance_sweep(self):
        self._sweep_deg = (self._sweep_deg + 360.0 / (SWEEP_PERIOD_S * 30.0)) % 360.0
        self.update()

    # ------------------------------------------------------------------
    def update_frame(self, frame):
        self._frame = frame
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        side = min(self.width(), self.height()) - 60
        if side <= 0:
            return
        cx, cy = self.width() / 2, self.height() / 2
        R = side / 2  # pixel radius for sensor_range_m

        def polar_to_px(range_m, bearing_deg):
            # bearing_deg: 0 = forward/up on screen, clockwise positive
            rad = math.radians(bearing_deg)
            rr = (range_m / self.sensor_range_m) * R
            x = cx + rr * math.sin(rad)
            y = cy - rr * math.cos(rad)
            return x, y

        self._draw_sweep(p, cx, cy, R)
        self._draw_range_rings(p, cx, cy, R)
        self._draw_ground_clutter(p, polar_to_px)
        self._draw_compass_bezel(p, cx, cy, R)

        if self._frame is not None:
            self._draw_density_field(p, polar_to_px)
            self._draw_planned_path(p, polar_to_px)
            self._draw_evasion_arc(p, polar_to_px)
            self._draw_tracks(p, polar_to_px)

        self._draw_ugv(p, cx, cy)
        p.end()

    # ------------------------------------------------------------------
    def _draw_sweep(self, p, cx, cy, R):
        # Layered pie slices with decreasing alpha, rather than a
        # QConicalGradient -- much easier to keep the "empty" 300+
        # degrees of the circle genuinely black instead of tinted.
        p.setPen(Qt.PenStyle.NoPen)
        slice_deg = 6
        n_slices = SWEEP_ARC_DEG // slice_deg
        rect = QRectF(cx - R, cy - R, 2 * R, 2 * R)
        for k in range(n_slices):
            bearing_lead = (self._sweep_deg - k * slice_deg) % 360
            qt_start = 90 - bearing_lead          # convert bearing(N,cw) -> Qt angle(E,ccw)
            alpha = max(0, int(75 * (1 - k / n_slices)))
            p.setBrush(QBrush(QColor(0, 255, 190, alpha)))
            p.drawPie(rect, int(qt_start * 16), int(-slice_deg * 16))

    def _draw_range_rings(self, p, cx, cy, R):
        p.setFont(QFont("Consolas", 8))
        for r_m in _range_rings(self.sensor_range_m):
            rr = (r_m / self.sensor_range_m) * R
            p.setPen(QPen(GRID, 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), rr, rr)
            p.setPen(QPen(GRID_TEXT))
            p.drawText(QPointF(cx + 4, cy - rr - 3), f"{r_m}m")

    def _draw_compass_bezel(self, p, cx, cy, R):
        p.setPen(QPen(QColor(0, 255, 200, 90), 1.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), R, R)

        p.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        cardinals = {0: "N", 90: "E", 180: "S", 270: "W"}
        for bearing in range(0, 360, 30):
            rad = math.radians(bearing)
            x1 = cx + R * math.sin(rad)
            y1 = cy - R * math.cos(rad)
            x2 = cx + (R + 10) * math.sin(rad)
            y2 = cy - (R + 10) * math.cos(rad)
            p.setPen(QPen(QColor(0, 255, 200, 55), 1))
            p.drawLine(QPointF(cx, cy), QPointF(x1, y1))

            label = cardinals.get(bearing)
            text = label if label else f"{bearing:03d}"
            color = ACCENT if label else GRID_TEXT
            p.setPen(QPen(color))
            tx = cx + (R + 22) * math.sin(rad)
            ty = cy - (R + 22) * math.cos(rad)
            fm = p.fontMetrics()
            w = fm.horizontalAdvance(text)
            p.drawText(QPointF(tx - w / 2, ty + 4), text)

    def _draw_ground_clutter(self, p, polar_to_px):
        p.setPen(Qt.PenStyle.NoPen)
        for ang, r, jitter in self._ground_dots:
            bearing = math.degrees(ang)
            x, y = polar_to_px(r, bearing)
            x += jitter * 0.3
            p.setBrush(QBrush(QColor(60, 220, 120, 90)))
            p.drawEllipse(QPointF(x, y), 1.3, 1.3)

    def _draw_planned_path(self, p, polar_to_px):
        """The forward corridor the navigation controller is actually
        scanning, plus the heading it has chosen. Because this is a
        heading-up display the corridor is always a vertical band rising
        from the UGV, so its edges are just two bearings at +/- the angle
        that CORRIDOR_HALF_WIDTH_M subtends at each range -- no new
        transform math, same polar_to_px the rest of the view uses."""
        frame = self._frame
        look = getattr(frame, "corridor_lookahead_m", None)
        if not look:
            return
        half = frame.corridor_half_width_m
        blocked = frame.corridor_blocked

        edge = QColor(AMBER) if blocked else QColor(ACCENT)
        edge.setAlpha(220 if blocked else 80)
        p.setPen(QPen(edge, 2.0 if blocked else 1.4,
                      Qt.PenStyle.SolidLine if blocked else Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for side in (-1, 1):
            path = QPainterPath()
            for i in range(13):
                r = 0.6 + (look - 0.6) * i / 12
                bearing = math.degrees(math.atan2(side * half, r))
                x, y = polar_to_px(math.hypot(r, half), bearing)
                path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
            p.drawPath(path)

        # Range readout on the blocking obstacle.
        if blocked and frame.corridor_closest_m is not None:
            x, y = polar_to_px(frame.corridor_closest_m, 0)
            p.setPen(QPen(AMBER, 1.6))
            p.drawLine(QPointF(x - 9, y), QPointF(x + 9, y))
            p.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            p.drawText(QPointF(x + 13, y + 3), f"{frame.corridor_closest_m:.1f}m")


    def _draw_evasion_arc(self, p, polar_to_px):
        frame = self._frame
        if not (frame.evasion_active and frame.evasion_target_heading_deg is not None):
            return
        # evasion_target_heading_deg is absolute (world); on a heading-up
        # display the offset from the current heading IS the on-screen angle.
        # This was hardcoded to +35, which drew a right-hand arc even when the
        # UGV was steering left -- harmless while the evasion state machine
        # never ran in live mode, actively misleading now that it does.
        rel = (self._frame.evasion_target_heading_deg
               - self._frame.ugv_heading_deg + 180) % 360 - 180
        path = QPainterPath()
        steps = 24
        for i in range(steps):
            t = i / (steps - 1)
            bearing = t * rel
            r = 2 + t * 20
            x, y = polar_to_px(r, bearing)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        pen = QPen(AMBER, 2.4, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    # ------------------------------------------------------------------
    def _draw_cube_icon(self, p, x, y, size, color, glow=False):
        """Small isometric-looking 'target' glyph -- a flat 2D trick
        (offset square + connecting lines) to read as a 3D box without
        actually doing any 3D projection."""
        if glow:
            grad = QRadialGradient(x, y, size * 2.6)
            glow_color = QColor(color)
            glow_color.setAlpha(140)
            grad.setColorAt(0.0, glow_color)
            grad.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(grad))
            p.drawEllipse(QPointF(x, y), size * 2.6, size * 2.6)

        off = size * 0.45
        front = QRectF(x - size / 2, y - size / 2 + off * 0.5, size, size)
        back = QRectF(x - size / 2 + off, y - size / 2 - off * 0.5, size, size)

        pen = QPen(color, 1.6)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(back)
        p.drawRect(front)
        for corner in (
            (back.left(), back.top(), front.left(), front.top()),
            (back.right(), back.top(), front.right(), front.top()),
            (back.right(), back.bottom(), front.right(), front.bottom()),
            (back.left(), back.bottom(), front.left(), front.bottom()),
        ):
            p.drawLine(QPointF(corner[0], corner[1]), QPointF(corner[2], corner[3]))

    def _world_to_px(self, wx, wy, polar_to_px):
        """Shared bearing/range/polar_to_px transform for any world-frame
        point relative to the current frame's UGV pose -- factored out so
        _draw_density_field doesn't reimplement the math _draw_tracks
        already needs for track heads, trail points, and velocity tips."""
        bearing = math.degrees(math.atan2(
            wx - self._frame.ugv_x, wy - self._frame.ugv_y
        )) - self._frame.ugv_heading_deg
        r = math.hypot(wx - self._frame.ugv_x, wy - self._frame.ugv_y)
        return polar_to_px(r, bearing), r

    def _draw_density_field(self, p, polar_to_px):
        # Non-tracked background classification (frame.density_points):
        # small, dim, plain dots -- reads as "sensor return" texture, not
        # discrete objects. Drawn before the evasion arc/tracks/UGV glyph
        # so it sits visually beneath them.
        p.setPen(Qt.PenStyle.NoPen)
        for wx, wy, label in self._frame.density_points:
            (x, y), r = self._world_to_px(wx, wy, polar_to_px)
            color = QColor(LABEL_COLORS.get(label, GREY))
            near = r <= DETAIL_RANGE_M
            color.setAlpha(DENSITY_DOT_ALPHA_NEAR if near else DENSITY_DOT_ALPHA_FAR)
            p.setBrush(QBrush(color))
            radius = DENSITY_DOT_RADIUS_NEAR if near else DENSITY_DOT_RADIUS_FAR
            p.drawEllipse(QPointF(x, y), radius, radius)

    def _draw_tracks(self, p, polar_to_px):
        for trk in self._frame.tracks:
            (x, y), r = self._world_to_px(trk.x, trk.y, polar_to_px)

            base_color = LABEL_COLORS.get(trk.label, GREY)
            color = RED if trk.threat else base_color

            if r > DETAIL_RANGE_M:
                # Beyond detail range: a plain dot, no glow/trail/arrow --
                # full glyph treatment is reserved for near obstacles so the
                # display reads clearly instead of every track competing for
                # attention regardless of distance.
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(color))
                p.drawEllipse(QPointF(x, y), FAR_DOT_RADIUS_PX, FAR_DOT_RADIUS_PX)
                continue

            if trk.is_dynamic and len(trk.history) > 1:
                path = QPainterPath()
                pts = trk.history + [(trk.x, trk.y)]
                for i, (hx, hy) in enumerate(pts):
                    (hxp, hyp), _hr = self._world_to_px(hx, hy, polar_to_px)
                    if i == 0:
                        path.moveTo(hxp, hyp)
                    else:
                        path.lineTo(hxp, hyp)
                trail_color = QColor(color)
                trail_color.setAlpha(130)
                p.setPen(QPen(trail_color, 1.8))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(path)

            dz_px = min(trk.z * ELEVATION_PX_PER_M, MAX_STEM_PX)
            icon_y = y - dz_px
            show_pin = trk.z >= MIN_ELEVATION_FOR_PIN_M

            if show_pin:
                shadow_color = QColor(color)
                shadow_color.setAlpha(90)
                p.setPen(QPen(shadow_color, 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(x, y), 4, 4)

                stem_color = QColor(color)
                stem_color.setAlpha(130)
                p.setPen(QPen(stem_color, 1.2))
                p.drawLine(QPointF(x, y), QPointF(x, icon_y))

            self._draw_cube_icon(p, x, icon_y, 16 if trk.is_dynamic else 12, color,
                                  glow=trk.threat)

            vx, vy = trk.velocity
            if trk.is_dynamic and abs(vx) + abs(vy) > 0.05:
                ox, oy = vx * 2.5, vy * 2.5
                reach = math.hypot(ox, oy)
                if reach > VELOCITY_ARROW_MAX_M:
                    scale = VELOCITY_ARROW_MAX_M / reach
                    ox, oy = ox * scale, oy * scale
                (tx, ty), _tip_r = self._world_to_px(trk.x + ox, trk.y + oy, polar_to_px)
                p.setPen(QPen(color, 2))
                p.drawLine(QPointF(x, y), QPointF(tx, ty))

    def _draw_ugv(self, p, cx, cy):
        glow = QRadialGradient(cx, cy, 30)
        glow.setColorAt(0.0, QColor(0, 255, 200, 130))
        glow.setColorAt(1.0, QColor(0, 255, 200, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(QPointF(cx, cy), 30, 30)

        path = QPainterPath()
        s = 11
        path.moveTo(cx, cy - s)
        path.lineTo(cx - s * 0.8, cy + s * 0.7)
        path.lineTo(cx + s * 0.8, cy + s * 0.7)
        path.closeSubpath()
        p.setPen(QPen(ACCENT, 2))
        p.setBrush(QBrush(QColor(0, 40, 35)))
        p.drawPath(path)
"""
avrlm_radar_app.py
------------------------------------------------------------------
AVRLM — DRDO-style tactical LiDAR perception dashboard.
Native desktop app (PyQt6 + PyQtGraph). Freeze into a .exe with
PyInstaller (see build_exe.md).

v4: flat, top-down 2D "PPI" radar view (radar_view_2d.RadarView2D,
plain QPainter) instead of the tilted 3D perspective. UGV, range
rings, object icons, comet trails, and the avoidance arc are drawn
on a heading-up polar display with a compass bezel and rotating
sweep wedge — matching the classic tactical-radar-screen look.

Layout:
  - Center: 2D polar radar. UGV fixed at the center facing "forward"
    (+Y), world rotates around it as it moves (heading-up display).
    Range rings at 20/40/60/80/100m. Static objects get outlined
    icons; dynamic objects get comet trails + velocity arrows;
    threats glow red with a dashed amber avoidance arc.
  - Top bar: title, mission clock, speed, heading, system status.
  - Left rail: engine performance metric cards.
  - Right rail: scrolling monospace event log.
  - Bottom strip: compact rolling sparkline charts (FPS / Latency /
    Spike Rate).

All data comes from engine_adapter.Engine — swap in your real
spiking_model.py / profiler.py / kitti_loader.py by editing ONLY
engine_adapter.py, never this file.

Install:
    pip install PyQt6 pyqtgraph numpy

Run:
    python avrlm_radar_app.py
"""

import sys
import math
import time
import collections
from dataclasses import replace

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, QElapsedTimer, QThread, pyqtSignal
from PyQt6.QtGui import QColor

import pyqtgraph as pg

from engine_adapter import Engine, TARGET_FPS
from radar_view_2d import RadarView2D, DISPLAY_RANGE_M
from terrain_relief import TerrainReliefView

pg.setConfigOptions(antialias=True, background=(4, 6, 12), foreground=(210, 235, 230))

ACCENT = (0, 1.0, 0.78, 1.0)      # RGBA 0-1 for OpenGL items
AMBER = (1.0, 0.69, 0.13, 1.0)
RED = (1.0, 0.30, 0.30, 1.0)
BLUE = (0.23, 0.63, 1.0, 1.0)
GREEN = (0.20, 0.82, 0.48, 1.0)
GREY = (0.6, 0.65, 0.7, 0.7)

ACCENT_HEX = "#00ffc8"

LABEL_COLORS_GL = {
    "obstacle": RED,
    "vehicle": BLUE,
    "vegetation": GREEN,
    "drivable": GREY,
}

ROLLING_WINDOW = 150
MAX_LOG_LINES = 200
WORKER_MAX_HZ = 60.0    # cap for synthetic mode (near-instant steps); real
                        # live-mode inference (~200ms/frame, measured) runs
                        # flat-out well below this cap regardless.


# ==========================================================================
# ENGINE WORKER
# ==========================================================================
class EngineWorker(QThread):
    """Runs Engine.step() on a background thread. Real live-mode inference
    measured at ~200ms/frame (6-10x the 30fps UI budget), so it must not
    block the Qt main thread -- the UI thread instead redraws at 30fps from
    whatever Frame this worker most recently emitted. Emits plain Frame
    dataclass objects only (no Qt objects), keeping all Qt-specific work on
    the receiving (main) thread, per the same Qt-free-core pattern already
    used in terrain_relief.py."""
    frame_ready = pyqtSignal(object)

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._running = True

    def run(self):
        min_interval = 1.0 / WORKER_MAX_HZ
        while self._running:
            t0 = time.perf_counter()
            frame = self.engine.step()
            self.frame_ready.emit(frame)
            remaining = min_interval - (time.perf_counter() - t0)
            if remaining > 0:
                self.msleep(int(remaining * 1000))

    def stop(self):
        self._running = False
        self.wait(2000)


# ==========================================================================
# METRIC CARD
# ==========================================================================
class MetricCard(QFrame):
    def __init__(self, title, unit=""):
        super().__init__()
        self.setStyleSheet("""
            QFrame { background-color: #0c1220; border: 1px solid rgba(0,255,200,55);
                     border-radius: 10px; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        t = QLabel(title)
        t.setStyleSheet("color:#8fa3b8; font-size:11px; border:none;")
        self.value_lbl = QLabel("--")
        self.value_lbl.setStyleSheet(f"color:{ACCENT_HEX}; font-size:21px; font-weight:700; border:none;")
        self.sub_lbl = QLabel(unit)
        self.sub_lbl.setStyleSheet("color:#54687a; font-size:9px; border:none;")
        layout.addWidget(t)
        layout.addWidget(self.value_lbl)
        layout.addWidget(self.sub_lbl)

    def set_value(self, v, sub=None):
        self.value_lbl.setText(v)
        if sub is not None:
            self.sub_lbl.setText(sub)


# ==========================================================================
# MAIN WINDOW
# ==========================================================================
class AVRLMRadarWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AVRLM — Tactical LiDAR Perception")
        self.resize(1600, 950)
        self.setStyleSheet("background-color: #05070d;")

        self.engine = Engine()
        self.fps_hist = collections.deque(maxlen=ROLLING_WINDOW)
        self.spike_hist = collections.deque(maxlen=ROLLING_WINDOW)
        self.latency_hist = collections.deque(maxlen=ROLLING_WINDOW)

        self.clock = QElapsedTimer()
        self.clock.start()

        self._build_ui()

        self._latest_frame = None
        self._prev_frame = None
        self._latest_frame_wall_time = None
        self._frame_interval = 1.0 / 30.0
        self._last_frame_id = None
        self._worker = EngineWorker(self.engine, parent=self)
        self._worker.frame_ready.connect(self._on_engine_frame)
        self._worker.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_frame)
        self.timer.start(int(1000 / 30))   # UI redraw cap: 30 fps, independent of engine fps

    def closeEvent(self, event):
        self._worker.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(8)

        top_bar = QHBoxLayout()
        title_col = QVBoxLayout()
        title = QLabel("AVRLM — TACTICAL LIDAR PERCEPTION")
        title.setStyleSheet("color:#e8fdf7; font-size:18px; font-weight:700; letter-spacing:1px;")
        subtitle = QLabel("LIDAR 360° / PPI VIEW")
        subtitle.setStyleSheet("color:#4a6a78; font-size:10px; letter-spacing:1px;")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        top_bar.addLayout(title_col)
        top_bar.addStretch()
        self.mission_clock_lbl = self._hud_label("T+00:00:00")
        self.speed_lbl = self._hud_label("SPEED 0.0 m/s")
        self.heading_lbl = self._hud_label("HDG 000°")
        for w in (self.mission_clock_lbl, self.speed_lbl, self.heading_lbl):
            top_bar.addWidget(w)
        self.status_badge = QLabel("●  SYSTEM NOMINAL")
        self.status_badge.setStyleSheet(
            "color:#2ecf6b; font-size:11px; font-weight:700; padding: 2px 10px;"
        )
        top_bar.addWidget(self.status_badge)
        operator_lbl = QLabel("OPERATOR\nALPHA-07")
        operator_lbl.setStyleSheet("color:#5c6f82; font-size:9px;")
        top_bar.addWidget(operator_lbl)
        root.addLayout(top_bar)

        main_row = QHBoxLayout()

        left_col = QVBoxLayout()
        left_col.addWidget(self._section_label("ENGINE METRICS"))
        self.card_fps = MetricCard("FPS (engine)")
        self.card_latency = MetricCard("Latency", "per frame")
        self.card_spike = MetricCard("Spike rate", "active neurons")
        self.card_sparsity = MetricCard("Sparsity", "compute saved")
        self.card_mode = MetricCard("Engine mode")
        for c in (self.card_fps, self.card_latency, self.card_spike, self.card_sparsity, self.card_mode):
            left_col.addWidget(c)
        left_col.addStretch()
        left_wrap = QWidget()
        left_wrap.setLayout(left_col)
        left_wrap.setFixedWidth(190)
        main_row.addWidget(left_wrap)

        # Dial spans DISPLAY_RANGE_M (40m), not the 100m sensor range -- see
        # radar_view_2d.DISPLAY_RANGE_M for the measured reason. Detection
        # range is unchanged; this is purely how much of it the dial shows.
        self.radar = RadarView2D(sensor_range_m=DISPLAY_RANGE_M)
        main_row.addWidget(self.radar, stretch=3)

        right_col = QVBoxLayout()
        right_col.addWidget(self._section_label("NEAR-FIELD TERRAIN"))
        self.terrain_view = TerrainReliefView()
        right_col.addWidget(self.terrain_view)
        right_col.addWidget(self._section_label("EVENT LOG"))
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("""
            QListWidget { background-color:#0a0e18; color:#33d17a; border:1px solid #16202e;
                          font-family:Consolas, monospace; font-size:11px; }
        """)
        right_col.addWidget(self.log_list)
        right_wrap = QWidget()
        right_wrap.setLayout(right_col)
        right_wrap.setFixedWidth(300)
        main_row.addWidget(right_wrap)

        root.addLayout(main_row, stretch=4)

        bottom_row = QHBoxLayout()
        self.fps_widget, self.fps_curve = self._make_sparkline("FPS", ACCENT_HEX)
        self.latency_widget, self.latency_curve = self._make_sparkline("LATENCY", "#ff4d4d")
        self.spike_widget, self.spike_curve = self._make_sparkline("SPIKE RATE", "#3aa0ff")
        for w in (self.fps_widget, self.latency_widget, self.spike_widget):
            bottom_row.addWidget(w)
        root.addLayout(bottom_row, stretch=1)

    def _hud_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{ACCENT_HEX}; font-size:12px; font-family:Consolas, monospace; "
                           f"padding: 2px 12px; border:1px solid rgba(0,255,200,50); border-radius:4px;")
        return lbl

    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#5c6f82; font-size:11px; font-weight:700; letter-spacing:1px;")
        return lbl

    def _make_sparkline(self, title, color):
        w = pg.PlotWidget(title=title)
        w.setMinimumHeight(90)
        w.setMaximumHeight(110)
        w.hideAxis("bottom")
        w.showGrid(y=True, alpha=0.1)
        curve = w.plot(pen=pg.mkPen(color, width=2))
        return w, curve

    # ------------------------------------------------------------------
    def _on_engine_frame(self, frame):
        now = time.perf_counter()
        if self._latest_frame is not None:
            self._prev_frame = self._latest_frame
            if self._latest_frame_wall_time is not None:
                self._frame_interval = max(now - self._latest_frame_wall_time, 1e-3)
        self._latest_frame = frame
        self._latest_frame_wall_time = now

    def _build_display_frame(self, latest):
        # Real engine frames arrive slower (and less regularly) than this
        # 30fps redraw tick -- holding the last frame static until the next
        # one arrives makes motion look stepped. Instead, extrapolate track/
        # UGV positions forward from the last two real frames at their
        # measured velocity, so on-screen motion looks smooth at whatever
        # the real engine rate ends up being. This is display-only: the
        # returned Frame is never fed back into the engine or tracker.
        prev = self._prev_frame
        if prev is None or self._latest_frame_wall_time is None:
            return latest
        t = (time.perf_counter() - self._latest_frame_wall_time) / self._frame_interval + 1.0
        t = max(0.0, min(t, 2.0))   # clamp: at most one extra full interval of extrapolation

        prev_by_id = {trk.track_id: trk for trk in prev.tracks}
        interp_tracks = []
        for trk in latest.tracks:
            p = prev_by_id.get(trk.track_id)
            if p is None:
                interp_tracks.append(trk)
                continue
            interp_tracks.append(replace(trk, x=p.x + (trk.x - p.x) * t, y=p.y + (trk.y - p.y) * t))

        iux = prev.ugv_x + (latest.ugv_x - prev.ugv_x) * t
        iuy = prev.ugv_y + (latest.ugv_y - prev.ugv_y) * t
        diff = (latest.ugv_heading_deg - prev.ugv_heading_deg + 180) % 360 - 180
        iuh = (prev.ugv_heading_deg + diff * t) % 360

        return replace(latest, tracks=interp_tracks, ugv_x=iux, ugv_y=iuy, ugv_heading_deg=iuh)

    def _on_frame(self):
        frame = self._latest_frame
        if frame is None:
            return   # worker hasn't produced a frame yet

        # Real live-mode inference (~200ms/frame, measured) runs slower than
        # this 30fps redraw tick, so the same Frame is often still the
        # latest across several consecutive calls -- only append to
        # history/log once per distinct engine frame, not once per redraw,
        # or the sparkline time axis and event log would both duplicate.
        is_new = frame.frame_id != self._last_frame_id
        self._last_frame_id = frame.frame_id

        if is_new:
            self.fps_hist.append(frame.fps)
            self.spike_hist.append(frame.spike_rate)
            self.latency_hist.append(frame.latency_ms)

        elapsed = self.clock.elapsed() // 1000
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        self.mission_clock_lbl.setText(f"T+{h:02d}:{m:02d}:{s:02d}")
        self.speed_lbl.setText(f"SPEED {frame.ugv_speed_mps:0.1f} m/s")
        self.heading_lbl.setText(f"HDG {frame.ugv_heading_deg:03.0f}°")

        # TARGET_FPS (261) is a leftover synthetic-mode display constant --
        # not a meaningful target for live inference, so don't show it then.
        self.card_fps.set_value(f"{frame.fps:.0f}",
                                 "30fps UI cap" if self.engine.live else f"target {TARGET_FPS}")
        self.card_latency.set_value(f"{frame.latency_ms:.2f} ms")
        self.card_spike.set_value(f"{frame.spike_rate * 100:.1f}%")
        self.card_sparsity.set_value(f"{frame.sparsity * 100:.1f}%")
        self.card_mode.set_value("LIVE" if self.engine.live else "SYNTHETIC",
                                  "spiking_model.py" if self.engine.live
                                  else (self.engine.live_init_error or "demo fallback"))

        display_frame = self._build_display_frame(frame)
        self.radar.update_frame(display_frame)
        self.terrain_view.update_frame(display_frame)

        if is_new:
            for event in frame.log_events:
                item = QListWidgetItem(f"> {event}")
                if "evasive" in event.lower():
                    item.setForeground(QColor("#ffb020"))
                self.log_list.addItem(item)
                if self.log_list.count() > MAX_LOG_LINES:
                    self.log_list.takeItem(0)
                self.log_list.scrollToBottom()

            self.fps_curve.setData(list(self.fps_hist))
            self.spike_curve.setData(list(self.spike_hist))
            self.latency_curve.setData(list(self.latency_hist))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AVRLMRadarWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
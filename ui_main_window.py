import time
import threading
from collections import OrderedDict
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid
from skimage import measure

from app_state import AppState
from ui_widgets import (
    ProjectionViewWidget,
    build_labeled_slider,
    build_title_label,
)


class LRUCache:
    def __init__(self, max_items: int) -> None:
        self._max_items = max_items
        self._data: OrderedDict[tuple, object] = OrderedDict()

    def get(self, key: tuple) -> object | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def get_with_hit(self, key: tuple) -> tuple[bool, object | None]:
        if key not in self._data:
            return False, None
        self._data.move_to_end(key)
        return True, self._data[key]

    def set(self, key: tuple, value: object) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._max_items:
            self._data.popitem(last=False)


def _volume_cache_key(
    orbital_name: str,
    projection_mode: str,
    angles: dict,
    resolution: int,
    samples: int,
    extent: float,
    view_id: str,
) -> tuple:
    angle_tuple = tuple(angles[name] for name in ["xy", "xz", "xw", "yz", "yw", "zw"])
    sample_value = samples if not projection_mode.startswith("slice") else 0
    return (orbital_name, projection_mode, angle_tuple, resolution, sample_value, extent, view_id)


def _mesh_cache_key(volume_key: tuple, iso_value: float, sign: int) -> tuple:
    return (volume_key, round(iso_value, 6), sign)


def _compose_rotation_matrix_from_angles(angles: dict) -> np.ndarray:
    return (
        _rotation_matrix_zw(angles["zw"])
        @ _rotation_matrix_yw(angles["yw"])
        @ _rotation_matrix_yz(angles["yz"])
        @ _rotation_matrix_xw(angles["xw"])
        @ _rotation_matrix_xz(angles["xz"])
        @ _rotation_matrix_xy(angles["xy"])
    )


def _generate_slice_volume(
    view_id: str,
    resolution: int,
    extent: float,
    rotation: np.ndarray,
) -> np.ndarray:
    coords = np.linspace(-extent, extent, resolution)
    grid_a, grid_b, grid_c = np.meshgrid(coords, coords, coords, indexing="ij")

    if view_id == "X":
        x = np.zeros_like(grid_a)
        y, z, w = grid_a, grid_b, grid_c
    elif view_id == "Y":
        y = np.zeros_like(grid_a)
        x, z, w = grid_a, grid_b, grid_c
    elif view_id == "Z":
        z = np.zeros_like(grid_a)
        x, y, w = grid_a, grid_b, grid_c
    else:
        w = np.zeros_like(grid_a)
        x, y, z = grid_a, grid_b, grid_c

    points = np.stack(
        [x.reshape(-1), y.reshape(-1), z.reshape(-1), w.reshape(-1)], axis=0
    )
    rotated = rotation @ points
    xr = rotated[0].reshape(x.shape)
    yr = rotated[1].reshape(y.shape)
    zr = rotated[2].reshape(z.shape)
    wr = rotated[3].reshape(w.shape)

    r = np.sqrt(xr**2 + yr**2 + zr**2 + wr**2)
    return np.exp(-r) * (xr**2 - yr**2 + 0.35 * zr - 0.25 * wr)


def _generate_integral_volume(
    view_id: str,
    resolution: int,
    extent: float,
    rotation: np.ndarray,
    samples: int,
    mode: str,
    cancel_event: threading.Event | None = None,
) -> np.ndarray | None:
    coords = np.linspace(-extent, extent, resolution)
    grid_a, grid_b, grid_c = np.meshgrid(coords, coords, coords, indexing="ij")

    if view_id == "X":
        axis_label = "x"
        y, z, w = grid_a, grid_b, grid_c
    elif view_id == "Y":
        axis_label = "y"
        x, z, w = grid_a, grid_b, grid_c
    elif view_id == "Z":
        axis_label = "z"
        x, y, w = grid_a, grid_b, grid_c
    else:
        axis_label = "w"
        x, y, z = grid_a, grid_b, grid_c

    nodes, weights = np.polynomial.legendre.leggauss(samples)
    nodes = nodes * extent
    weights = weights * extent

    grid_flat = grid_a.reshape(-1)
    if view_id == "X":
        y_flat, z_flat, w_flat = y.reshape(-1), z.reshape(-1), w.reshape(-1)
    elif view_id == "Y":
        x_flat, z_flat, w_flat = x.reshape(-1), z.reshape(-1), w.reshape(-1)
    elif view_id == "Z":
        x_flat, y_flat, w_flat = x.reshape(-1), y.reshape(-1), w.reshape(-1)
    else:
        x_flat, y_flat, z_flat = x.reshape(-1), y.reshape(-1), z.reshape(-1)

    accumulator = np.zeros_like(grid_flat, dtype=np.float64)

    for t_value, weight in zip(nodes, weights):
        if cancel_event is not None and cancel_event.is_set():
            return None
        if axis_label == "x":
            x_vals = np.full_like(grid_flat, t_value, dtype=np.float64)
            points = np.stack([x_vals, y_flat, z_flat, w_flat], axis=1)
        elif axis_label == "y":
            y_vals = np.full_like(grid_flat, t_value, dtype=np.float64)
            points = np.stack([x_flat, y_vals, z_flat, w_flat], axis=1)
        elif axis_label == "z":
            z_vals = np.full_like(grid_flat, t_value, dtype=np.float64)
            points = np.stack([x_flat, y_flat, z_vals, w_flat], axis=1)
        else:
            w_vals = np.full_like(grid_flat, t_value, dtype=np.float64)
            points = np.stack([x_flat, y_flat, z_flat, w_vals], axis=1)

        rotated = points @ rotation.T
        xr, yr, zr, wr = rotated[:, 0], rotated[:, 1], rotated[:, 2], rotated[:, 3]
        r = np.sqrt(xr**2 + yr**2 + zr**2 + wr**2)
        psi = np.exp(-r) * (xr**2 - yr**2 + 0.35 * zr - 0.25 * wr)

        if mode.startswith("integral ψ"):
            accumulator += psi * weight
        elif mode.startswith("integral |ψ|"):
            accumulator += np.abs(psi) * weight
        else:
            accumulator = np.maximum(accumulator, np.abs(psi))

    return accumulator.reshape(grid_a.shape)


def _extract_mesh(
    vol: np.ndarray,
    level: float,
    extent: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if level == 0 or vol.size == 0:
        return None
    vol_min = float(np.min(vol))
    vol_max = float(np.max(vol))
    if not (vol_min <= level <= vol_max):
        return None
    spacing = (2 * extent) / max(vol.shape[0] - 1, 1)
    origin = np.array([-extent, -extent, -extent], dtype=float)
    try:
        verts, faces, _, _ = measure.marching_cubes(
            vol,
            level=level,
            spacing=(spacing, spacing, spacing),
        )
    except RuntimeError:
        return None
    if verts.size == 0 or faces.size == 0:
        return None
    verts = verts + origin
    return verts, faces


class RenderWorker(QtCore.QObject):
    view_ready = QtCore.Signal(int, str, object, object, dict)
    log_line = QtCore.Signal(str)
    finished = QtCore.Signal(int)

    def __init__(
        self,
        request_id: int,
        params: dict,
        volume_cache: LRUCache,
        mesh_cache: LRUCache,
        cache_lock: threading.Lock,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self._request_id = request_id
        self._params = params
        self._volume_cache = volume_cache
        self._mesh_cache = mesh_cache
        self._cache_lock = cache_lock
        self._cancel_event = cancel_event

    def run(self) -> None:
        start_total = time.perf_counter()
        mode = self._params["projection_mode"]
        quality_label = self._params["quality_label"]
        resolution = self._params["resolution"]
        samples = self._params["samples"]
        iso_percent = self._params["iso_percent"]
        extent = self._params["extent"]

        self.log_line.emit(
            "Render#{rid} start ({quality}): mode={mode}, N={res}, M={samples}".format(
                rid=self._request_id,
                quality=quality_label,
                mode=mode,
                res=resolution,
                samples=samples,
            )
        )

        rotation = _compose_rotation_matrix_from_angles(self._params["angles"])
        for index, view_id in enumerate(["X", "Y", "Z", "W"], start=1):
            if self._cancel_event.is_set():
                break

            view_start = time.perf_counter()
            volume_key = _volume_cache_key(
                self._params["orbital_name"],
                mode,
                self._params["angles"],
                resolution,
                samples,
                extent,
                view_id,
            )

            volume_hit = False
            with self._cache_lock:
                volume_hit, cached_vol = self._volume_cache.get_with_hit(volume_key)
            if volume_hit:
                vol = cached_vol
            else:
                if mode.startswith("slice"):
                    vol = _generate_slice_volume(view_id, resolution, extent, rotation)
                else:
                    vol = _generate_integral_volume(
                        view_id,
                        resolution,
                        extent,
                        rotation,
                        samples,
                        mode,
                        cancel_event=self._cancel_event,
                    )
                    if vol is None:
                        break
                if vol.size and not np.isfinite(vol).all():
                    vol = np.array([])
                with self._cache_lock:
                    self._volume_cache.set(volume_key, vol)

            max_abs = float(np.max(np.abs(vol))) if vol.size else 0.0
            iso_value = (iso_percent / 100.0) * max_abs if iso_percent > 0 else 0.0

            mesh_hit_pos = False
            mesh_hit_neg = False
            mesh_pos = None
            mesh_neg = None

            if iso_value > 0 and vol.size:
                pos_key = _mesh_cache_key(volume_key, iso_value, 1)
                neg_key = _mesh_cache_key(volume_key, iso_value, -1)
                with self._cache_lock:
                    mesh_hit_pos, cached_pos = self._mesh_cache.get_with_hit(pos_key)
                    mesh_hit_neg, cached_neg = self._mesh_cache.get_with_hit(neg_key)
                if mesh_hit_pos:
                    mesh_pos = cached_pos
                if mesh_hit_neg:
                    mesh_neg = cached_neg

                if mesh_pos is None:
                    mesh_pos = _extract_mesh(vol, iso_value, extent)
                    with self._cache_lock:
                        self._mesh_cache.set(pos_key, mesh_pos)
                if mesh_neg is None:
                    mesh_neg = _extract_mesh(vol, -iso_value, extent)
                    with self._cache_lock:
                        self._mesh_cache.set(neg_key, mesh_neg)

            view_time_ms = int((time.perf_counter() - view_start) * 1000)
            info = {
                "view_index": index,
                "volume_hit": volume_hit,
                "mesh_hit_pos": mesh_hit_pos,
                "mesh_hit_neg": mesh_hit_neg,
                "view_time_ms": view_time_ms,
                "quality_label": quality_label,
                "mode": mode,
                "resolution": resolution,
                "samples": samples,
                "iso_value": iso_value,
            }
            self.view_ready.emit(self._request_id, view_id, mesh_pos, mesh_neg, info)

        total_ms = int((time.perf_counter() - start_total) * 1000)
        self.log_line.emit(f"Render#{self._request_id} finished in {total_ms} ms")
        self.finished.emit(self._request_id)


def _rotation_matrix_xy(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.eye(4, dtype=np.float64)
    mat[0, 0] = c
    mat[0, 1] = -s
    mat[1, 0] = s
    mat[1, 1] = c
    return mat


def _rotation_matrix_xz(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.eye(4, dtype=np.float64)
    mat[0, 0] = c
    mat[0, 2] = -s
    mat[2, 0] = s
    mat[2, 2] = c
    return mat


def _rotation_matrix_xw(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.eye(4, dtype=np.float64)
    mat[0, 0] = c
    mat[0, 3] = -s
    mat[3, 0] = s
    mat[3, 3] = c
    return mat


def _rotation_matrix_yz(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.eye(4, dtype=np.float64)
    mat[1, 1] = c
    mat[1, 2] = -s
    mat[2, 1] = s
    mat[2, 2] = c
    return mat


def _rotation_matrix_yw(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.eye(4, dtype=np.float64)
    mat[1, 1] = c
    mat[1, 3] = -s
    mat[3, 1] = s
    mat[3, 3] = c
    return mat


def _rotation_matrix_zw(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.eye(4, dtype=np.float64)
    mat[2, 2] = c
    mat[2, 3] = -s
    mat[3, 2] = s
    mat[3, 3] = c
    return mat


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("4D Atomic Orbital Viewer")
        self.resize(1200, 720)

        self.state = AppState()
        self._ready = False
        self._extent = 6.0
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._trigger_scheduled_render)
        self._pending_quality_label = "Preview"

        self._volume_cache = LRUCache(max_items=16)
        self._mesh_cache = LRUCache(max_items=32)
        self._cache_lock = threading.Lock()

        self._render_request_id = 0
        self._active_request_id = 0
        self._cancel_event: threading.Event | None = None
        self._render_thread: QtCore.QThread | None = None
        self._render_worker: RenderWorker | None = None
        self._last_params: dict | None = None

        self._build_menu()
        self._build_status_bar()
        self._build_central()

        self._ready = True
        self.on_ui_changed()

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.addMenu("File")
        menu_bar.addMenu("Export")
        menu_bar.addMenu("View")
        menu_bar.addMenu("Help")

    def _build_status_bar(self) -> None:
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)

    def _build_central(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)

    def _build_left_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(8)

        layout.addWidget(build_title_label("Orbitals"))

        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Search orbitals...")
        self.search_input.textChanged.connect(self.on_ui_changed)
        layout.addWidget(self.search_input)

        self.orbital_list = QtWidgets.QListWidget()
        self.orbital_list.addItems(["1s", "2p (set A)", "3d (set A)", "4f (set A)"])
        self.orbital_list.setCurrentRow(0)
        self.orbital_list.currentTextChanged.connect(self._on_orbital_changed)
        layout.addWidget(self.orbital_list, 1)

        return panel

    def _build_center_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(panel)
        layout.setSpacing(10)

        self.projection_views = {
            "X": ProjectionViewWidget("Projection X (to YZW)", extent=self._extent),
            "Y": ProjectionViewWidget("Projection Y (to XZW)", extent=self._extent),
            "Z": ProjectionViewWidget("Projection Z (to XYW)", extent=self._extent),
            "W": ProjectionViewWidget("Projection W (to XYZ)", extent=self._extent),
        }

        layout.addWidget(self.projection_views["X"], 0, 0)
        layout.addWidget(self.projection_views["Y"], 0, 1)
        layout.addWidget(self.projection_views["Z"], 1, 0)
        layout.addWidget(self.projection_views["W"], 1, 1)

        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        return panel

    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(10)

        layout.addWidget(build_title_label("Controls"))

        slider_grid = QtWidgets.QGridLayout()
        slider_grid.setHorizontalSpacing(8)
        slider_grid.setVerticalSpacing(6)

        tooltip_map = {
            "xy": "rotate within the x-y plane (mix x and y; z,w unchanged)",
            "xz": "rotate within the x-z plane (mix x and z; y,w unchanged)",
            "xw": "rotate within the x-w plane (mix x and w; y,z unchanged)",
            "yz": "rotate within the y-z plane (mix y and z; x,w unchanged)",
            "yw": "rotate within the y-w plane (mix y and w; x,z unchanged)",
            "zw": "rotate within the z-w plane (mix z and w; x,y unchanged)",
        }

        self.angle_controls = {}
        for row, name in enumerate(["xy", "xz", "xw", "yz", "yw", "zw"]):
            widgets = build_labeled_slider(name, tooltip_map[name], self._on_angle_changed)
            self.angle_controls[name] = widgets
            widgets["slider"].sliderReleased.connect(
                lambda name=name: self._on_angle_released(name)
            )
            slider_grid.addWidget(widgets["label"], row, 0)
            slider_grid.addWidget(widgets["slider"], row, 1)
            slider_grid.addWidget(widgets["value_label"], row, 2)
            slider_grid.addWidget(widgets["info_button"], row, 3)

        layout.addLayout(slider_grid)

        self.projection_mode = QtWidgets.QComboBox()
        self.projection_mode.addItems(
            [
                "slice (fast)",
                "integral ψ (strict)",
                "integral |ψ| (stable)",
                "max |ψ| (visual)",
            ]
        )
        self.projection_mode.currentTextChanged.connect(self.on_ui_changed)

        self.isosurface_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.isosurface_slider.setRange(0, 100)
        self.isosurface_slider.setValue(0)
        self.isosurface_value = QtWidgets.QLabel("0%")
        self.isosurface_slider.valueChanged.connect(self._on_isosurface_changed)
        self.isosurface_slider.sliderReleased.connect(self._on_isosurface_released)

        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(0)
        self.opacity_value = QtWidgets.QLabel("0%")
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.opacity_slider.sliderReleased.connect(self._on_opacity_released)

        self.resolution_combo = QtWidgets.QComboBox()
        self.resolution_combo.addItems(["64", "96", "128"])
        self.resolution_combo.currentTextChanged.connect(self.on_ui_changed)

        self.samples_combo = QtWidgets.QComboBox()
        self.samples_combo.addItems(["32", "64", "128", "256"])
        self.samples_combo.currentTextChanged.connect(self.on_ui_changed)

        self.preview_quality_combo = QtWidgets.QComboBox()
        self.preview_quality_combo.addItems(
            [
                "Fast (N=64, samples=32)",
                "Medium (N=96, samples=64)",
            ]
        )
        self.preview_quality_combo.currentTextChanged.connect(self.on_ui_changed)

        self.final_quality_combo = QtWidgets.QComboBox()
        self.final_quality_combo.addItems(
            [
                "High (N=128, samples=128)",
                "Ultra (N=160, samples=256)",
            ]
        )
        self.final_quality_combo.currentTextChanged.connect(self.on_ui_changed)

        self.auto_refine = QtWidgets.QCheckBox("Auto Refine")
        self.auto_refine.setChecked(True)
        self.auto_refine.toggled.connect(self.on_ui_changed)

        self.live_update = QtWidgets.QCheckBox("Live Update")
        self.live_update.setChecked(True)
        self.live_update.toggled.connect(self.on_ui_changed)

        self.reset_button = QtWidgets.QPushButton("Reset Angles")
        self.reset_button.clicked.connect(self._reset_angles)

        self.render_button = QtWidgets.QPushButton("Render Now")
        self.render_button.clicked.connect(self._on_render_now)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel_render)

        layout.addWidget(QtWidgets.QLabel("Projection Mode"))
        layout.addWidget(self.projection_mode)

        iso_layout = QtWidgets.QHBoxLayout()
        iso_layout.addWidget(QtWidgets.QLabel("Isosurface Level"))
        iso_layout.addStretch(1)
        iso_layout.addWidget(self.isosurface_value)
        layout.addLayout(iso_layout)
        layout.addWidget(self.isosurface_slider)

        opacity_layout = QtWidgets.QHBoxLayout()
        opacity_layout.addWidget(QtWidgets.QLabel("Opacity"))
        opacity_layout.addStretch(1)
        opacity_layout.addWidget(self.opacity_value)
        layout.addLayout(opacity_layout)
        layout.addWidget(self.opacity_slider)

        layout.addWidget(QtWidgets.QLabel("Resolution"))
        layout.addWidget(self.resolution_combo)

        layout.addWidget(QtWidgets.QLabel("Integral Samples"))
        layout.addWidget(self.samples_combo)

        layout.addWidget(QtWidgets.QLabel("Preview Quality"))
        layout.addWidget(self.preview_quality_combo)

        layout.addWidget(QtWidgets.QLabel("Final Quality"))
        layout.addWidget(self.final_quality_combo)

        layout.addWidget(self.auto_refine)
        layout.addWidget(self.live_update)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.render_button)
        layout.addWidget(self.cancel_button)

        self.log_panel = QtWidgets.QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setPlaceholderText("UI event log...")
        layout.addWidget(self.log_panel, 1)

        return panel

    def _on_angle_changed(self, name: str, value: int) -> None:
        self.state.angles[name] = value
        self._handle_value_change("angles")

    def _on_angle_released(self, name: str) -> None:
        self.state.angles[name] = self.angle_controls[name]["slider"].value()
        self._handle_slider_released("angles")

    def _on_orbital_changed(self, text: str) -> None:
        self.state.orbital_name = text
        self.on_ui_changed()

    def _on_isosurface_changed(self, value: int) -> None:
        self.isosurface_value.setText(f"{value}%")
        self.state.iso_percent = value
        self._handle_value_change("iso")

    def _on_isosurface_released(self) -> None:
        self._handle_slider_released("iso")

    def _on_opacity_changed(self, value: int) -> None:
        self.opacity_value.setText(f"{value}%")
        self.state.opacity_percent = value
        self._handle_value_change("opacity")

    def _on_opacity_released(self) -> None:
        self._handle_slider_released("opacity")

    def _reset_angles(self) -> None:
        for name, widgets in self.angle_controls.items():
            widgets["slider"].blockSignals(True)
            widgets["slider"].setValue(0)
            widgets["slider"].blockSignals(False)
            self.state.angles[name] = 0
        self.on_ui_changed()

    def _on_render_now(self) -> None:
        self._start_render(self._resolve_quality_label(final=True), "manual")

    def _on_cancel_render(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self.log_panel.appendPlainText("Cancel requested")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._render_timer.isActive():
            self._render_timer.stop()
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._render_thread is not None and self._render_thread.isRunning():
            self._render_thread.quit()
            self._render_thread.wait(1000)
        for view in self.projection_views.values():
            view.plotter.close()
        super().closeEvent(event)

    def on_ui_changed(self, schedule_render: bool = True) -> None:
        if not self._ready:
            return

        self.state.projection_mode = self.projection_mode.currentText()
        self.state.resolution = int(self.resolution_combo.currentText())
        self.state.integral_samples = int(self.samples_combo.currentText())
        self.state.live_update = self.live_update.isChecked()
        self.state.auto_refine = self.auto_refine.isChecked()
        self.state.preview_quality = self.preview_quality_combo.currentText()
        self.state.final_quality = self.final_quality_combo.currentText()

        if self.orbital_list.currentItem() is not None:
            self.state.orbital_name = self.orbital_list.currentItem().text()

        self.status_bar.showMessage(f"Mode={self.state.projection_mode} | Quality=Idle")
        self.log_panel.appendPlainText(self.state.log_line())

        change_kind = self._detect_change_kind()
        if not self.state.live_update or not schedule_render:
            return

        if change_kind == "opacity":
            self._apply_opacity_only()
            return

        quality_label = self._resolve_quality_label(final=True)
        self._schedule_render(quality_label)

    def _detect_change_kind(self) -> str:
        current = self._current_params()
        if self._last_params is None:
            self._last_params = current
            return "volume"
        previous = self._last_params
        self._last_params = current

        def changed(keys: list[str]) -> bool:
            return any(previous[key] != current[key] for key in keys)

        volume_keys = [
            "orbital_name",
            "projection_mode",
            "angles",
            "resolution",
            "integral_samples",
            "preview_quality",
            "final_quality",
            "auto_refine",
        ]
        if changed(volume_keys):
            return "volume"
        if previous["iso_percent"] != current["iso_percent"]:
            return "iso"
        if previous["opacity_percent"] != current["opacity_percent"]:
            return "opacity"
        return "none"

    def _current_params(self) -> dict:
        return {
            "orbital_name": self.state.orbital_name,
            "projection_mode": self.state.projection_mode,
            "angles": dict(self.state.angles),
            "resolution": self.state.resolution,
            "integral_samples": self.state.integral_samples,
            "iso_percent": self.state.iso_percent,
            "opacity_percent": self.state.opacity_percent,
            "preview_quality": self.state.preview_quality,
            "final_quality": self.state.final_quality,
            "auto_refine": self.state.auto_refine,
        }

    def _handle_value_change(self, change_kind: str) -> None:
        self.on_ui_changed(schedule_render=False)
        if not self.state.live_update:
            return
        if change_kind == "opacity":
            if self.state.live_update:
                self._apply_opacity_only()
            return
        if change_kind == "iso":
            self._schedule_render(self._resolve_quality_label(final=False))
            return
        self._schedule_render(self._resolve_quality_label(final=False))

    def _handle_slider_released(self, change_kind: str) -> None:
        if not self.state.auto_refine:
            return
        if not self.state.live_update:
            return
        if change_kind == "opacity":
            self._apply_opacity_only()
            return
        self._start_render(self._resolve_quality_label(final=True), "release")

    def _resolve_quality_label(self, final: bool) -> str:
        if not self.state.auto_refine:
            return "Manual"
        return "Final" if final else "Preview"

    def _quality_to_settings(self, quality_label: str) -> tuple[int, int]:
        if quality_label == "Preview":
            current = self.preview_quality_combo.currentText()
            mapping = {
                "Fast (N=64, samples=32)": (64, 32),
                "Medium (N=96, samples=64)": (96, 64),
            }
            return mapping[current]
        if quality_label == "Final":
            current = self.final_quality_combo.currentText()
            mapping = {
                "High (N=128, samples=128)": (128, 128),
                "Ultra (N=160, samples=256)": (160, 256),
            }
            return mapping[current]
        return self.state.resolution, self.state.integral_samples

    def _schedule_render(self, quality_label: str) -> None:
        self._pending_quality_label = quality_label
        if self._render_timer.isActive():
            return
        self._render_timer.start(60)

    def _trigger_scheduled_render(self) -> None:
        self._start_render(self._pending_quality_label, "scheduled")

    def _start_render(self, quality_label: str, reason: str) -> None:
        if not self.isVisible():
            return

        if self._cancel_event is not None:
            self._cancel_event.set()
        if (
            self._render_thread is not None
            and isValid(self._render_thread)
            and self._render_thread.isRunning()
        ):
            self._render_thread.quit()
            self._render_thread.wait(250)

        self._render_request_id += 1
        request_id = self._render_request_id
        self._active_request_id = request_id
        self._cancel_event = threading.Event()

        resolution, samples = self._quality_to_settings(quality_label)
        params = {
            "orbital_name": self.state.orbital_name,
            "projection_mode": self.state.projection_mode,
            "mode": self.state.projection_mode,
            "angles": dict(self.state.angles),
            "resolution": resolution,
            "samples": samples,
            "iso_percent": self.state.iso_percent,
            "opacity": self.state.opacity_percent / 100.0,
            "extent": self._extent,
            "quality_label": quality_label,
        }

        self._update_status(params, 0, "Cache=--")
        self.log_panel.appendPlainText(
            f"Render#{request_id} queued ({quality_label}, reason={reason})"
        )

        self._render_thread = QtCore.QThread()
        self._render_worker = RenderWorker(
            request_id,
            params,
            self._volume_cache,
            self._mesh_cache,
            self._cache_lock,
            self._cancel_event,
        )
        self._render_worker.moveToThread(self._render_thread)
        self._render_thread.started.connect(self._render_worker.run)
        self._render_worker.view_ready.connect(self._on_view_ready)
        self._render_worker.log_line.connect(self._append_log)
        self._render_worker.finished.connect(self._on_render_finished)
        self._render_worker.finished.connect(self._render_thread.quit)
        self._render_worker.finished.connect(self._render_worker.deleteLater)
        self._render_thread.finished.connect(self._render_thread.deleteLater)
        self._render_thread.finished.connect(self._on_render_thread_finished)
        self._render_thread.start()

    def _append_log(self, message: str) -> None:
        self.log_panel.appendPlainText(message)

    def _on_view_ready(
        self,
        request_id: int,
        view_id: str,
        mesh_pos: object,
        mesh_neg: object,
        info: dict,
    ) -> None:
        if request_id != self._active_request_id:
            return

        quality_label = info["quality_label"]
        cache_text = "Cache vol={v} mesh={m}".format(
            v="HIT" if info["volume_hit"] else "MISS",
            m="HIT" if (info["mesh_hit_pos"] and info["mesh_hit_neg"]) else "MISS",
        )
        self._update_status(info, info["view_index"], cache_text)

        mesh_pos_data = mesh_pos if isinstance(mesh_pos, tuple) else None
        mesh_neg_data = mesh_neg if isinstance(mesh_neg, tuple) else None
        self.projection_views[view_id].set_meshes(
            mesh_pos_data,
            mesh_neg_data,
            self.state.opacity_percent / 100.0,
        )

        self.log_panel.appendPlainText(
            "View {view} volume: cache {vol}".format(
                view=view_id,
                vol="HIT" if info["volume_hit"] else "MISS",
            )
        )
        if info["iso_value"] <= 0:
            self.log_panel.appendPlainText(f"View {view_id} mesh: skipped (iso=0)")
        else:
            self.log_panel.appendPlainText(
                "View {view} mesh: cache {mesh}".format(
                    view=view_id,
                    mesh="HIT" if (info["mesh_hit_pos"] and info["mesh_hit_neg"]) else "MISS",
                )
            )
        self.log_panel.appendPlainText(
            f"View {view_id} done in {info['view_time_ms']} ms"
        )

    def _on_render_finished(self, request_id: int) -> None:
        if request_id != self._active_request_id:
            return
        self.status_bar.showMessage(
            f"Mode={self.state.projection_mode} | Quality=Idle"
        )

    def _on_render_thread_finished(self) -> None:
        self._render_thread = None
        self._render_worker = None

    def _apply_opacity_only(self) -> None:
        opacity = self.state.opacity_percent / 100.0
        for view in self.projection_views.values():
            view.set_opacity(opacity)
        self.log_panel.appendPlainText("Opacity only: actor updated")

    def _update_status(self, info: dict, view_index: int, cache_text: str) -> None:
        mode = info["mode"] if "mode" in info else self.state.projection_mode
        quality_label = info["quality_label"] if "quality_label" in info else "Manual"
        self.status_bar.showMessage(
            f"Mode={mode} | Quality={quality_label} | Computing... ({view_index}/4) | {cache_text}"
        )

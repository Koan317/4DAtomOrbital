import time
import threading
from collections import OrderedDict
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid
from skimage import measure

from app_state import AppState
from orbitals import get_orbital_by_display_name, list_orbitals, run_orbital_self_check
from ui_widgets import ProjectionViewWidget, build_labeled_slider, build_title_label


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
    sample_value = samples if not projection_mode.startswith("切片") else 0
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
    orbital_name: str,
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

    orbital = get_orbital_by_display_name(orbital_name)
    return orbital.evaluate(xr, yr, zr, wr)


def _generate_integral_volume(
    view_id: str,
    resolution: int,
    extent: float,
    rotation: np.ndarray,
    samples: int,
    mode: str,
    orbital_name: str,
    cancel_event: threading.Event | None = None,
) -> np.ndarray | None:
    rotation = np.ascontiguousarray(rotation, dtype=np.float64)
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
    orbital = get_orbital_by_display_name(orbital_name)

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

        p0 = points[:, 0]
        p1 = points[:, 1]
        p2 = points[:, 2]
        p3 = points[:, 3]
        xr = p0 * rotation[0, 0] + p1 * rotation[0, 1] + p2 * rotation[0, 2] + p3 * rotation[0, 3]
        yr = p0 * rotation[1, 0] + p1 * rotation[1, 1] + p2 * rotation[1, 2] + p3 * rotation[1, 3]
        zr = p0 * rotation[2, 0] + p1 * rotation[2, 1] + p2 * rotation[2, 2] + p3 * rotation[2, 3]
        wr = p0 * rotation[3, 0] + p1 * rotation[3, 1] + p2 * rotation[3, 2] + p3 * rotation[3, 3]
        psi = orbital.evaluate(xr, yr, zr, wr)

        if mode.startswith("积分 ψ"):
            accumulator += psi * weight
        elif mode.startswith("积分 |ψ|"):
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
            "渲染#{rid} 开始（{quality}）：模式={mode}, N={res}, M={samples}".format(
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
                if mode.startswith("切片"):
                    vol = _generate_slice_volume(
                        view_id,
                        resolution,
                        extent,
                        rotation,
                        self._params["orbital_name"],
                    )
                else:
                    vol = _generate_integral_volume(
                        view_id,
                        resolution,
                        extent,
                        rotation,
                        samples,
                        mode,
                        self._params["orbital_name"],
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
        self.log_line.emit(f"渲染#{self._request_id} 完成，用时 {total_ms} ms")
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
        self.setWindowTitle("4D 原子轨道查看器")
        self.resize(1200, 720)

        self.state = AppState()
        self._ready = False
        self._extent = 14.4
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._trigger_scheduled_render)
        self._pending_quality_label = "预览"
        self._iso_timer = QtCore.QTimer(self)
        self._iso_timer.setSingleShot(True)
        self._iso_timer.timeout.connect(lambda: self._update_mesh_only("预览"))

        self._volume_cache = LRUCache(max_items=16)
        self._mesh_cache = LRUCache(max_items=32)
        self._cache_lock = threading.Lock()

        self._render_request_id = 0
        self._active_request_id = 0
        self._cancel_event: threading.Event | None = None
        self._render_thread: QtCore.QThread | None = None
        self._render_worker: RenderWorker | None = None
        self._retired_threads: list[tuple[QtCore.QThread, RenderWorker, threading.Event]] = []
        self._last_params: dict | None = None
        self._pending_render_request: tuple[str, str] | None = None

        self._build_status_bar()
        self._build_central()

        self._run_orbital_self_check()
        self._ready = True
        self.on_ui_changed()

    def _run_orbital_self_check(self) -> None:
        warnings = run_orbital_self_check()
        if not warnings:
            self.log_panel.appendPlainText("轨道自检：通过")
            return
        for warning in warnings:
            self.log_panel.appendPlainText(f"轨道自检警告：{warning}")

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

        left_panel = self._build_left_panel()
        center_panel = self._build_center_panel()
        right_panel = self._build_right_panel()

        left_panel.setMinimumWidth(180)
        right_panel.setMinimumWidth(240)

        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([200, 650, 300])

    def _build_left_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(8)

        layout.addWidget(build_title_label("轨道"))

        self.orbital_list = QtWidgets.QListWidget()
        self.orbital_list.addItems([orb.display_name for orb in list_orbitals()])
        self.orbital_list.setCurrentRow(0)
        self.orbital_list.currentTextChanged.connect(self._on_orbital_changed)
        layout.addWidget(self.orbital_list, 1)

        return panel

    def _build_center_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(panel)
        layout.setSpacing(10)

        self.projection_views = {
            "X": ProjectionViewWidget("投影 X（到 YZW）", extent=self._extent),
            "Y": ProjectionViewWidget("投影 Y（到 XZW）", extent=self._extent),
            "Z": ProjectionViewWidget("投影 Z（到 XYW）", extent=self._extent),
            "W": ProjectionViewWidget("投影 W（到 XYZ）", extent=self._extent),
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

        layout.addWidget(build_title_label("控制"))

        slider_grid = QtWidgets.QGridLayout()
        slider_grid.setHorizontalSpacing(8)
        slider_grid.setVerticalSpacing(6)

        tooltip_map = {
            "xy": "在 x-y 平面内旋转（混合 x 与 y；z,w 不变）",
            "xz": "在 x-z 平面内旋转（混合 x 与 z；y,w 不变）",
            "xw": "在 x-w 平面内旋转（混合 x 与 w；y,z 不变）",
            "yz": "在 y-z 平面内旋转（混合 y 与 z；x,w 不变）",
            "yw": "在 y-w 平面内旋转（混合 y 与 w；x,z 不变）",
            "zw": "在 z-w 平面内旋转（混合 z 与 w；x,y 不变）",
        }

        self.angle_controls = {}
        for row, name in enumerate(["xy", "xz", "xw", "yz", "yw", "zw"]):
            widgets = build_labeled_slider(name, tooltip_map[name], self._on_angle_changed)
            self.angle_controls[name] = widgets
            widgets["slider"].sliderReleased.connect(
                lambda name=name: self._on_angle_released(name)
            )
            slider_grid.addWidget(widgets["label"], row, 0)
            slider_grid.addWidget(widgets["minus_button"], row, 1)
            slider_grid.addWidget(widgets["slider"], row, 2)
            slider_grid.addWidget(widgets["plus_button"], row, 3)
            slider_grid.addWidget(widgets["value_label"], row, 4)
            slider_grid.addWidget(widgets["info_button"], row, 5)

        layout.addLayout(slider_grid)

        self.projection_mode = QtWidgets.QComboBox()
        self.projection_mode.addItems(
            [
                "切片（快速）",
                "积分 ψ（严格）",
                "积分 |ψ|（稳定）",
                "最大 |ψ|（可视化）",
            ]
        )
        self.projection_mode.currentTextChanged.connect(self.on_ui_changed)

        self.isosurface_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.isosurface_slider.setRange(0, 100)
        self.isosurface_slider.setValue(0)
        self.isosurface_value = QtWidgets.QLabel("0%")
        self.isosurface_slider.valueChanged.connect(self._on_isosurface_changed)
        self.isosurface_slider.sliderReleased.connect(self._on_isosurface_released)

        self.resolution_combo = QtWidgets.QComboBox()
        self.resolution_combo.addItems(["64", "96", "128"])
        self.resolution_combo.currentTextChanged.connect(self.on_ui_changed)

        self.samples_combo = QtWidgets.QComboBox()
        self.samples_combo.addItems(["32", "64", "128", "256"])
        self.samples_combo.currentTextChanged.connect(self.on_ui_changed)

        self.preview_quality_combo = QtWidgets.QComboBox()
        self.preview_quality_combo.addItems(
            [
                "快速（N=64，采样=32）",
                "中等（N=96，采样=64）",
            ]
        )
        self.preview_quality_combo.currentTextChanged.connect(self.on_ui_changed)

        self.final_quality_combo = QtWidgets.QComboBox()
        self.final_quality_combo.addItems(
            [
                "高（N=128，采样=128）",
                "超高（N=160，采样=256）",
            ]
        )
        self.final_quality_combo.currentTextChanged.connect(self.on_ui_changed)

        self.auto_refine = QtWidgets.QCheckBox("自动精细化")
        self.auto_refine.setChecked(True)
        self.auto_refine.toggled.connect(self.on_ui_changed)

        self.live_update = QtWidgets.QCheckBox("实时更新")
        self.live_update.setChecked(True)
        self.live_update.toggled.connect(self.on_ui_changed)

        self.reset_button = QtWidgets.QPushButton("重置角度")
        self.reset_button.clicked.connect(self._reset_angles)

        layout.addWidget(QtWidgets.QLabel("投影模式"))
        layout.addWidget(self.projection_mode)

        iso_layout = QtWidgets.QHBoxLayout()
        iso_layout.addWidget(QtWidgets.QLabel("等值面级别"))
        iso_layout.addStretch(1)
        iso_layout.addWidget(self.isosurface_value)
        layout.addLayout(iso_layout)
        layout.addWidget(self.isosurface_slider)

        layout.addWidget(QtWidgets.QLabel("分辨率"))
        layout.addWidget(self.resolution_combo)

        layout.addWidget(QtWidgets.QLabel("积分采样"))
        layout.addWidget(self.samples_combo)

        layout.addWidget(QtWidgets.QLabel("预览质量"))
        layout.addWidget(self.preview_quality_combo)

        layout.addWidget(QtWidgets.QLabel("最终质量"))
        layout.addWidget(self.final_quality_combo)

        layout.addWidget(self.auto_refine)
        layout.addWidget(self.live_update)
        layout.addWidget(self.reset_button)
        self.log_panel = QtWidgets.QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setPlaceholderText("界面事件日志...")
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

    def _reset_angles(self) -> None:
        for name, widgets in self.angle_controls.items():
            widgets["slider"].blockSignals(True)
            widgets["slider"].setValue(0)
            widgets["slider"].blockSignals(False)
            self.state.angles[name] = 0
            widgets["value_label"].setText("000°")
        self.on_ui_changed()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._render_timer.isActive():
            self._render_timer.stop()
        if self._iso_timer.isActive():
            self._iso_timer.stop()
        if self._cancel_event is not None:
            self._cancel_event.set()
        if (
            self._render_thread is not None
            and isValid(self._render_thread)
            and self._render_thread.isRunning()
        ):
            self._render_thread.quit()
            self._render_thread.wait(1000)
        self._cleanup_retired_threads(force=True)
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

        self.status_bar.showMessage(f"模式={self.state.projection_mode} | 质量=空闲")
        self.log_panel.appendPlainText(self.state.log_line())

        change_kind = self._detect_change_kind()
        if not self.state.live_update or not schedule_render:
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
        return "none"

    def _current_params(self) -> dict:
        return {
            "orbital_name": self.state.orbital_name,
            "projection_mode": self.state.projection_mode,
            "angles": dict(self.state.angles),
            "resolution": self.state.resolution,
            "integral_samples": self.state.integral_samples,
            "iso_percent": self.state.iso_percent,
            "preview_quality": self.state.preview_quality,
            "final_quality": self.state.final_quality,
            "auto_refine": self.state.auto_refine,
        }

    def _handle_value_change(self, change_kind: str) -> None:
        self.on_ui_changed(schedule_render=False)
        if not self.state.live_update:
            return
        if change_kind == "iso":
            self._schedule_iso_mesh_update()
            return
        self._schedule_render(self._resolve_quality_label(final=False))

    def _handle_slider_released(self, change_kind: str) -> None:
        if not self.state.auto_refine:
            return
        if not self.state.live_update:
            return
        self._start_render(self._resolve_quality_label(final=True), "release")

    def _resolve_quality_label(self, final: bool) -> str:
        if not self.state.auto_refine:
            return "手动"
        return "最终" if final else "预览"

    def _quality_to_settings(self, quality_label: str) -> tuple[int, int]:
        if quality_label == "预览":
            current = self.preview_quality_combo.currentText()
            mapping = {
                "快速（N=64，采样=32）": (64, 32),
                "中等（N=96，采样=64）": (96, 64),
            }
            return mapping[current]
        if quality_label == "最终":
            current = self.final_quality_combo.currentText()
            mapping = {
                "高（N=128，采样=128）": (128, 128),
                "超高（N=160，采样=256）": (160, 256),
            }
            return mapping[current]
        return self.state.resolution, self.state.integral_samples

    def _schedule_render(self, quality_label: str) -> None:
        self._pending_quality_label = quality_label
        if self._render_timer.isActive():
            return
        self._render_timer.start(60)

    def _schedule_iso_mesh_update(self) -> None:
        if self._iso_timer.isActive():
            self._iso_timer.stop()
        self._iso_timer.start(60)

    def _trigger_scheduled_render(self) -> None:
        self._start_render(self._pending_quality_label, "scheduled")

    def _update_mesh_only(self, quality_label: str) -> None:
        if not self.isVisible():
            return
        resolution, samples = self._quality_to_settings(quality_label)
        mode = self.state.projection_mode
        angles = dict(self.state.angles)
        extent = self._extent
        iso_percent = self.state.iso_percent

        volume_hits: dict[str, np.ndarray] = {}
        for view_id in ["X", "Y", "Z", "W"]:
            volume_key = _volume_cache_key(
                self.state.orbital_name,
                mode,
                angles,
                resolution,
                samples,
                extent,
                view_id,
            )
            with self._cache_lock:
                hit, cached_vol = self._volume_cache.get_with_hit(volume_key)
            if not hit:
                self._start_render(quality_label, "iso-mesh-miss")
                return
            volume_hits[view_id] = cached_vol

        for view_id, vol in volume_hits.items():
            max_abs = float(np.max(np.abs(vol))) if vol.size else 0.0
            iso_value = (iso_percent / 100.0) * max_abs if iso_percent > 0 else 0.0
            mesh_pos = _extract_mesh(vol, iso_value, extent) if iso_value > 0 else None
            mesh_neg = _extract_mesh(vol, -iso_value, extent) if iso_value > 0 else None
            self.projection_views[view_id].set_meshes(mesh_pos, mesh_neg, 1.0)
        self.log_panel.appendPlainText("等值面仅网格更新（预览）")

    def _start_render(self, quality_label: str, reason: str) -> None:
        if not self.isVisible():
            return

        if (
            self._render_thread is not None
            and self._render_worker is not None
            and self._cancel_event is not None
            and isValid(self._render_thread)
            and self._render_thread.isRunning()
        ):
            self._pending_render_request = (quality_label, reason)
            should_cancel = reason == "release" or quality_label == "最终"
            if should_cancel:
                self._cancel_event.set()
                message = f"渲染请求已合并并取消当前计算（{quality_label}，原因={reason}）"
            else:
                message = f"渲染请求已合并（不取消当前计算，{quality_label}，原因={reason}）"
            self.log_panel.appendPlainText(message)
            return

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
            "extent": self._extent,
            "quality_label": quality_label,
        }

        reason_map = {
            "manual": "手动",
            "release": "释放",
            "scheduled": "定时",
            "iso-mesh-miss": "等值面缓存未命中",
        }
        reason_text = reason_map.get(reason, reason)

        self._update_status(params, 0, "缓存=--")
        self.log_panel.appendPlainText(
            f"渲染#{request_id} 已排队（{quality_label}，原因={reason_text}）"
        )

        self._render_thread = QtCore.QThread(self)
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
        cache_text = "缓存 体积={v} 网格={m}".format(
            v="命中" if info["volume_hit"] else "未命中",
            m="命中" if (info["mesh_hit_pos"] and info["mesh_hit_neg"]) else "未命中",
        )
        self._update_status(info, info["view_index"], cache_text)

        mesh_pos_data = mesh_pos if isinstance(mesh_pos, tuple) else None
        mesh_neg_data = mesh_neg if isinstance(mesh_neg, tuple) else None
        self.projection_views[view_id].set_meshes(mesh_pos_data, mesh_neg_data, 1.0)

        self.log_panel.appendPlainText(
            "视图 {view} 体积：缓存 {vol}".format(
                view=view_id,
                vol="命中" if info["volume_hit"] else "未命中",
            )
        )
        if info["iso_value"] <= 0:
            self.log_panel.appendPlainText(f"视图 {view_id} 网格：跳过（iso=0）")
        else:
            self.log_panel.appendPlainText(
                "视图 {view} 网格：缓存 {mesh}".format(
                    view=view_id,
                    mesh="命中" if (info["mesh_hit_pos"] and info["mesh_hit_neg"]) else "未命中",
                )
            )
        self.log_panel.appendPlainText(
            f"视图 {view_id} 完成，用时 {info['view_time_ms']} ms"
        )

    def _on_render_finished(self, request_id: int) -> None:
        if request_id != self._active_request_id:
            return
        self.status_bar.showMessage(f"模式={self.state.projection_mode} | 质量=空闲")

    def _on_render_thread_finished(self) -> None:
        self._render_thread = None
        self._render_worker = None
        try:
            self._cleanup_retired_threads(force=False)
        except RuntimeError:
            self._retired_threads = []
        if self._pending_render_request is not None:
            quality_label, reason = self._pending_render_request
            self._pending_render_request = None
            self._start_render(quality_label, reason)

    def _cleanup_retired_threads(self, force: bool) -> None:
        remaining: list[tuple[QtCore.QThread, RenderWorker, threading.Event]] = []
        for thread, worker, cancel_event in self._retired_threads:
            if thread is None or not isValid(thread):
                continue
            try:
                running = thread.isRunning()
            except RuntimeError:
                continue
            if force:
                cancel_event.set()
                if running:
                    thread.quit()
                    thread.wait(1000)
                continue
            if running:
                remaining.append((thread, worker, cancel_event))
            else:
                try:
                    thread.quit()
                except RuntimeError:
                    continue
        self._retired_threads = remaining

    def _update_status(self, info: dict, view_index: int, cache_text: str) -> None:
        mode = info["mode"] if "mode" in info else self.state.projection_mode
        quality_label = info["quality_label"] if "quality_label" in info else "手动"
        self.status_bar.showMessage(
            f"模式={mode} | 质量={quality_label} | 计算中... ({view_index}/4) | {cache_text}"
        )

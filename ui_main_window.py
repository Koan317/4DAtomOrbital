import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app_state import AppState
from ui_widgets import (
    ProjectionViewWidget,
    build_labeled_slider,
    build_title_label,
)


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
        self._render_timer.timeout.connect(self.render_all_views)

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

        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(0)
        self.opacity_value = QtWidgets.QLabel("0%")
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)

        self.resolution_combo = QtWidgets.QComboBox()
        self.resolution_combo.addItems(["64", "96", "128"])
        self.resolution_combo.currentTextChanged.connect(self.on_ui_changed)

        self.samples_combo = QtWidgets.QComboBox()
        self.samples_combo.addItems(["32", "64", "128", "256"])
        self.samples_combo.currentTextChanged.connect(self.on_ui_changed)

        self.live_update = QtWidgets.QCheckBox("Live Update")
        self.live_update.setChecked(True)
        self.live_update.toggled.connect(self.on_ui_changed)

        self.reset_button = QtWidgets.QPushButton("Reset Angles")
        self.reset_button.clicked.connect(self._reset_angles)

        self.render_button = QtWidgets.QPushButton("Render Now")
        self.render_button.clicked.connect(self._on_render_now)

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

        layout.addWidget(self.live_update)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.render_button)

        self.log_panel = QtWidgets.QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setPlaceholderText("UI event log...")
        layout.addWidget(self.log_panel, 1)

        return panel

    def _on_angle_changed(self, name: str, value: int) -> None:
        self.state.angles[name] = value
        self.on_ui_changed()

    def _on_orbital_changed(self, text: str) -> None:
        self.state.orbital_name = text
        self.on_ui_changed()

    def _on_isosurface_changed(self, value: int) -> None:
        self.isosurface_value.setText(f"{value}%")
        self.state.iso_percent = value
        self.on_ui_changed()

    def _on_opacity_changed(self, value: int) -> None:
        self.opacity_value.setText(f"{value}%")
        self.state.opacity_percent = value
        self.on_ui_changed()

    def _reset_angles(self) -> None:
        for name, widgets in self.angle_controls.items():
            widgets["slider"].setValue(0)
            self.state.angles[name] = 0
        self.on_ui_changed()

    def _on_render_now(self) -> None:
        self.render_all_views()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._render_timer.isActive():
            self._render_timer.stop()
        for view in self.projection_views.values():
            view.plotter.close()
        super().closeEvent(event)

    def on_ui_changed(self) -> None:
        if not self._ready:
            return

        self.state.projection_mode = self.projection_mode.currentText()
        self.state.resolution = int(self.resolution_combo.currentText())
        self.state.integral_samples = int(self.samples_combo.currentText())
        self.state.live_update = self.live_update.isChecked()

        if self.orbital_list.currentItem() is not None:
            self.state.orbital_name = self.orbital_list.currentItem().text()

        self.status_bar.showMessage(self.state.status_text())
        self.log_panel.appendPlainText(self.state.log_line())

        if self.state.live_update:
            self._schedule_render()

    def render_all_views(self) -> None:
        if not self.isVisible():
            return

        mode = self.state.projection_mode
        mode_label = "slice" if mode.startswith("slice") else mode
        iso_percent = self.state.iso_percent
        opacity = self.state.opacity_percent / 100.0
        resolution = self.state.resolution

        self.log_panel.appendPlainText(
            f"Render: N={resolution}, L={self._extent:.1f}, iso={iso_percent}%, "
            f"opacity={self.state.opacity_percent}%, mode={mode_label}"
        )

        if not mode.startswith("slice"):
            self.log_panel.appendPlainText(f"Render skipped: mode '{mode}' not implemented yet.")
            return

        for view_id in ["X", "Y", "Z", "W"]:
            vol = self._generate_slice_volume(view_id, resolution)
            max_abs = float(np.max(np.abs(vol))) if vol.size else 0.0
            iso_value = (iso_percent / 100.0) * max_abs if iso_percent > 0 else 0.0
            self.projection_views[view_id].set_volume_and_render(vol, iso_value, opacity)

        self.log_panel.appendPlainText("Render done: View X/Y/Z/W updated")

    def _schedule_render(self) -> None:
        if self._render_timer.isActive():
            return
        self._render_timer.start(30)

    def _generate_slice_volume(self, view_id: str, resolution: int) -> np.ndarray:
        coords = np.linspace(-self._extent, self._extent, resolution)
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

        r = np.sqrt(x**2 + y**2 + z**2 + w**2)
        return np.exp(-r) * (x**2 - y**2 + 0.35 * z - 0.25 * w)

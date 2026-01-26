from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor
from skimage import measure
import numpy as np
import pyvista as pv


def build_title_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    font = QtGui.QFont()
    font.setBold(True)
    label.setFont(font)
    return label


def build_placeholder_panel(title: str) -> QtWidgets.QWidget:
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    title_label = QtWidgets.QLabel(title)
    title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
    title_label.setStyleSheet("color: #dddddd;")

    placeholder = QtWidgets.QFrame()
    placeholder.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
    placeholder.setStyleSheet("background-color: #202020; border: 1px solid #303030;")

    layout.addWidget(title_label)
    layout.addWidget(placeholder, 1)
    return panel


def build_labeled_slider(
    name: str,
    tooltip: str,
    on_change,
) -> dict:
    label = QtWidgets.QLabel(name)
    label.setMinimumWidth(32)

    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setRange(0, 360)
    slider.setValue(0)

    value_label = QtWidgets.QLabel("000°")
    value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    value_label.setMinimumWidth(48)

    info_button = QtWidgets.QToolButton()
    info_button.setText("?")
    info_button.setToolTip(tooltip)
    info_button.setFixedSize(20, 20)

    def handle_value_changed(value: int) -> None:
        value_label.setText(f"{value:03d}°")
        on_change(name, value)

    slider.valueChanged.connect(handle_value_changed)

    return {
        "label": label,
        "slider": slider,
        "value_label": value_label,
        "info_button": info_button,
    }


class ProjectionViewWidget(QtWidgets.QWidget):
    def __init__(self, title: str, extent: float = 6.0) -> None:
        super().__init__()
        self._extent = extent

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title_label = QtWidgets.QLabel(title)
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        title_label.setStyleSheet("color: #dddddd;")
        layout.addWidget(title_label)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#111111")
        layout.addWidget(self.plotter, 1)

    def set_volume_and_render(self, vol: np.ndarray, iso_value: float, opacity: float) -> None:
        if not self.isVisible():
            return

        self.plotter.clear()

        if vol.size == 0 or iso_value <= 0:
            self.plotter.render()
            return

        spacing = (2 * self._extent) / max(vol.shape[0] - 1, 1)
        origin = np.array([-self._extent, -self._extent, -self._extent], dtype=float)
        vmin = float(vol.min())
        vmax = float(vol.max())

        self._add_isosurface(vol, iso_value, spacing, origin, vmax, color="#4fc3f7", opacity=opacity)
        self._add_isosurface(vol, -iso_value, spacing, origin, vmin, color="#f06292", opacity=opacity)

        self.plotter.reset_camera()
        self.plotter.render()

    def _add_isosurface(
        self,
        vol: np.ndarray,
        level: float,
        spacing: float,
        origin: np.ndarray,
        bound: float,
        color: str,
        opacity: float,
    ) -> None:
        if (level > 0 and level > bound) or (level < 0 and level < bound):
            return

        verts, faces, _, _ = measure.marching_cubes(vol, level=level, spacing=(spacing, spacing, spacing))
        if verts.size == 0 or faces.size == 0:
            return
        verts = verts + origin
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).ravel()
        mesh = pv.PolyData(verts, faces_pv)
        self.plotter.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True)

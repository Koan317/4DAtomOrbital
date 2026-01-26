import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor


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


class ProjectionViewWidget(QtWidgets.QWidget):
    def __init__(self, title: str, extent: float) -> None:
        super().__init__()
        self._extent = float(extent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title_label = QtWidgets.QLabel(title)
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        title_label.setStyleSheet("color: #dddddd;")
        layout.addWidget(title_label)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#101010")
        layout.addWidget(self.plotter.interactor, 1)

        self._actor_positive = None
        self._actor_negative = None

    def _build_grid(self, volume: np.ndarray) -> pv.UniformGrid:
        shape = volume.shape
        grid = pv.UniformGrid()
        grid.dimensions = shape
        spacing = (
            (2 * self._extent) / (shape[0] - 1),
            (2 * self._extent) / (shape[1] - 1),
            (2 * self._extent) / (shape[2] - 1),
        )
        grid.origin = (-self._extent, -self._extent, -self._extent)
        grid.spacing = spacing
        grid["values"] = volume.ravel(order="F")
        return grid

    def set_volume_and_render(self, volume: np.ndarray, iso_value: float, opacity: float) -> None:
        self.plotter.clear()

        if iso_value <= 0:
            self.plotter.render()
            return

        grid = self._build_grid(volume)
        positive = grid.contour([iso_value])
        negative = grid.contour([-iso_value])

        if positive.n_points > 0:
            self._actor_positive = self.plotter.add_mesh(
                positive,
                color="#5dade2",
                opacity=opacity,
                specular=0.3,
            )
        if negative.n_points > 0:
            self._actor_negative = self.plotter.add_mesh(
                negative,
                color="#e67e22",
                opacity=opacity,
                specular=0.3,
            )

        self.plotter.reset_camera()
        self.plotter.render()


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

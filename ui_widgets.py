from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor
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
    slider.setSingleStep(1)

    minus_button = QtWidgets.QToolButton()
    minus_button.setText("−")
    minus_button.setFixedSize(22, 22)

    plus_button = QtWidgets.QToolButton()
    plus_button.setText("+")
    plus_button.setFixedSize(22, 22)

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

    def adjust_by(delta: int) -> None:
        slider.setValue(max(slider.minimum(), min(slider.maximum(), slider.value() + delta)))

    minus_button.clicked.connect(lambda: adjust_by(-1))
    plus_button.clicked.connect(lambda: adjust_by(1))

    return {
        "label": label,
        "minus_button": minus_button,
        "slider": slider,
        "plus_button": plus_button,
        "value_label": value_label,
        "info_button": info_button,
    }


class ProjectionViewWidget(QtWidgets.QWidget):
    def __init__(self, title: str, extent: float = 6.0) -> None:
        super().__init__()
        self._extent = extent
        self._mesh_actors: dict[str, pv.Actor] = {}
        self._camera_initialized = False
        self._empty_overlay_visible = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title_label = QtWidgets.QLabel(title)
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title_label)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#111111")

        self._view_container = QtWidgets.QWidget()
        container_layout = QtWidgets.QGridLayout(self._view_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self.plotter, 0, 0)

        self._empty_overlay = QtWidgets.QLabel("无函数值，展示为空")
        self._empty_overlay.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self._empty_overlay.setStyleSheet(
            "color: #e0e0e0; background-color: rgba(0, 0, 0, 140); "
            "padding: 6px; border-radius: 6px;"
        )
        self._empty_overlay.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._empty_overlay.setVisible(False)
        container_layout.addWidget(self._empty_overlay, 0, 0, QtCore.Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._view_container, 1)

    def set_meshes(
        self,
        positive_mesh: tuple[np.ndarray, np.ndarray] | None,
        negative_mesh: tuple[np.ndarray, np.ndarray] | None,
        opacity: float,
    ) -> None:
        if not self.isVisible():
            return

        for actor in self._mesh_actors.values():
            try:
                self.plotter.remove_actor(actor)
            except (AttributeError, RuntimeError):
                continue
        self._mesh_actors.clear()
        has_mesh = False

        if positive_mesh is not None:
            actor = self._add_mesh(positive_mesh, color="#4fc3f7", opacity=opacity)
            if actor is not None:
                self._mesh_actors["positive"] = actor
                has_mesh = True

        if negative_mesh is not None:
            actor = self._add_mesh(negative_mesh, color="#f06292", opacity=opacity)
            if actor is not None:
                self._mesh_actors["negative"] = actor
                has_mesh = True

        if has_mesh and not self._camera_initialized:
            self.plotter.reset_camera()
            self._camera_initialized = True
        self.plotter.render()

    def set_empty_overlay(self, visible: bool) -> None:
        self.set_empty_message_visible(visible)

    def set_empty_message_visible(self, visible: bool) -> None:
        if self._empty_overlay_visible == visible:
            return
        self._empty_overlay_visible = visible
        self._empty_overlay.setVisible(visible)

    def set_empty_message_text(self, text: str) -> None:
        self._empty_overlay.setText(text)

    def _add_mesh(
        self,
        mesh_data: tuple[np.ndarray, np.ndarray],
        color: str,
        opacity: float,
    ) -> pv.Actor | None:
        verts, faces = mesh_data
        if verts.size == 0 or faces.size == 0:
            return None
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).ravel()
        mesh = pv.PolyData(verts, faces_pv)
        return self.plotter.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=False)

    def set_opacity(self, opacity: float) -> None:
        if not self.isVisible():
            return
        for actor in self._mesh_actors.values():
            try:
                actor.GetProperty().SetOpacity(opacity)
            except AttributeError:
                continue
        self.plotter.render()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.plotter.close()
        super().closeEvent(event)

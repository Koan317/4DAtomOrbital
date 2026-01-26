from PySide6 import QtCore, QtGui, QtWidgets


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

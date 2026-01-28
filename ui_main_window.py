import json
import math
import os
import time
import threading
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from numba import njit, prange
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from shiboken6 import isValid
    _QT_AVAILABLE = True
except ImportError:  # pragma: no cover - used for headless selftest
    class _DummySignal:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def emit(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _DummyQtCore:
        Signal = _DummySignal

        class QObject:  # noqa: D401 - minimal stub
            """Placeholder QObject for headless selftest."""

    class _DummyQtWidgets:
        def __getattr__(self, _name: str) -> type:
            return object

    class _DummyQtGui:
        def __getattr__(self, _name: str) -> type:
            return object

    QtCore = _DummyQtCore()
    QtGui = _DummyQtGui()
    QtWidgets = _DummyQtWidgets()
    _QT_AVAILABLE = False

    def isValid(*_args: object, **_kwargs: object) -> bool:
        return False

from skimage import measure

from app_state import AppState, MODE_KEY_MAP, list_mode_keys, mode_key_from_ui_label
from constants import (
    ALLOWED_RESOLUTIONS,
    DEFAULT_EXTENT_EFFECTIVE,
    DEFAULT_EXTENT_BASE,
    EXTENT_TABLE_PATH,
    ISO_ABS_EPS,
    ISO_PERCENT_FIXED,
    STRICT_ABS_EPS,
    STRICT_CANCEL_RATIO,
    STRICT_MAX_CAP,
    STRICT_PRECHECK_SAMPLES,
    VOLUME_ABS_EPS,
)
from orbitals import (
    DEMO_ORBITAL,
    get_orbital_by_id,
    get_orbital_manifest,
    list_orbital_ids,
    list_orbitals,
    run_orbital_self_check,
)
if _QT_AVAILABLE:
    from ui_widgets import ProjectionViewWidget, build_labeled_slider, build_title_label
else:  # pragma: no cover - headless selftest
    ProjectionViewWidget = None
    build_labeled_slider = None
    build_title_label = None


EXTENT_TABLE = {
    ("1s(k=0)", "slice"): 10.0,
    ("1s(k=0)", "integral_strict"): 10.0,
    ("1s(k=0)", "integral_abs"): 10.0,
    ("1s(k=0)", "max_abs"): 10.0,
    ("2p(k=0)", "slice"): 10.0,
    ("2p(k=0)", "integral_strict"): 10.0,
    ("2p(k=0)", "integral_abs"): 10.0,
    ("2p(k=0)", "max_abs"): 10.0,
    ("2p(k=1)", "slice"): 10.0,
    ("2p(k=1)", "integral_strict"): 10.0,
    ("2p(k=1)", "integral_abs"): 10.0,
    ("2p(k=1)", "max_abs"): 10.0,
    ("3d(k=0)", "slice"): 20.0,
    ("3d(k=0)", "integral_strict"): 20.0,
    ("3d(k=0)", "integral_abs"): 20.0,
    ("3d(k=0)", "max_abs"): 20.0,
    ("3d(k=1)", "slice"): 20.0,
    ("3d(k=1)", "integral_strict"): 20.0,
    ("3d(k=1)", "integral_abs"): 20.0,
    ("3d(k=1)", "max_abs"): 20.0,
    ("3d(k=2)", "slice"): 20.0,
    ("3d(k=2)", "integral_strict"): 20.0,
    ("3d(k=2)", "integral_abs"): 20.0,
    ("3d(k=2)", "max_abs"): 20.0,
    ("4f(k=0)", "slice"): 20.0,
    ("4f(k=0)", "integral_strict"): 20.0,
    ("4f(k=0)", "integral_abs"): 20.0,
    ("4f(k=0)", "max_abs"): 20.0,
    ("4f(k=1)", "slice"): 20.0,
    ("4f(k=1)", "integral_strict"): 20.0,
    ("4f(k=1)", "integral_abs"): 20.0,
    ("4f(k=1)", "max_abs"): 20.0,
    ("4f(k=2)", "slice"): 20.0,
    ("4f(k=2)", "integral_strict"): 20.0,
    ("4f(k=2)", "integral_abs"): 20.0,
    ("4f(k=2)", "max_abs"): 20.0,
    ("4f(k=3)", "slice"): 20.0,
    ("4f(k=3)", "integral_strict"): 20.0,
    ("4f(k=3)", "integral_abs"): 20.0,
    ("4f(k=3)", "max_abs"): 20.0,
    ("5g(k=0)", "slice"): 30.0,
    ("5g(k=0)", "integral_strict"): 30.0,
    ("5g(k=0)", "integral_abs"): 30.0,
    ("5g(k=0)", "max_abs"): 30.0,
    ("5g(k=1)", "slice"): 30.0,
    ("5g(k=1)", "integral_strict"): 30.0,
    ("5g(k=1)", "integral_abs"): 30.0,
    ("5g(k=1)", "max_abs"): 30.0,
    ("5g(k=2)", "slice"): 30.0,
    ("5g(k=2)", "integral_strict"): 30.0,
    ("5g(k=2)", "integral_abs"): 30.0,
    ("5g(k=2)", "max_abs"): 30.0,
    ("5g(k=3)", "slice"): 30.0,
    ("5g(k=3)", "integral_strict"): 30.0,
    ("5g(k=3)", "integral_abs"): 30.0,
    ("5g(k=3)", "max_abs"): 30.0,
    ("5g(k=4)", "slice"): 30.0,
    ("5g(k=4)", "integral_strict"): 30.0,
    ("5g(k=4)", "integral_abs"): 30.0,
    ("5g(k=4)", "max_abs"): 30.0,
    ("6h(k=0)", "slice"): 30.0,
    ("6h(k=0)", "integral_strict"): 30.0,
    ("6h(k=0)", "integral_abs"): 30.0,
    ("6h(k=0)", "max_abs"): 30.0,
    ("6h(k=1)", "slice"): 30.0,
    ("6h(k=1)", "integral_strict"): 30.0,
    ("6h(k=1)", "integral_abs"): 30.0,
    ("6h(k=1)", "max_abs"): 30.0,
    ("6h(k=2)", "slice"): 30.0,
    ("6h(k=2)", "integral_strict"): 30.0,
    ("6h(k=2)", "integral_abs"): 30.0,
    ("6h(k=2)", "max_abs"): 30.0,
    ("6h(k=3)", "slice"): 30.0,
    ("6h(k=3)", "integral_strict"): 30.0,
    ("6h(k=3)", "integral_abs"): 30.0,
    ("6h(k=3)", "max_abs"): 30.0,
    ("6h(k=4)", "slice"): 30.0,
    ("6h(k=4)", "integral_strict"): 30.0,
    ("6h(k=4)", "integral_abs"): 30.0,
    ("6h(k=4)", "max_abs"): 30.0,
    ("6h(k=5)", "slice"): 30.0,
    ("6h(k=5)", "integral_strict"): 30.0,
    ("6h(k=5)", "integral_abs"): 30.0,
    ("6h(k=5)", "max_abs"): 30.0,
    ("7i(k=0)", "slice"): 30.0,
    ("7i(k=0)", "integral_strict"): 30.0,
    ("7i(k=0)", "integral_abs"): 30.0,
    ("7i(k=0)", "max_abs"): 30.0,
    ("7i(k=1)", "slice"): 30.0,
    ("7i(k=1)", "integral_strict"): 30.0,
    ("7i(k=1)", "integral_abs"): 30.0,
    ("7i(k=1)", "max_abs"): 30.0,
    ("7i(k=2)", "slice"): 30.0,
    ("7i(k=2)", "integral_strict"): 30.0,
    ("7i(k=2)", "integral_abs"): 30.0,
    ("7i(k=2)", "max_abs"): 30.0,
    ("7i(k=3)", "slice"): 30.0,
    ("7i(k=3)", "integral_strict"): 30.0,
    ("7i(k=3)", "integral_abs"): 30.0,
    ("7i(k=3)", "max_abs"): 30.0,
    ("7i(k=4)", "slice"): 30.0,
    ("7i(k=4)", "integral_strict"): 30.0,
    ("7i(k=4)", "integral_abs"): 30.0,
    ("7i(k=4)", "max_abs"): 30.0,
    ("7i(k=5)", "slice"): 30.0,
    ("7i(k=5)", "integral_strict"): 30.0,
    ("7i(k=5)", "integral_abs"): 30.0,
    ("7i(k=5)", "max_abs"): 30.0,
    ("7i(k=6)", "slice"): 30.0,
    ("7i(k=6)", "integral_strict"): 30.0,
    ("7i(k=6)", "integral_abs"): 30.0,
    ("7i(k=6)", "max_abs"): 30.0,
    ("演示/假场 (Debug)", "slice"): 10.0,
    ("演示/假场 (Debug)", "integral_strict"): 10.0,
    ("演示/假场 (Debug)", "integral_abs"): 10.0,
    ("演示/假场 (Debug)", "max_abs"): 10.0,
}

_EXTENT_TABLE_PATH = EXTENT_TABLE_PATH


def _log_event(event: str, **fields: object) -> str:
    parts = [f"{key}={value}" for key, value in fields.items()]
    suffix = ", ".join(parts)
    return f"{event}: {suffix}" if suffix else event


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


_ORBITAL_LIST = list_orbitals(include_demo=True)
_ORBITAL_INDEX = {orb.orbital_id: idx for idx, orb in enumerate(_ORBITAL_LIST)}
_DEMO_ORBITAL_INDEX = _ORBITAL_INDEX.get(DEMO_ORBITAL.orbital_id, -1)
_ORBITAL_L = np.array([orb.parameters.get("l", 0) for orb in _ORBITAL_LIST], dtype=np.int64)
_ORBITAL_K = np.array([orb.parameters.get("k", 0) for orb in _ORBITAL_LIST], dtype=np.int64)
_ORBITAL_N = np.array([orb.parameters.get("n", 1) for orb in _ORBITAL_LIST], dtype=np.float64)
_ORBITAL_ALPHA = np.array([orb.parameters.get("alpha", 1.0) for orb in _ORBITAL_LIST], dtype=np.float64)


def _build_transverse_tables(max_k: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_terms = max_k // 2 + 1
    coeffs = np.zeros((max_k + 1, max_terms), dtype=np.float64)
    pow_x = np.zeros((max_k + 1, max_terms), dtype=np.int64)
    pow_y = np.zeros((max_k + 1, max_terms), dtype=np.int64)
    term_count = np.zeros(max_k + 1, dtype=np.int64)
    for k in range(max_k + 1):
        terms = k // 2 + 1
        term_count[k] = terms
        for j in range(terms):
            m = 2 * j
            coeffs[k, j] = float(math.comb(k, m)) * (-1.0 if j % 2 else 1.0)
            pow_x[k, j] = k - m
            pow_y[k, j] = m
    return coeffs, pow_x, pow_y, term_count


_TRANSVERSE_COEFFS, _TRANSVERSE_POW_X, _TRANSVERSE_POW_Y, _TRANSVERSE_TERM_COUNT = (
    _build_transverse_tables(6)
)

_VIEW_AXIS = {"X": 0, "Y": 1, "Z": 2, "W": 3}


@njit(parallel=True)
def _integral_volume_kernel(
    view_axis: int,
    coords: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
    rotation: np.ndarray,
    orbital_index: int,
    mode_flag: int,
) -> np.ndarray:
    n = coords.size
    total = n * n * n
    out = np.empty(total, dtype=np.float64)
    eps = 1e-12

    for idx in prange(total):
        i = idx // (n * n)
        j = (idx // n) % n
        k = idx % n

        if view_axis == 0:
            base_x = 0.0
            base_y = coords[i]
            base_z = coords[j]
            base_w = coords[k]
        elif view_axis == 1:
            base_x = coords[i]
            base_y = 0.0
            base_z = coords[j]
            base_w = coords[k]
        elif view_axis == 2:
            base_x = coords[i]
            base_y = coords[j]
            base_z = 0.0
            base_w = coords[k]
        else:
            base_x = coords[i]
            base_y = coords[j]
            base_z = coords[k]
            base_w = 0.0

        acc = 0.0
        max_val = 0.0
        for m in range(nodes.size):
            if view_axis == 0:
                x = nodes[m]
                y = base_y
                z = base_z
                w = base_w
            elif view_axis == 1:
                x = base_x
                y = nodes[m]
                z = base_z
                w = base_w
            elif view_axis == 2:
                x = base_x
                y = base_y
                z = nodes[m]
                w = base_w
            else:
                x = base_x
                y = base_y
                z = base_z
                w = nodes[m]

            xr = (
                x * rotation[0, 0]
                + y * rotation[0, 1]
                + z * rotation[0, 2]
                + w * rotation[0, 3]
            )
            yr = (
                x * rotation[1, 0]
                + y * rotation[1, 1]
                + z * rotation[1, 2]
                + w * rotation[1, 3]
            )
            zr = (
                x * rotation[2, 0]
                + y * rotation[2, 1]
                + z * rotation[2, 2]
                + w * rotation[2, 3]
            )
            wr = (
                x * rotation[3, 0]
                + y * rotation[3, 1]
                + z * rotation[3, 2]
                + w * rotation[3, 3]
            )

            if orbital_index == _DEMO_ORBITAL_INDEX:
                r = np.sqrt(xr * xr + yr * yr + zr * zr + wr * wr)
                psi = np.exp(-r) * (xr * xr - yr * yr + 0.35 * zr - 0.25 * wr)
            else:
                r = np.sqrt(xr * xr + yr * yr + zr * zr + wr * wr + eps)
                l_val = _ORBITAL_L[orbital_index]
                k_val = _ORBITAL_K[orbital_index]
                n_val = _ORBITAL_N[orbital_index]
                alpha_val = _ORBITAL_ALPHA[orbital_index]

                transverse = 0.0
                for t in range(_TRANSVERSE_TERM_COUNT[k_val]):
                    transverse += (
                        _TRANSVERSE_COEFFS[k_val, t]
                        * (xr ** _TRANSVERSE_POW_X[k_val, t])
                        * (yr ** _TRANSVERSE_POW_Y[k_val, t])
                    )

                if l_val == 0:
                    angular = transverse
                else:
                    w_power = wr ** (l_val - k_val) if l_val != k_val else 1.0
                    angular = (w_power * transverse) / (r**l_val)

                radial = np.exp(-(alpha_val / n_val) * r)
                psi = radial * angular

            if mode_flag == 0:
                acc += psi * weights[m]
            elif mode_flag == 1:
                acc += abs(psi) * weights[m]
            else:
                val = abs(psi)
                if val > max_val:
                    max_val = val

        out[idx] = acc if mode_flag != 2 else max_val

    return out.reshape((n, n, n))


def _compute_integral_volume_task(
    view_id: str,
    resolution: int,
    extent: float,
    rotation: np.ndarray,
    samples: int,
    mode_key: str,
    orbital_id: str,
) -> tuple[str, np.ndarray]:
    coords = np.linspace(-extent, extent, resolution, dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(samples)
    nodes = nodes * extent
    weights = weights * extent
    view_axis = _VIEW_AXIS[view_id]
    orbital_index = _ORBITAL_INDEX.get(orbital_id, 0)
    mode_flag = 0 if mode_key == "integral_strict" else 1 if mode_key == "integral_abs" else 2
    volume = _integral_volume_kernel(
        view_axis,
        coords,
        nodes,
        weights,
        rotation,
        orbital_index,
        mode_flag,
    )
    return view_id, volume


def _volume_cache_key(
    orbital_id: str,
    mode_key: str,
    angles: dict,
    resolution: int,
    samples: int,
    extent: float,
    view_id: str,
) -> tuple:
    angle_tuple = tuple(angles[name] for name in ["xy", "xz", "xw", "yz", "yw", "zw"])
    sample_value = samples if mode_key != "slice" else 0
    return (orbital_id, mode_key, angle_tuple, resolution, sample_value, extent, view_id)


def _mesh_cache_key(volume_key: tuple, iso_value: float, sign: int) -> tuple:
    return (volume_key, round(iso_value, 6), sign)


def _compose_rotation_matrix_from_angles(angles: dict) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    for key in _ROTATION_ORDER:
        axis_a, axis_b = _ROTATION_PLANES[key]
        mat = _rotation_matrix_for_plane(axis_a, axis_b, angles[key]) @ mat
    return mat


def _generate_slice_volume(
    view_id: str,
    resolution: int,
    extent: float,
    rotation: np.ndarray,
    orbital_id: str,
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

    orbital = get_orbital_by_id(orbital_id)
    return orbital.evaluate(xr, yr, zr, wr)


def _generate_integral_volume(
    view_id: str,
    resolution: int,
    extent: float,
    rotation: np.ndarray,
    samples: int,
    mode_key: str,
    orbital_id: str,
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
    orbital = get_orbital_by_id(orbital_id)

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

        if mode_key == "integral_strict":
            accumulator += psi * weight
        elif mode_key == "integral_abs":
            accumulator += np.abs(psi) * weight
        else:
            accumulator = np.maximum(accumulator, np.abs(psi))

    return accumulator.reshape(grid_a.shape)


def _extract_mesh(
    vol: np.ndarray,
    level: float,
    extent: float,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, str]:
    if level == 0 or vol.size == 0:
        return None, "level=0 or empty volume"
    vol_min = float(np.min(vol))
    vol_max = float(np.max(vol))
    if not (vol_min <= level <= vol_max):
        return None, "level out of range"
    spacing = (2 * extent) / max(vol.shape[0] - 1, 1)
    origin = np.array([-extent, -extent, -extent], dtype=float)
    try:
        verts, faces, _, _ = measure.marching_cubes(
            vol,
            level=level,
            spacing=(spacing, spacing, spacing),
        )
    except RuntimeError:
        return None, "marching cubes failed"
    if verts.size == 0 or faces.size == 0:
        return None, "empty topology"
    verts = verts + origin
    return (verts, faces), ""


def _precheck_strict_integral(
    view_id: str,
    resolution: int,
    extent: float,
    rotation: np.ndarray,
    orbital_id: str,
) -> dict[str, float]:
    coords = np.linspace(-extent, extent, resolution, dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(STRICT_PRECHECK_SAMPLES)
    nodes = nodes * extent
    weights = weights * extent
    view_axis = _VIEW_AXIS[view_id]
    orbital_index = _ORBITAL_INDEX.get(orbital_id, 0)
    vol_pre = _integral_volume_kernel(
        view_axis,
        coords,
        nodes,
        weights,
        rotation,
        orbital_index,
        0,
    )
    vol_abs_pre = _integral_volume_kernel(
        view_axis,
        coords,
        nodes,
        weights,
        rotation,
        orbital_index,
        1,
    )
    if vol_pre.size == 0:
        return {"strict_max": 0.0, "abs_max": 0.0}
    abs_vol = np.abs(vol_pre)
    return {
        "strict_max": float(np.max(abs_vol)),
        "abs_max": float(np.max(vol_abs_pre)) if vol_abs_pre.size else 0.0,
    }


def _boundary_max_abs(vol: np.ndarray) -> float:
    if vol.size == 0:
        return 0.0
    abs_vol = np.abs(vol)
    return float(
        max(
            abs_vol[0, :, :].max(),
            abs_vol[-1, :, :].max(),
            abs_vol[:, 0, :].max(),
            abs_vol[:, -1, :].max(),
            abs_vol[:, :, 0].max(),
            abs_vol[:, :, -1].max(),
        )
    )


def _load_extent_table() -> tuple[dict[tuple[str, str], float] | None, str]:
    if not os.path.exists(_EXTENT_TABLE_PATH):
        return None, "missing"
    try:
        with open(_EXTENT_TABLE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, "invalid"
    if payload.get("iso_percent_fixed") != ISO_PERCENT_FIXED:
        return None, "iso_mismatch"
    entries = payload.get("entries", {})
    table: dict[tuple[str, str], float] = {}
    for key, value in entries.items():
        if not isinstance(key, str) or "|" not in key:
            continue
        orbital_id, mode_key = key.split("|", 1)
        try:
            table[(orbital_id, mode_key)] = float(value)
        except (TypeError, ValueError):
            continue
    return table, "ok"


def _persist_extent_table(table: dict[tuple[str, str], float]) -> None:
    entries = {f"{orbital_id}|{mode_key}": value for (orbital_id, mode_key), value in table.items()}
    payload = {
        "version": 1,
        "iso_percent_fixed": ISO_PERCENT_FIXED,
        "entries": entries,
    }
    tmp_path = f"{_EXTENT_TABLE_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, _EXTENT_TABLE_PATH)


def calibrate_extents(
    rotations: int = 32,
    seed: int = 42,
    n_pre: int = 32,
    m_pre: int = 10,
    max_iter: int = 6,
    base_extent: float = DEFAULT_EXTENT_BASE,
) -> dict[tuple[str, str], float]:
    rng = np.random.default_rng(seed)
    orbitals = [orb for orb in list_orbitals(include_demo=True) if orb.display_name != DEMO_ORBITAL.display_name]
    mode_keys = list(MODE_KEY_MAP.values())
    angles_list = [
        {
            "xy": float(rng.uniform(0, 360)),
            "xz": float(rng.uniform(0, 360)),
            "xw": float(rng.uniform(0, 360)),
            "yz": float(rng.uniform(0, 360)),
            "yw": float(rng.uniform(0, 360)),
            "zw": float(rng.uniform(0, 360)),
        }
        for _ in range(rotations)
    ]

    table: dict[tuple[str, str], float] = {}
    coords = np.linspace(-base_extent, base_extent, n_pre, dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(m_pre)
    nodes = nodes * base_extent
    weights = weights * base_extent

    for orb in orbitals:
        for mode_key in mode_keys:
            required = base_extent
            for angles in angles_list:
                rotation = _compose_rotation_matrix_from_angles(angles)
                extent = base_extent
                for _ in range(max_iter):
                    if mode_key == "slice":
                        vols = [
                            _generate_slice_volume(view_id, n_pre, extent, rotation, orb.orbital_id)
                            for view_id in ["X", "Y", "Z", "W"]
                        ]
                    else:
                        mode_flag = 0 if mode_key == "integral_strict" else 1 if mode_key == "integral_abs" else 2
                        vols = [
                            _integral_volume_kernel(
                                _VIEW_AXIS[view_id],
                                np.linspace(-extent, extent, n_pre, dtype=np.float64),
                                nodes * (extent / base_extent),
                                weights * (extent / base_extent),
                                rotation,
                                _ORBITAL_INDEX[orb.orbital_id],
                                mode_flag,
                            )
                            for view_id in ["X", "Y", "Z", "W"]
                        ]

                    max_abs = max(float(np.max(np.abs(vol))) for vol in vols) if vols else 0.0
                    if max_abs == 0.0:
                        break
                    iso_value = (ISO_PERCENT_FIXED / 100.0) * max_abs
                    shell_max = max(_boundary_max_abs(vol) for vol in vols)
                    if shell_max >= iso_value:
                        extent *= 1.25
                        continue
                    break
                required = max(required, extent)
            final = math.ceil((1.2 * required) / 10.0) * 10.0
            table[(orb.orbital_id, mode_key)] = float(final)
    return table


def _selftest_rotations(seed: int, count: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    return [
        {
            "xy": float(rng.uniform(0, 360)),
            "xz": float(rng.uniform(0, 360)),
            "xw": float(rng.uniform(0, 360)),
            "yz": float(rng.uniform(0, 360)),
            "yw": float(rng.uniform(0, 360)),
            "zw": float(rng.uniform(0, 360)),
        }
        for _ in range(count)
    ]


def _selftest_orbitals(fast: bool) -> list[str]:
    if fast:
        return ["1s(k=0)", "2p(k=0)", "3d(k=0)", "4f(k=0)"]
    return [
        "1s(k=0)",
        "2p(k=0)",
        "2p(k=1)",
        "3d(k=0)",
        "3d(k=1)",
        "3d(k=2)",
        "4f(k=0)",
    ]


def run_selftest(fast: bool = False) -> int:
    loaded, _ = _load_extent_table()
    if loaded is not None:
        EXTENT_TABLE.clear()
        EXTENT_TABLE.update(loaded)

    mode_keys = list_mode_keys()
    view_ids = ["X", "Y", "Z", "W"]
    resolution = 24 if fast else 32
    samples = 6 if fast else 8
    rotations = _selftest_rotations(seed=42, count=2 if fast else 8)
    requested_orbitals = _selftest_orbitals(fast=fast)
    available_orbitals = set(list_orbital_ids(include_demo=True))
    errors: list[dict] = []
    total_cases = 0
    epsilon = 1e-10

    for orbital_id in requested_orbitals:
        if orbital_id not in available_orbitals:
            errors.append(
                {
                    "error": "missing orbital_id",
                    "orbital_id": orbital_id,
                }
            )
            continue
        for mode_key in mode_keys:
            extent = EXTENT_TABLE.get((orbital_id, mode_key))
            if extent is None:
                errors.append(
                    {
                        "error": "missing extent_table entry",
                        "orbital_id": orbital_id,
                        "mode_key": mode_key,
                    }
                )
                continue
            coords = np.linspace(-extent, extent, resolution, dtype=np.float64)
            nodes, weights = np.polynomial.legendre.leggauss(samples)
            nodes = nodes * extent
            weights = weights * extent

            for angles in rotations:
                rotation = _compose_rotation_matrix_from_angles(angles)
                for view_id in view_ids:
                    total_cases += 1
                    strict_empty = False
                    strict_info = {}
                    vol = np.array([])
                    if mode_key == "slice":
                        vol = _generate_slice_volume(
                            view_id, resolution, extent, rotation, orbital_id
                        )
                    elif mode_key == "integral_strict":
                        precheck = _precheck_strict_integral(
                            view_id, resolution, extent, rotation, orbital_id
                        )
                        abs_max = precheck["abs_max"]
                        strict_max = precheck["strict_max"]
                        ratio = (strict_max / abs_max) if abs_max > 0 else 0.0
                        strict_info = {
                            "strict_max": strict_max,
                            "abs_max": abs_max,
                            "ratio": ratio,
                        }
                        if abs_max <= 0.0 or (
                            ratio < STRICT_CANCEL_RATIO and strict_max < STRICT_MAX_CAP
                        ):
                            strict_empty = True
                            print(
                                _log_event(
                                    "strict_empty",
                                    orbital_id=orbital_id,
                                    mode_key=mode_key,
                                    view_id=view_id,
                                    strict_max=f"{strict_max:.3e}",
                                    abs_max=f"{abs_max:.3e}",
                                    ratio=f"{ratio:.3e}",
                                )
                            )
                        else:
                            vol = _integral_volume_kernel(
                                _VIEW_AXIS[view_id],
                                coords,
                                nodes,
                                weights,
                                rotation,
                                _ORBITAL_INDEX.get(orbital_id, 0),
                                0,
                            )
                    else:
                        mode_flag = 1 if mode_key == "integral_abs" else 2
                        vol = _integral_volume_kernel(
                            _VIEW_AXIS[view_id],
                            coords,
                            nodes,
                            weights,
                            rotation,
                            _ORBITAL_INDEX.get(orbital_id, 0),
                            mode_flag,
                        )

                    if strict_empty and orbital_id == "1s(k=0)" and mode_key == "integral_strict":
                        errors.append(
                            {
                                "error": "strict_empty_not_allowed",
                                "orbital_id": orbital_id,
                                "mode_key": mode_key,
                                "view_id": view_id,
                                "angles": angles,
                                "metrics": strict_info,
                            }
                        )
                        continue

                    if vol.size and not np.isfinite(vol).all():
                        errors.append(
                            {
                                "error": "non-finite volume",
                                "orbital_id": orbital_id,
                                "mode_key": mode_key,
                                "view_id": view_id,
                                "angles": angles,
                            }
                        )
                        continue

                    if mode_key in {"integral_abs", "max_abs"} and vol.size:
                        min_val = float(np.min(vol))
                        if min_val < -epsilon:
                            errors.append(
                                {
                                    "error": "negative value in non-negative mode",
                                    "orbital_id": orbital_id,
                                    "mode_key": mode_key,
                                    "view_id": view_id,
                                    "angles": angles,
                                    "metrics": {"min": min_val},
                                }
                            )
                            continue

                    max_abs = float(np.max(np.abs(vol))) if vol.size else 0.0
                    iso_value = (ISO_PERCENT_FIXED / 100.0) * max_abs if max_abs > 0 else 0.0
                    if max_abs > 0 and not strict_empty:
                        mesh_pos, _ = _extract_mesh(vol, iso_value, extent)
                        mesh_neg = None
                        if mode_key in {"slice", "integral_strict"}:
                            mesh_neg, _ = _extract_mesh(vol, -iso_value, extent)
                        _ = mesh_pos, mesh_neg

    print(f"selftest: cases={total_cases}, errors={len(errors)}")
    if errors:
        first = errors[0]
        metrics = first.get("metrics", {})
        print(
            _log_event(
                "selftest_failure",
                error=first.get("error"),
                orbital_id=first.get("orbital_id"),
                mode_key=first.get("mode_key"),
                view_id=first.get("view_id"),
                angles=first.get("angles"),
                metrics=metrics,
            )
        )
        return 1
    return 0


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
        mode_label = self._params["mode_label"]
        mode_key = self._params["mode_key"]
        quality_label = self._params["quality_label"]
        resolution = self._params["resolution"]
        samples = self._params["samples"]
        iso_percent = ISO_PERCENT_FIXED
        extent = self._params["extent"]
        orbital_id = self._params["orbital_id"]
        allow_mesh = self._params.get("allow_mesh", True)

        # Logging intentionally minimal; strict-empty decisions are logged separately.

        rotation = _compose_rotation_matrix_from_angles(self._params["angles"])
        rotation = np.ascontiguousarray(rotation, dtype=np.float64)
        view_ids = ["X", "Y", "Z", "W"]
        view_index = {view_id: idx for idx, view_id in enumerate(view_ids, start=1)}

        def handle_volume(view_id: str, vol: np.ndarray, volume_hit: bool) -> None:
            if self._cancel_event.is_set():
                return
            view_start = time.perf_counter()
            vol_min = float(np.min(vol)) if vol.size else 0.0
            vol_max = float(np.max(vol)) if vol.size else 0.0
            max_abs = max(abs(vol_min), abs(vol_max)) if vol.size else 0.0
            iso_value = (iso_percent / 100.0) * max_abs if iso_percent > 0 else 0.0
            is_non_negative_mode = mode_key in {"integral_abs", "max_abs"}
            strict_mode = mode_key == "integral_strict"
            strict_empty_info = strict_early_empty_info.get(view_id)
            strict_cancelled = strict_mode and strict_empty_info is not None
            strict_empty_reason = strict_empty_info.get("reason") if strict_empty_info else ""
            numerical_empty = False

            if max_abs <= VOLUME_ABS_EPS or iso_value <= ISO_ABS_EPS:
                numerical_empty = True
                strict_cancelled = False
                strict_empty_reason = "numerical-empty"
                self.log_line.emit(
                    _log_event(
                        "numerical_empty",
                        orbital_id=orbital_id,
                        mode_key=mode_key,
                        view_id=view_id,
                        max_abs=f"{max_abs:.3e}",
                        iso_value=f"{iso_value:.3e}",
                    )
                )

            if strict_mode and not strict_cancelled and vol.size:
                if max_abs <= STRICT_ABS_EPS:
                    strict_cancelled = True
                    strict_empty_reason = "strict-empty-post"
                    strict_empty_info = {
                        "strict_max": max_abs,
                        "abs_max": 0.0,
                        "ratio": 0.0,
                        "reason": strict_empty_reason,
                    }
                    self.log_line.emit(
                        _log_event(
                            "strict_empty",
                            orbital_id=orbital_id,
                            mode_key=mode_key,
                            view_id=view_id,
                            strict_max=f"{max_abs:.3e}",
                            abs_max="0.000e+00",
                            ratio="0.000e+00",
                            samples_pre=STRICT_PRECHECK_SAMPLES,
                        )
                    )

            mesh_hit_pos = False
            mesh_hit_neg = False
            mesh_pos = None
            mesh_neg = None
            mesh_pos_reason = ""
            mesh_neg_reason = ""

            volume_key = _volume_cache_key(
                orbital_id,
                mode_key,
                self._params["angles"],
                resolution,
                samples,
                extent,
                view_id,
            )

            if numerical_empty:
                mesh_pos_reason = "numerical-empty"
                mesh_neg_reason = "numerical-empty"
                mesh_pos = None
                mesh_neg = None
            elif iso_value > 0 and vol.size and not strict_cancelled:
                pos_key = _mesh_cache_key(volume_key, iso_value, 1)
                neg_key = _mesh_cache_key(volume_key, iso_value, -1)
                with self._cache_lock:
                    mesh_hit_pos, cached_pos = self._mesh_cache.get_with_hit(pos_key)
                    if not is_non_negative_mode:
                        mesh_hit_neg, cached_neg = self._mesh_cache.get_with_hit(neg_key)
                if mesh_hit_pos:
                    mesh_pos = cached_pos
                if mesh_hit_neg:
                    mesh_neg = cached_neg

                if mesh_pos is None:
                    if allow_mesh:
                        mesh_pos, mesh_pos_reason = _extract_mesh(vol, iso_value, extent)
                        with self._cache_lock:
                            self._mesh_cache.set(pos_key, mesh_pos)
                    else:
                        mesh_pos_reason = "throttled"
                if not is_non_negative_mode and mesh_neg is None:
                    if allow_mesh:
                        mesh_neg, mesh_neg_reason = _extract_mesh(vol, -iso_value, extent)
                        with self._cache_lock:
                            self._mesh_cache.set(neg_key, mesh_neg)
                    else:
                        mesh_neg_reason = "throttled"
            elif strict_cancelled:
                mesh_pos_reason = strict_empty_reason or "strict-empty"
                mesh_neg_reason = strict_empty_reason or "strict-empty"

            clipped = False
            shell_max = 0.0
            if quality_label == "最终" and not strict_cancelled:
                if max_abs > 0:
                    shell_max = _boundary_max_abs(vol)
                    if shell_max >= iso_value:
                        clipped = True

            view_time_ms = int((time.perf_counter() - view_start) * 1000)
            pos_verts = mesh_pos[0].shape[0] if isinstance(mesh_pos, tuple) else 0
            pos_faces = mesh_pos[1].shape[0] if isinstance(mesh_pos, tuple) else 0
            neg_verts = mesh_neg[0].shape[0] if isinstance(mesh_neg, tuple) else 0
            neg_faces = mesh_neg[1].shape[0] if isinstance(mesh_neg, tuple) else 0
            info = {
                "view_index": view_index[view_id],
                "volume_hit": volume_hit,
                "mesh_hit_pos": mesh_hit_pos,
                "mesh_hit_neg": mesh_hit_neg,
                "view_time_ms": view_time_ms,
                "quality_label": quality_label,
                "mode_label": mode_label,
                "mode_key": mode_key,
                "resolution": resolution,
                "samples": samples,
                "extent": extent,
                "iso_value": iso_value,
                "vol_min": vol_min,
                "vol_max": vol_max,
                "max_abs": max_abs,
                "mesh_pos_reason": mesh_pos_reason,
                "mesh_neg_reason": mesh_neg_reason,
                "strict_cancelled": strict_cancelled,
                "strict_empty_info": strict_empty_info,
                "strict_empty_reason": strict_empty_reason,
                "clipped": clipped,
                "shell_max": shell_max,
            }
            self.log_line.emit(
                _log_event(
                    "view_debug",
                    request_id=self._request_id,
                    view_id=view_id,
                    mode_key=mode_key,
                    resolution=resolution,
                    extent=round(extent, 2),
                    samples=samples,
                    max_abs=f"{max_abs:.3e}",
                    iso_value=f"{iso_value:.3e}",
                    volume_hit=volume_hit,
                    mesh_hit_pos=mesh_hit_pos,
                    mesh_hit_neg=mesh_hit_neg,
                    pos_verts=pos_verts,
                    pos_faces=pos_faces,
                    neg_verts=neg_verts,
                    neg_faces=neg_faces,
                )
            )
            self.view_ready.emit(self._request_id, view_id, mesh_pos, mesh_neg, info)

        strict_early_empty_info: dict[str, dict] = {}

        if mode_key == "slice":
            for view_id in view_ids:
                if self._cancel_event.is_set():
                    break
                volume_key = _volume_cache_key(
                    orbital_id,
                    mode_key,
                    self._params["angles"],
                    resolution,
                    samples,
                    extent,
                    view_id,
                )
                with self._cache_lock:
                    volume_hit, cached_vol = self._volume_cache.get_with_hit(volume_key)
                if volume_hit:
                    vol = cached_vol
                else:
                    vol = _generate_slice_volume(
                        view_id,
                        resolution,
                        extent,
                        rotation,
                        orbital_id,
                    )
                    if vol.size and not np.isfinite(vol).all():
                        vol = np.array([])
                    with self._cache_lock:
                        self._volume_cache.set(volume_key, vol)
                handle_volume(view_id, vol, volume_hit)
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=4) as executor:
                try:
                    for view_id in view_ids:
                        if self._cancel_event.is_set():
                            for future in futures:
                                future.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)
                            futures.clear()
                            break
                        volume_key = _volume_cache_key(
                            orbital_id,
                            mode_key,
                            self._params["angles"],
                            resolution,
                            samples,
                            extent,
                            view_id,
                        )
                        with self._cache_lock:
                            volume_hit, cached_vol = self._volume_cache.get_with_hit(volume_key)
                        if volume_hit:
                            handle_volume(view_id, cached_vol, True)
                            continue
                        if mode_key == "integral_strict":
                            precheck = _precheck_strict_integral(
                                view_id,
                                resolution,
                                extent,
                                rotation,
                                orbital_id,
                            )
                            abs_max = precheck["abs_max"]
                            strict_max = precheck["strict_max"]
                            ratio = (strict_max / abs_max) if abs_max > 0 else 0.0
                            if abs_max <= 0.0:
                                strict_early_empty_info[view_id] = {
                                    "strict_max": strict_max,
                                    "abs_max": abs_max,
                                    "ratio": ratio,
                                    "extent": extent,
                                    "reason": "strict-empty-abs-zero",
                                }
                                self.log_line.emit(
                                    _log_event(
                                        "strict_empty",
                                        orbital_id=orbital_id,
                                        mode_key=mode_key,
                                        view_id=view_id,
                                        strict_max=f"{strict_max:.3e}",
                                        abs_max=f"{abs_max:.3e}",
                                        ratio=f"{ratio:.3e}",
                                        samples_pre=STRICT_PRECHECK_SAMPLES,
                                    )
                                )
                                empty_vol = np.array([])
                                with self._cache_lock:
                                    self._volume_cache.set(volume_key, empty_vol)
                                handle_volume(view_id, empty_vol, False)
                                continue
                            if ratio < STRICT_CANCEL_RATIO and strict_max < STRICT_MAX_CAP:
                                strict_early_empty_info[view_id] = {
                                    "strict_max": strict_max,
                                    "abs_max": abs_max,
                                    "ratio": ratio,
                                    "extent": extent,
                                    "reason": "strict-empty-precheck",
                                }
                                self.log_line.emit(
                                    _log_event(
                                        "strict_empty",
                                        orbital_id=orbital_id,
                                        mode_key=mode_key,
                                        view_id=view_id,
                                        strict_max=f"{strict_max:.3e}",
                                        abs_max=f"{abs_max:.3e}",
                                        ratio=f"{ratio:.3e}",
                                        samples_pre=STRICT_PRECHECK_SAMPLES,
                                    )
                                )
                                empty_vol = np.array([])
                                with self._cache_lock:
                                    self._volume_cache.set(volume_key, empty_vol)
                                handle_volume(view_id, empty_vol, False)
                                continue
                            self.log_line.emit(
                                _log_event(
                                    "strict_not_empty",
                                    orbital_id=orbital_id,
                                    mode_key=mode_key,
                                    view_id=view_id,
                                    strict_max=f"{strict_max:.3e}",
                                    abs_max=f"{abs_max:.3e}",
                                    ratio=f"{ratio:.3e}",
                                    samples_pre=STRICT_PRECHECK_SAMPLES,
                                )
                            )
                        future = executor.submit(
                            _compute_integral_volume_task,
                            view_id,
                            resolution,
                            extent,
                            rotation,
                            samples,
                            mode_key,
                            orbital_id,
                        )
                        futures[future] = volume_key

                    for future in as_completed(futures):
                        if self._cancel_event.is_set():
                            for pending in futures:
                                pending.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                        volume_key = futures[future]
                        try:
                            view_id, vol = future.result()
                        except Exception:
                            continue
                        if vol.size and not np.isfinite(vol).all():
                            vol = np.array([])
                        with self._cache_lock:
                            self._volume_cache.set(volume_key, vol)
                        handle_volume(view_id, vol, False)
                finally:
                    if self._cancel_event.is_set():
                        for future in futures:
                            future.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)

        _ = start_total
        self.finished.emit(self._request_id)


_ROTATION_PLANES = {
    "xy": (0, 1),
    "xz": (0, 2),
    "xw": (0, 3),
    "yz": (1, 2),
    "yw": (1, 3),
    "zw": (2, 3),
}

_ROTATION_ORDER = ("zw", "yw", "yz", "xw", "xz", "xy")


def _rotation_matrix_for_plane(axis_a: int, axis_b: int, theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.eye(4, dtype=np.float64)
    mat[axis_a, axis_a] = c
    mat[axis_a, axis_b] = -s
    mat[axis_b, axis_a] = s
    mat[axis_b, axis_b] = c
    return mat


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("4D 原子轨道查看器")
        self.resize(1200, 720)

        self.state = AppState()
        self._ready = False
        self._dx_target = 0.25
        self._allowed_resolutions = list(ALLOWED_RESOLUTIONS)
        self._extent = DEFAULT_EXTENT_EFFECTIVE
        self.state.extent_effective = self._extent
        self._last_mesh_time = 0.0
        self._mesh_throttle_s = 0.3
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._trigger_scheduled_render)
        self._pending_quality_label = "预览"
        self._volume_cache = LRUCache(max_items=16)
        self._mesh_cache = LRUCache(max_items=32)
        self._cache_lock = threading.Lock()

        self._render_request_id = 0
        self._active_request_id = 0
        self._cancel_event: threading.Event | None = None
        self._render_thread: QtCore.QThread | None = None
        self._render_worker: RenderWorker | None = None
        self._retired_threads: list[QtCore.QThread] = []
        self._retired_workers: list[QtCore.QObject] = []
        self._last_params: dict | None = None
        self._pending_render_request: tuple[str, str] | None = None
        self._pending_after_finish = False
        self._shutting_down = False
        self._extent_retry_request_id: int | None = None
        self._extent_retry_attempted = False
        self._last_extent_log: tuple[str, str, float] | None = None

        self._build_status_bar()
        self._build_central()
        self._load_extent_table()

        self._run_orbital_self_check()
        self._ready = True
        self.on_ui_changed()

    def _run_orbital_self_check(self) -> None:
        warnings = run_orbital_self_check()
        if not warnings:
            return
        for warning in warnings:
            self._append_log(_log_event("orbital_self_check_warning", detail=warning))

    def _load_extent_table(self) -> None:
        loaded, status = _load_extent_table()
        if loaded is None:
            return
        EXTENT_TABLE.clear()
        EXTENT_TABLE.update(loaded)

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
        manifest = get_orbital_manifest(include_demo=True)
        for entry in manifest:
            item = QtWidgets.QListWidgetItem(entry["display_name"])
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry["id"])
            if entry["desc"]:
                item.setToolTip(entry["desc"])
            self.orbital_list.addItem(item)
        assert [self.orbital_list.item(i).text() for i in range(self.orbital_list.count())] == [
            entry["display_name"] for entry in manifest
        ], "轨道清单顺序与左侧列表不一致"
        self.orbital_list.setCurrentRow(0)
        self.orbital_list.currentTextChanged.connect(self._on_orbital_changed)
        layout.addWidget(self.orbital_list, 1)

        return panel

    def _build_center_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(panel)
        layout.setSpacing(10)

        self.projection_views = {
            "X": ProjectionViewWidget("沿x轴投影到yzw空间", extent=self._extent),
            "Y": ProjectionViewWidget("沿y轴投影到xzw空间", extent=self._extent),
            "Z": ProjectionViewWidget("沿z轴投影到xyw空间", extent=self._extent),
            "W": ProjectionViewWidget("沿w轴投影到xyz空间", extent=self._extent),
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

        title_row = QtWidgets.QHBoxLayout()
        title_label = build_title_label("控制")
        title_row.addWidget(title_label)
        help_button = QtWidgets.QToolButton()
        help_button.setText("？")
        help_button.setFixedSize(20, 20)
        help_button.setToolTip(
            "六个滑条对应 4D 的六个坐标平面旋转：XY、XZ、XW、YZ、YW、ZW。\n"
            "拖动滑条会在对应平面内旋转 0–360°。\n"
            "最终 4D 姿态为这些平面旋转的组合，任一滑条变化都会同步更新所有投影。\n"
            "XY：旋转 x-y 轴，z/w 在该次旋转中保持不变。\n"
            "XZ：旋转 x-z。\n"
            "XW：旋转 x-w。\n"
            "YZ：旋转 y-z。\n"
            "YW：旋转 y-w。\n"
            "ZW：旋转 z-w。"
        )
        title_row.addWidget(help_button)
        title_row.addStretch()
        layout.addLayout(title_row)

        slider_grid = QtWidgets.QGridLayout()
        slider_grid.setHorizontalSpacing(8)
        slider_grid.setVerticalSpacing(6)

        self.angle_controls = {}
        for row, name in enumerate(["xy", "xz", "xw", "yz", "yw", "zw"]):
            widgets = build_labeled_slider(name, self._on_angle_changed)
            self.angle_controls[name] = widgets
            widgets["slider"].sliderReleased.connect(
                lambda name=name: self._on_angle_released(name)
            )
            slider_grid.addWidget(widgets["label"], row, 0)
            slider_grid.addWidget(widgets["minus_button"], row, 1)
            slider_grid.addWidget(widgets["slider"], row, 2)
            slider_grid.addWidget(widgets["plus_button"], row, 3)
            slider_grid.addWidget(widgets["value_label"], row, 4)

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


        self.resolution_combo = QtWidgets.QComboBox()
        self.resolution_combo.addItems([str(val) for val in self._allowed_resolutions])
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

        self.reset_button = QtWidgets.QPushButton("重置角度")
        self.reset_button.clicked.connect(self._reset_angles)

        layout.addWidget(QtWidgets.QLabel("投影模式"))
        layout.addWidget(self.projection_mode)

        layout.addWidget(QtWidgets.QLabel("分辨率"))
        layout.addWidget(self.resolution_combo)

        layout.addWidget(QtWidgets.QLabel("积分采样"))
        layout.addWidget(self.samples_combo)

        layout.addWidget(QtWidgets.QLabel("预览质量"))
        layout.addWidget(self.preview_quality_combo)

        layout.addWidget(QtWidgets.QLabel("最终质量"))
        layout.addWidget(self.final_quality_combo)

        layout.addWidget(self.reset_button)

        self.explanation_group = QtWidgets.QGroupBox("说明")
        explanation_layout = QtWidgets.QVBoxLayout(self.explanation_group)
        self.explanation_text = QtWidgets.QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMinimumHeight(220)
        self.explanation_text.setPlaceholderText("轨道/投影/旋转说明将在此显示。")
        explanation_layout.addWidget(self.explanation_text)
        layout.addWidget(self.explanation_group)

        self.log_panel = QtWidgets.QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setPlaceholderText("界面事件日志...")
        log_header = QtWidgets.QHBoxLayout()
        log_header.addWidget(QtWidgets.QLabel("日志"))
        log_header.addStretch()
        self.log_clear_button = QtWidgets.QToolButton()
        self.log_clear_button.setText("清空日志")
        self.log_clear_button.clicked.connect(self._clear_log_panel)
        log_header.addWidget(self.log_clear_button)
        layout.addLayout(log_header)
        layout.addWidget(self.log_panel, 1)

        return panel

    def _update_explanation_panel(self) -> None:
        if not getattr(self, "explanation_text", None):
            return
        orbital = get_orbital_by_id(self.state.orbital_id)
        desc = orbital.short_desc_zh or orbital.display_name
        k_value = orbital.k if orbital.k is not None else orbital.parameters.get("k")
        params = f"参数：n={orbital.n}, l={orbital.l}"
        if k_value is not None:
            params += f", k={k_value}"

        angle_text = ", ".join(
            f"{key}={self.state.angles[key]}°"
            for key in ["xy", "xz", "xw", "yz", "yw", "zw"]
        )

        text = "\n".join(
            [
                "当前轨道：",
                f"  名称（严格ID）：{orbital.orbital_id}",
                f"  {params}",
                f"  一句话描述：{desc}",
                "",
                "当前投影模式：",
                "  切片（快速）：在某一轴固定切片得到3D场，再抽等值面",
                "  积分 ψ（严格）：沿被投影轴对 ψ 做有符号积分（可能发生正负抵消，部分角度可能为空）",
                "  积分（稳定）：沿被投影轴对 |ψ| 积分（更稳定，通常更易出图）",
                "  最大（可视化）：沿被投影轴取 max |ψ|（强调包络/峰值）",
                "",
                "旋转控制（6个平面旋转）：",
                "  XY / XZ / XW / YZ / YW / ZW：分别表示在对应坐标平面内旋转",
                "  组合效果：最终姿态是6个平面旋转叠加；调整任意一个会同步影响全部4个投影视图",
                f"  当前角度：{angle_text}",
            ]
        )
        self.explanation_text.setPlainText(text)

    def _on_angle_changed(self, name: str, value: int) -> None:
        self.state.angles[name] = value
        self._update_explanation_panel()
        self._handle_value_change("angles")

    def _on_angle_released(self, name: str) -> None:
        self.state.angles[name] = self.angle_controls[name]["slider"].value()
        self._handle_slider_released("angles")

    def _on_orbital_changed(self, text: str) -> None:
        item = self.orbital_list.currentItem()
        if item is not None:
            orbital_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
            self.state.orbital_id = orbital_id
            self.state.orbital_display_name = text
            orbital = get_orbital_by_id(orbital_id)
            mode_key = self.state.projection_mode_key
            self._append_log(
                _log_event(
                    "orbital_selected",
                    id=orbital.orbital_id,
                    n=orbital.n,
                    l=orbital.l,
                    mode=mode_key,
                )
            )
        self._update_explanation_panel()
        self.on_ui_changed()

    def _reset_angles(self) -> None:
        for name, widgets in self.angle_controls.items():
            widgets["slider"].blockSignals(True)
            widgets["slider"].setValue(0)
            widgets["slider"].blockSignals(False)
            self.state.angles[name] = 0
            widgets["value_label"].setText("0°")
        self.on_ui_changed()

    def shutdown(self) -> bool:
        if self._shutting_down:
            return True
        self._shutting_down = True
        if self._render_timer.isActive():
            self._render_timer.stop()
        if self._cancel_event is not None:
            self._cancel_event.set()
        if (
            self._render_thread is not None
            and isValid(self._render_thread)
            and self._render_thread.isRunning()
        ):
            self._append_log("Shutdown: waiting render thread...")
            self._render_thread.quit()
            start_time = time.monotonic()
            while self._render_thread.isRunning() and (time.monotonic() - start_time) < 10.0:
                self._render_thread.wait(100)
                QtWidgets.QApplication.processEvents()
            if self._render_thread.isRunning():
                self._append_log(
                    "Shutdown: render thread did not exit in time; keeping references to avoid QThread destroyed"
                )
                self.status_bar.showMessage("正在退出：等待后台计算结束…")
                return False
        self._cleanup_retired_threads(force=True)
        return True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if not self.shutdown():
            event.ignore()
            return
        for view in self.projection_views.values():
            view.plotter.close()
        super().closeEvent(event)

    def on_ui_changed(self, schedule_render: bool = True) -> None:
        if not self._ready:
            return

        self.state.projection_mode = self.projection_mode.currentText()
        self.state.projection_mode_key = mode_key_from_ui_label(self.state.projection_mode)
        self.state.resolution = int(self.resolution_combo.currentText())
        self.state.integral_samples = int(self.samples_combo.currentText())
        self.state.preview_quality = self.preview_quality_combo.currentText()
        self.state.final_quality = self.final_quality_combo.currentText()
        if self.orbital_list.currentItem() is not None:
            self.state.orbital_display_name = self.orbital_list.currentItem().text()
            self.state.orbital_id = self.orbital_list.currentItem().data(
                QtCore.Qt.ItemDataRole.UserRole
            )

        self._update_explanation_panel()
        self.status_bar.showMessage(f"模式={self.state.projection_mode} | 质量=空闲")

        change_kind = self._detect_change_kind()
        if not schedule_render:
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
            "orbital_id",
            "projection_mode_key",
            "angles",
            "resolution",
            "integral_samples",
            "extent",
            "preview_quality",
            "final_quality",
        ]
        if changed(volume_keys):
            return "volume"
        return "none"

    def _current_params(self) -> dict:
        return {
            "orbital_id": self.state.orbital_id,
            "projection_mode_key": self.state.projection_mode_key,
            "angles": dict(self.state.angles),
            "resolution": self.state.resolution,
            "integral_samples": self.state.integral_samples,
            "extent": self.state.extent_effective,
            "preview_quality": self.state.preview_quality,
            "final_quality": self.state.final_quality,
        }

    def _handle_value_change(self, change_kind: str) -> None:
        self.on_ui_changed(schedule_render=False)
        self._schedule_render(self._resolve_quality_label(final=False))

    def _handle_slider_released(self, change_kind: str) -> None:
        self._start_render(self._resolve_quality_label(final=True), "release")

    def _resolve_quality_label(self, final: bool) -> str:
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
        if self._shutting_down:
            return
        self._pending_quality_label = quality_label
        if self._render_timer.isActive():
            return
        self._render_timer.start(60)

    def _trigger_scheduled_render(self) -> None:
        if self._shutting_down:
            return
        self._start_render(self._pending_quality_label, "scheduled")

    def _start_render(self, quality_label: str, reason: str) -> None:
        if not self.isVisible():
            return
        if self._shutting_down:
            return

        if (
            self._render_thread is not None
            and self._render_worker is not None
            and self._cancel_event is not None
            and isValid(self._render_thread)
            and self._render_thread.isRunning()
        ):
            self._pending_render_request = (quality_label, reason)
            self._pending_after_finish = True
            self._cancel_event.set()
            try:
                self._render_thread.requestInterruption()
            except RuntimeError:
                pass
            self._append_log(
                "Render request skipped: previous render still running (cancel requested)"
            )
            return

        self._render_request_id += 1
        request_id = self._render_request_id
        self._active_request_id = request_id
        self._extent_retry_request_id = request_id
        self._extent_retry_attempted = reason == "extent-retry"
        self._cancel_event = threading.Event()
        self._clear_views("计算中...")

        orbital = get_orbital_by_id(self.state.orbital_id)
        mode_key = self.state.projection_mode_key
        extent = EXTENT_TABLE.get((orbital.orbital_id, mode_key), self.state.extent_base)
        self._extent = extent
        self.state.extent_effective = extent
        for view in self.projection_views.values():
            view.set_extent(extent)
        self._log_extent_used(orbital.orbital_id, mode_key, extent)
        allow_mesh = True
        if mode_key != "slice" and quality_label == "预览":
            now = time.perf_counter()
            if now - self._last_mesh_time < self._mesh_throttle_s:
                allow_mesh = False
            else:
                self._last_mesh_time = now

        resolution, samples = self._quality_to_settings(quality_label)
        params = {
            "orbital_id": orbital.orbital_id,
            "mode_label": self.state.projection_mode,
            "mode_key": mode_key,
            "angles": dict(self.state.angles),
            "resolution": resolution,
            "samples": samples,
            "extent": extent,
            "quality_label": quality_label,
            "allow_mesh": allow_mesh,
        }

        reason_map = {
            "manual": "手动",
            "release": "释放",
            "scheduled": "定时",
            "iso-mesh-miss": "等值面缓存未命中",
        }
        reason_text = reason_map.get(reason, reason)

        self._update_status(params, 0, "缓存=--")
        _ = reason_text

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

    def _clear_log_panel(self) -> None:
        self.log_panel.clear()

    def _log_extent_used(self, orbital_id: str, mode_key: str, extent: float) -> None:
        entry = (orbital_id, mode_key, round(extent, 2))
        if entry == self._last_extent_log:
            return
        self._last_extent_log = entry
        self._append_log(
            _log_event(
                "extent_used",
                orbital_id=orbital_id,
                mode_key=mode_key,
                extent=entry[2],
            )
        )

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
        self._append_log(
            _log_event(
                "view_status",
                view_id=view_id,
                resolution=info.get("resolution"),
                samples=info.get("samples"),
                extent=round(float(info.get("extent", self._extent)), 2)
                if info.get("extent") is not None
                else round(self._extent, 2),
                mesh_hit_pos=info.get("mesh_hit_pos"),
                mesh_hit_neg=info.get("mesh_hit_neg"),
                mesh_pos_reason=info.get("mesh_pos_reason"),
                mesh_neg_reason=info.get("mesh_neg_reason"),
                strict_cancelled=info.get("strict_cancelled"),
                strict_empty_reason=info.get("strict_empty_reason"),
            )
        )

        mesh_pos_data = mesh_pos if isinstance(mesh_pos, tuple) else None
        mesh_neg_data = mesh_neg if isinstance(mesh_neg, tuple) else None
        if info.get("strict_cancelled"):
            self.projection_views[view_id].set_meshes(None, None, 1.0)
        else:
            self.projection_views[view_id].set_meshes(mesh_pos_data, mesh_neg_data, 1.0)

        if (
            info.get("clipped")
            and self._extent_retry_request_id == request_id
            and not self._extent_retry_attempted
        ):
            self._extent_retry_attempted = True
            orbital_id = self.state.orbital_id
            mode_key = self.state.projection_mode_key
            old_extent = self._extent
            new_extent = math.ceil((old_extent * 1.25 * 1.20) / 10.0) * 10.0
            EXTENT_TABLE[(orbital_id, mode_key)] = float(new_extent)
            _persist_extent_table(EXTENT_TABLE)
            self._append_log(
                _log_event(
                    "extent_table_updated",
                    key=f"{orbital_id}|{mode_key}",
                    old=round(old_extent, 2),
                    new=round(new_extent, 2),
                )
            )
            self._start_render(self._resolve_quality_label(final=True), "extent-retry")
        elif info.get("clipped") and self._extent_retry_attempted:
            self._append_log(
                _log_event("extent_retry_failed", extent=round(self._extent, 2))
            )

        if info.get("strict_cancelled"):
            self.status_bar.showMessage(
                "模式={mode} | 质量={quality} | View {view} 严格预检为空".format(
                    mode=info["mode_label"],
                    quality=quality_label,
                    view=view_id,
                )
            )
        elif info["iso_value"] <= 0:
            self.status_bar.showMessage(
                "模式={mode} | 质量={quality} | 无网格：iso=0".format(
                    mode=info["mode_label"],
                    quality=quality_label,
                )
            )

    def _on_render_finished(self, request_id: int) -> None:
        if request_id != self._active_request_id:
            return
        self.status_bar.showMessage(f"模式={self.state.projection_mode} | 质量=空闲")

    def _on_render_thread_finished(self) -> None:
        thread = self._render_thread
        worker = self._render_worker
        self._render_thread = None
        self._render_worker = None
        if thread is not None and worker is not None:
            self._retired_threads.append(thread)
            self._retired_workers.append(worker)
        self._cleanup_retired_threads()
        if (
            self._pending_after_finish
            and self._pending_render_request is not None
            and not self._shutting_down
        ):
            quality_label, reason = self._pending_render_request
            self._pending_render_request = None
            self._pending_after_finish = False
            self._start_render(quality_label, "deferred")
        else:
            self._pending_render_request = None
            self._pending_after_finish = False

    def _cleanup_retired_threads(self, force: bool = False) -> None:
        remaining_threads: list[QtCore.QThread] = []
        remaining_workers: list[QtCore.QObject] = []
        count = min(len(self._retired_threads), len(self._retired_workers))
        for idx in range(count):
            thread = self._retired_threads[idx]
            worker = self._retired_workers[idx]
            if thread is None or not isValid(thread):
                continue
            try:
                running = thread.isRunning()
            except RuntimeError as exc:
                self._append_log(f"cleanup_retired_thread_check_failed: {exc}")
                remaining_threads.append(thread)
                if worker is not None:
                    remaining_workers.append(worker)
                continue
            if running:
                if force:
                    try:
                        thread.quit()
                    except RuntimeError as exc:
                        self._append_log(f"cleanup_retired_thread_quit_failed: {exc}")
                    try:
                        thread.wait(10000)
                    except RuntimeError as exc:
                        self._append_log(f"cleanup_retired_thread_wait_failed: {exc}")
                    try:
                        running = thread.isRunning()
                    except RuntimeError as exc:
                        self._append_log(f"cleanup_retired_thread_check_failed: {exc}")
                        remaining_threads.append(thread)
                        if worker is not None:
                            remaining_workers.append(worker)
                        continue
                    if running:
                        remaining_threads.append(thread)
                        if worker is not None:
                            remaining_workers.append(worker)
                        continue
                else:
                    remaining_threads.append(thread)
                    if worker is not None:
                        remaining_workers.append(worker)
                    continue
            try:
                thread.deleteLater()
            except RuntimeError as exc:
                self._append_log(f"cleanup_retired_thread_delete_failed: {exc}")
                remaining_threads.append(thread)
                if worker is not None:
                    remaining_workers.append(worker)
                continue
            if worker is not None and isValid(worker):
                try:
                    worker.deleteLater()
                except RuntimeError as exc:
                    self._append_log(f"cleanup_retired_worker_delete_failed: {exc}")
                    remaining_threads.append(thread)
                    remaining_workers.append(worker)
        self._retired_threads = remaining_threads
        self._retired_workers = remaining_workers

    def _update_status(self, info: dict, view_index: int, cache_text: str) -> None:
        mode = info["mode_label"] if "mode_label" in info else self.state.projection_mode
        quality_label = info["quality_label"] if "quality_label" in info else "手动"
        self.status_bar.showMessage(
            f"模式={mode} | 质量={quality_label} | 计算中... ({view_index}/4) | {cache_text}"
        )

    def _current_orbital_n(self) -> int:
        orbital = get_orbital_by_id(self.state.orbital_id)
        return orbital.n

    def _snap_resolution(self, target: int) -> int:
        return min(self._allowed_resolutions, key=lambda res: abs(res - target))

    def _clear_views(self, overlay_text: str | None = None) -> None:
        for view in self.projection_views.values():
            view.set_meshes(None, None, 1.0)
            if overlay_text:
                view.set_overlay(overlay_text)
            else:
                view.set_overlay(None)

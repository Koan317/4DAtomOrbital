from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Orbital:
    orbital_id: str
    display_name: str
    parameters: dict = field(default_factory=dict)
    _evaluator: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] = (
        lambda x, y, z, w: np.zeros_like(x)
    )

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        w: np.ndarray,
    ) -> np.ndarray:
        return self._evaluator(x, y, z, w)


def hyperspherical_coords(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian (x,y,z,w) to 4D hyperspherical (r, chi, theta, phi).

    Parameterization on S^3:
        x = r * sin(chi) * sin(theta) * cos(phi)
        y = r * sin(chi) * sin(theta) * sin(phi)
        z = r * sin(chi) * cos(theta)
        w = r * cos(chi)
    Ranges: chi ∈ [0, π], theta ∈ [0, π], phi ∈ [0, 2π).
    """
    r = np.sqrt(x**2 + y**2 + z**2 + w**2)
    rho = np.sqrt(x**2 + y**2 + z**2)

    chi = np.arctan2(rho, w)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)

    if np.any(r < eps):
        mask = r < eps
        chi = np.where(mask, 0.0, chi)
        theta = np.where(mask, 0.0, theta)
        phi = np.where(mask, 0.0, phi)

    return r, chi, theta, phi


def _radial_envelope(r: np.ndarray, n: int, alpha: float = 1.0) -> np.ndarray:
    """Simple decaying radial envelope exp(-alpha * r / n) for stable visuals."""
    scale = max(float(n), 1.0)
    return np.exp(-(alpha / scale) * r)


def _angular_1s(
    chi: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    return np.ones_like(chi)


def _angular_p_real(chi: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real p-like harmonic using x/r = sin(chi) sin(theta) cos(phi)."""
    return np.sin(chi) * np.sin(theta) * np.cos(phi)


def _angular_d_real(chi: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real d-like harmonic using (x^2 - y^2)/r^2 = sin^2(chi) sin^2(theta) cos(2phi)."""
    return (np.sin(chi) ** 2) * (np.sin(theta) ** 2) * np.cos(2.0 * phi)


def _make_orbital(
    orbital_id: str,
    display_name: str,
    n: int,
    angular_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    alpha: float = 1.0,
) -> Orbital:
    def evaluator(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        r, chi, theta, phi = hyperspherical_coords(x, y, z, w)
        radial = _radial_envelope(r, n=n, alpha=alpha)
        angular = angular_fn(chi, theta, phi)
        return radial * angular

    return Orbital(
        orbital_id=orbital_id,
        display_name=display_name,
        parameters={"n": n, "alpha": alpha},
        _evaluator=evaluator,
    )


ORBITALS: list[Orbital] = [
    _make_orbital("4d_1s", "4D 1s（球对称）", n=1, angular_fn=_angular_1s, alpha=1.0),
    _make_orbital("4d_p_real", "4D p（实 A）", n=2, angular_fn=_angular_p_real, alpha=1.1),
    _make_orbital("4d_d_real", "4D d（实 A）", n=3, angular_fn=_angular_d_real, alpha=1.15),
]


def _fake_field(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    r = np.sqrt(x**2 + y**2 + z**2 + w**2)
    return np.exp(-r) * (x**2 - y**2 + 0.35 * z - 0.25 * w)


DEMO_ORBITAL = Orbital(
    orbital_id="demo_fake",
    display_name="演示/假场 (Debug)",
    parameters={"note": "legacy demo field"},
    _evaluator=_fake_field,
)


def list_orbitals(include_demo: bool = True) -> list[Orbital]:
    if include_demo:
        return ORBITALS + [DEMO_ORBITAL]
    return list(ORBITALS)


def get_orbital_by_display_name(name: str) -> Orbital:
    for orbital in list_orbitals(include_demo=True):
        if orbital.display_name == name:
            return orbital
    return ORBITALS[0]


def run_orbital_self_check() -> list[str]:
    warnings: list[str] = []
    coords = np.linspace(-1.0, 1.0, 3, dtype=np.float64)
    grid = np.meshgrid(coords, coords, coords, coords, indexing="ij")
    x, y, z, w = (axis.reshape(-1) for axis in grid)

    for orbital in list_orbitals(include_demo=True):
        try:
            psi = orbital.evaluate(x, y, z, w)
        except Exception as exc:  # noqa: BLE001 - log and continue
            warnings.append(f"轨道 {orbital.display_name} 评估失败: {exc}")
            continue
        if not np.isfinite(psi).all():
            warnings.append(f"轨道 {orbital.display_name} 结果包含 NaN/Inf")

    base_orbital = ORBITALS[0]
    psi = base_orbital.evaluate(x, y, z, w)
    psi_neg = base_orbital.evaluate(-x, -y, -z, -w)
    max_diff = float(np.max(np.abs(psi - psi_neg)))
    if max_diff > 1e-6:
        warnings.append(
            "1s 对称性检查未通过：max|ψ(x)-ψ(-x)|={:.3e}".format(max_diff)
        )

    return warnings

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Orbital:
    orbital_id: str
    display_name: str
    n: int = 1
    description: str = ""
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


def _radial_envelope(r: np.ndarray, n: int, alpha: float = 1.0) -> np.ndarray:
    """Simple decaying radial envelope exp(-alpha * r / n) for stable visuals."""
    scale = max(float(n), 1.0)
    return np.exp(-(alpha / scale) * r)


def _transverse_terms(k: int) -> list[tuple[float, int, int]]:
    """Build terms for T_k(x,y) = Re((x + i y)^k)."""
    terms: list[tuple[float, int, int]] = []
    for j in range(k // 2 + 1):
        m = 2 * j
        coeff = float(math.comb(k, m)) * (-1.0 if j % 2 else 1.0)
        terms.append((coeff, k - m, m))
    return terms


def _make_family_orbital(l: int, k: int, alpha: float = 1.0, eps: float = 1e-12) -> Orbital:
    n = l + 1
    letter_map = ["s", "p", "d", "f", "g", "h", "i"]
    display_name = f"{n}{letter_map[l]}(k={k})"
    description = (
        "4D 分族：以 w 为极轴；k={k}；角向核 ∝ w^{l_minus_k}·Re((x+i y)^{k})/r^{l}"
    ).format(k=k, l_minus_k=l - k, l=l)
    terms = _transverse_terms(k)

    def evaluator(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2 + z**2 + w**2 + eps)
        radial = _radial_envelope(r, n=n, alpha=alpha)

        transverse = np.zeros_like(r)
        for coeff, pow_x, pow_y in terms:
            transverse += coeff * (x**pow_x) * (y**pow_y)

        if l == 0:
            angular = transverse
        else:
            w_power = w ** (l - k) if l != k else 1.0
            angular = (w_power * transverse) / (r**l)

        # ψ_{n,l,k}(x,y,z,w) = R_n(r) * A_{l,k}(x,y,z,w)
        # A_{l,k} = w^(l-k) * Re((x+i y)^k) / r^l
        return radial * angular

    return Orbital(
        orbital_id=display_name,
        display_name=display_name,
        n=n,
        description=description,
        parameters={"n": n, "l": l, "k": k, "alpha": alpha, "eps": eps},
        _evaluator=evaluator,
    )


ORBITALS: list[Orbital] = []
for l in range(7):
    for k in range(l + 1):
        # ψ_{n,l,k}(x,y,z,w) = R_n(r) * (w^(l-k) * T_k(x,y)) / r^l
        ORBITALS.append(_make_family_orbital(l, k, alpha=1.0))


def _fake_field(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    r = np.sqrt(x**2 + y**2 + z**2 + w**2)
    return np.exp(-r) * (x**2 - y**2 + 0.35 * z - 0.25 * w)


DEMO_ORBITAL = Orbital(
    orbital_id="演示/假场 (Debug)",
    display_name="演示/假场 (Debug)",
    n=1,
    description="演示用假场（Debug）",
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
    rng = np.random.default_rng(42)
    samples = rng.uniform(-1.0, 1.0, size=(32, 4)).astype(np.float64)
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]
    w = samples[:, 3]

    for orbital in list_orbitals(include_demo=True):
        if orbital.display_name == DEMO_ORBITAL.display_name:
            continue
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

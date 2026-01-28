from dataclasses import dataclass, field

from constants import (
    DEFAULT_EXTENT_BASE,
    DEFAULT_EXTENT_EFFECTIVE,
    DEFAULT_FINAL_QUALITY,
    DEFAULT_INTEGRAL_SAMPLES,
    DEFAULT_MODE_KEY,
    DEFAULT_MODE_LABEL,
    DEFAULT_ORBITAL_ID,
    DEFAULT_PREVIEW_QUALITY,
    DEFAULT_RESOLUTION,
    ISO_PERCENT_FIXED,
)


@dataclass
class AppState:
    orbital_id: str = DEFAULT_ORBITAL_ID
    orbital_display_name: str = DEFAULT_ORBITAL_ID
    auto_extent: bool = True
    extent_base: float = DEFAULT_EXTENT_BASE
    extent_effective: float = DEFAULT_EXTENT_EFFECTIVE
    angles: dict = field(
        default_factory=lambda: {
            "xy": 0,
            "xz": 0,
            "xw": 0,
            "yz": 0,
            "yw": 0,
            "zw": 0,
        }
    )
    projection_mode: str = DEFAULT_MODE_LABEL
    projection_mode_key: str = DEFAULT_MODE_KEY
    resolution: int = DEFAULT_RESOLUTION
    integral_samples: int = DEFAULT_INTEGRAL_SAMPLES
    preview_quality: str = DEFAULT_PREVIEW_QUALITY
    final_quality: str = DEFAULT_FINAL_QUALITY

    def angle_summary(self) -> str:
        return ",".join(
            f"{key}={self.angles[key]}" for key in ["xy", "xz", "xw", "yz", "yw", "zw"]
        )

    def status_text(self) -> str:
        angle_text = " ".join(
            f"{key}={self.angles[key]}°"
            for key in ["xy", "xz", "xw", "yz", "yw", "zw"]
        )
        return f"轨道：{self.orbital_display_name} | {angle_text}"

    def log_line(self) -> str:
        return (
            f"轨道={self.orbital_display_name}, 角度=[{self.angle_summary()}], "
            f"模式={self.projection_mode}({self.projection_mode_key}), "
            f"iso_fixed={ISO_PERCENT_FIXED:.1f}%, "
            f"范围={self.extent_effective:.1f}, 分辨率={self.resolution}, "
            f"采样={self.integral_samples}, "
            f"自动范围={self.auto_extent}"
        )


MODE_KEY_MAP = {
    "切片（快速）": "slice",
    "积分 ψ（严格）": "integral_strict",
    "积分 |ψ|（稳定）": "integral_abs",
    "最大 |ψ|（可视化）": "max_abs",
}

MODE_LABEL_MAP = {value: key for key, value in MODE_KEY_MAP.items()}


def mode_key_from_ui_label(label_zh: str) -> str:
    return MODE_KEY_MAP.get(label_zh, "slice")


def ui_label_from_mode_key(mode_key: str) -> str:
    return MODE_LABEL_MAP.get(mode_key, "切片（快速）")


def list_mode_keys() -> list[str]:
    return list(MODE_KEY_MAP.values())

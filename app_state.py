from dataclasses import dataclass, field


@dataclass
class AppState:
    orbital_name: str = "1s"
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
    projection_mode: str = "切片（快速）"
    iso_percent: int = 0
    resolution: int = 64
    integral_samples: int = 32
    live_update: bool = True
    auto_refine: bool = True
    preview_quality: str = "快速"
    final_quality: str = "高"

    def angle_summary(self) -> str:
        return ",".join(
            f"{key}={self.angles[key]}" for key in ["xy", "xz", "xw", "yz", "yw", "zw"]
        )

    def status_text(self) -> str:
        angle_text = " ".join(
            f"{key}={self.angles[key]:03d}°"
            for key in ["xy", "xz", "xw", "yz", "yw", "zw"]
        )
        return f"轨道：{self.orbital_name} | {angle_text}"

    def log_line(self) -> str:
        return (
            f"轨道={self.orbital_name}, 角度=[{self.angle_summary()}], "
            f"模式={self.projection_mode}, 等值={self.iso_percent}%, "
            f"分辨率={self.resolution}, "
            f"采样={self.integral_samples}, 自动精细化={self.auto_refine}"
        )

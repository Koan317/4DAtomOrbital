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
    projection_mode: str = "slice (fast)"
    isosurface_level: int = 0
    opacity: int = 0
    resolution: str = "64"
    integral_samples: str = "32"
    live_update: bool = True

    def angle_summary(self) -> str:
        return ",".join(
            f"{key}={self.angles[key]}" for key in ["xy", "xz", "xw", "yz", "yw", "zw"]
        )

    def status_text(self) -> str:
        angle_text = " ".join(
            f"{key}={self.angles[key]:03d}°"
            for key in ["xy", "xz", "xw", "yz", "yw", "zw"]
        )
        return f"Orbital: {self.orbital_name} | {angle_text}"

    def log_line(self) -> str:
        return (
            f"Orbital={self.orbital_name}, angles=[{self.angle_summary()}], "
            f"mode={self.projection_mode}, iso={self.isosurface_level}%, "
            f"opacity={self.opacity}%"
        )

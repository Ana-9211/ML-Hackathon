"""Configuration loaded from environment; private evaluator assets stay on Laptop 1."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent


def _configured_path(name: str, fallback: Path) -> Path:
    """Resolve relative data/model/log paths from the app folder, not the shell cwd."""
    raw = Path(os.getenv(name, str(fallback))).expanduser()
    return (raw if raw.is_absolute() else ROOT / raw).resolve()


@dataclass(frozen=True)
class Settings:
    mode: str = "demo"  # use included sample trips unless trained mode is explicitly selected
    eval_dir: Path = ROOT / "private_evaluator"
    demo_dir: Path = ROOT / "demo_data"
    model_path: Path = ROOT / "artifacts" / "final_model.joblib"
    team_predictor: str = ""  # optional package.module:function loaded by Laptop 1
    log_path: Path = ROOT / "logs" / "requests.jsonl"
    origins: tuple[str, ...] = ("http://localhost:5500", "http://127.0.0.1:5500")
    max_points: int = 20000
    max_arcs: int = 200000

    @property
    def data_dir(self) -> Path:
        return self.demo_dir if self.mode == "demo" else self.eval_dir

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            mode=os.getenv("M4_MODE", "demo").strip().lower(),
            eval_dir=_configured_path("M4_EVAL_DIR", ROOT / "private_evaluator"),
            demo_dir=_configured_path("M4_DEMO_DIR", ROOT / "demo_data"),
            model_path=_configured_path("M4_MODEL_PATH", ROOT / "artifacts" / "final_model.joblib"),
            team_predictor=os.getenv("M4_TEAM_PREDICTOR", "").strip(),
            log_path=_configured_path("M4_LOG_PATH", ROOT / "logs" / "requests.jsonl"),
            origins=tuple(s.strip().rstrip("/") for s in os.getenv(
                "M4_ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500"
            ).split(",") if s.strip()),
            max_points=int(os.getenv("M4_MAX_POINTS", "20000")),
            max_arcs=int(os.getenv("M4_MAX_ARCS", "200000")),
        )

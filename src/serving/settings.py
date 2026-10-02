"""Настройки сервиса. Всё берётся из тех же файлов, что и при обучении (configs/transformer.yaml,
reports/calibration.json), чтобы сервис не расходился с экспериментами. Переопределение через
переменные окружения (удобно в Docker)."""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    model_name: str
    max_length: int
    temperature: float
    calibration_seed: int
    device: str = "cpu"
    num_threads: int | None = None


def load_settings(env=None) -> Settings:
    """CONFIG_PATH, CALIBRATION_PATH, MODEL_DIR, MAX_LENGTH, DEVICE, TORCH_THREADS."""
    env = os.environ if env is None else env
    cfg = yaml.safe_load(Path(env.get("CONFIG_PATH", "configs/transformer.yaml")).read_text(encoding="utf-8"))
    cal = json.loads(Path(env.get("CALIBRATION_PATH", "reports/calibration.json")).read_text(encoding="utf-8"))

    temperature = float(cal["temperature"]["transformer"])
    if not temperature > 0:
        raise ValueError(f"Температура должна быть положительной, получено {temperature}")
    seed = int(cal["transformer_seed"])
    model_dir = Path(env.get("MODEL_DIR") or Path(cfg["output"]["models_dir"]) / f"seed_{seed}")
    threads = env.get("TORCH_THREADS") or cfg.get("num_threads")
    return Settings(
        model_dir=model_dir,
        model_name=str(cfg["model_name"]),
        max_length=int(env.get("MAX_LENGTH") or cfg["train"]["max_length"]),
        temperature=temperature,
        calibration_seed=seed,
        device=str(env.get("DEVICE", "cpu")),
        num_threads=int(threads) if threads else None,
    )

import csv
import json
import os
import random
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

FIELDS = [
    "experiment_id", "phase", "model", "dataset_size", "split_version", "seed",
    "tfidf_params", "model_params", "eval_split", "macro_f1", "mae", "accuracy",
    "train_time", "inference_time", "notes",
]


def set_seed(seed: int) -> None:
    """Фиксирует seed в random, numpy и torch (если установлен)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def log_experiment(row: dict, path: str | Path = "reports/experiments.csv") -> str:
    """Добавляет строку в experiments.csv, возвращает experiment_id."""
    unknown = set(row) - set(FIELDS)
    if unknown:
        raise ValueError(f"Неизвестные поля: {sorted(unknown)}")

    row = dict(row)
    row.setdefault(
        "experiment_id",
        f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
    )
    for key in ("tfidf_params", "model_params"):
        if isinstance(row.get(key), dict):
            row[key] = json.dumps(row[key], ensure_ascii=False, sort_keys=True)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})
    return row["experiment_id"]
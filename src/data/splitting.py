"""
Фаза 2: групповой split train/val/test (70/15/15) по group_key.

    python -m src.data.splitting            # создать split v1 (если его ещё нет)
    python -m src.data.splitting --force    # пересоздать (ОСТОРОЖНО: меняет test)

Индексы строк сохраняются в data/splits/split_<version>.parquet, параметры и
хэш test-части в data/splits/split_<version>.json. Split не пересчитывается
при каждом запуске: повторное создание без --force запрещено.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.loader import load_config

CLEAN_FILE = "reviews_clean.parquet"
GROUP_COLUMNS = {"name": "gk_name", "name_addr": "gk_name_addr"}
PARTS = ("train", "val", "test")


def group_column(cfg: dict) -> str:
    key = cfg["split"].get("group_key")
    if key not in GROUP_COLUMNS:
        raise ValueError("Задай split.group_key в configs/data.yaml: 'name' или 'name_addr'.")
    return GROUP_COLUMNS[key]


def ids_hash(ids) -> str:
    """Хэш множества row_id (не зависит от порядка)."""
    arr = np.sort(np.asarray(ids, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def make_split(
    df: pd.DataFrame,
    seed: int,
    fractions=(0.70, 0.15, 0.15),
    group_col: str = "gk_name",
    label_col: str = "rating",
    n_folds: int = 20,
) -> pd.Series:
    """Групповой split со стратификацией по рейтингу.

    Данные делятся на n_folds фолдов (StratifiedGroupKFold: группы не пересекаются,
    распределение рейтингов близко), затем фолды раздаются по частям.
    """
    n_train, n_val, n_test = (f * n_folds for f in fractions)
    if any(abs(x - round(x)) > 1e-9 for x in (n_train, n_val, n_test)) or \
            round(n_train + n_val + n_test) != n_folds:
        raise ValueError(f"fractions {fractions} не раскладываются на {n_folds} равных фолдов")
    n_val, n_test = int(round(n_val)), int(round(n_test))

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=np.int16)
    for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(df)), df[label_col], groups=df[group_col])):
        fold[idx] = k
    part = np.where(fold < n_test, "test", np.where(fold < n_test + n_val, "val", "train"))
    return pd.Series(part, index=df.index, name="split")


def save_split(df: pd.DataFrame, part: pd.Series, cfg: dict, force: bool = False) -> Path:
    scfg = cfg["split"]
    out_dir = Path(cfg["paths"]["splits_dir"])
    version = scfg["version"]
    path = out_dir / f"split_{version}.parquet"
    meta_path = out_dir / f"split_{version}.json"
    if path.exists() and not force:
        raise FileExistsError(
            f"Split {version} уже существует: {path}. Test-часть не должна меняться. "
            "Для нового split поменяй split.version в configs/data.yaml."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    gcol = group_column(cfg)
    table = pd.DataFrame({"row_id": df["row_id"].to_numpy(), "split": part.to_numpy()})
    table.to_parquet(path, index=False)

    meta = {
        "version": version,
        "seed": scfg["seed"],
        "fractions": {"train": scfg["train"], "val": scfg["val"], "test": scfg["test"]},
        "group_key": scfg["group_key"],
        "rows": {p: int((part == p).sum()) for p in PARTS},
        "groups": {p: int(df.loc[part == p, gcol].nunique()) for p in PARTS},
        "test_ids_sha256": ids_hash(df.loc[part == "test", "row_id"]),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_split_data(config_path: str = "configs/data.yaml") -> pd.DataFrame:
    """Очищенный датасет + колонка split (train/val/test) из сохранённого split."""
    cfg = load_config(config_path)
    df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / CLEAN_FILE)
    path = Path(cfg["paths"]["splits_dir"]) / f"split_{cfg['split']['version']}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Нет {path}. Сначала: python -m src.data.splitting")
    df = df.merge(pd.read_parquet(path), on="row_id", how="left", validate="one_to_one")
    if df["split"].isna().any():
        raise ValueError("Есть строки без split: пересоздай очищенный датасет и split согласованно.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="перезаписать существующий split")
    args = ap.parse_args()

    cfg = load_config()
    scfg = cfg["split"]
    gcol = group_column(cfg)
    df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / CLEAN_FILE)

    part = make_split(df, seed=scfg["seed"], fractions=(scfg["train"], scfg["val"], scfg["test"]),
                      group_col=gcol)
    path = save_split(df, part, cfg, force=args.force)

    print(f"Сохранено: {path}  (group_key = {scfg['group_key']}, колонка {gcol})")
    summary = pd.DataFrame({
        "строк": part.value_counts().reindex(PARTS),
        "доля": (part.value_counts(normalize=True).reindex(PARTS)).round(4),
        "групп": [df.loc[part == p, gcol].nunique() for p in PARTS],
    })
    print(summary)
    print((pd.crosstab(part, df["rating"], normalize="index").reindex(PARTS) * 100).round(2))


if __name__ == "__main__":
    main()

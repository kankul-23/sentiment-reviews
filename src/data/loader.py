import warnings
from pathlib import Path

import pandas as pd
import yaml

FIELDS = ("address", "name_ru", "rubrics", "rating", "text")


def load_config(path: str = "configs/data.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_tskv(path: str | Path) -> pd.DataFrame:
    """Читает tskv: строка = запись, поля key=value через TAB.

    Значения берутся как есть (без раскодирования \\n и т.п.):
    решение о чистке принимаем после аудита.
    """
    rows, incomplete = [], 0
    # newline="\n": строку заканчивает только \n, одиночный \r внутри текста не режет запись
    with open(path, encoding="utf-8", newline="\n") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            rec, last = {}, None
            for part in line.split("\t"):
                key, sep, value = part.partition("=")
                if sep and key in FIELDS:
                    rec[key] = value
                    last = key
                elif last is not None:
                    rec[last] += "\t" + part  # TAB внутри значения
                # иначе это служебный маркер "tskv" — пропускаем
            if len(rec) < len(FIELDS):
                incomplete += 1
            rows.append(rec)

    if incomplete:
        warnings.warn(f"Записей с пропущенными полями: {incomplete}")
    return pd.DataFrame(rows, columns=list(FIELDS))


def load_raw(config_path: str = "configs/data.yaml") -> pd.DataFrame:
    """Читает локальный tskv (data/raw), кэширует разбор в parquet (data/processed)."""
    cfg = load_config(config_path)
    raw = Path(cfg["paths"]["raw_dir"]) / cfg["paths"]["raw_file"]
    cache = Path(cfg["paths"]["processed_dir"]) / "geo_reviews_parsed.parquet"

    if not raw.exists():
        raise FileNotFoundError(
            f"Нет файла {raw}. Скачай датасет вручную и положи его в {raw.parent}."
        )
    if cache.exists() and cache.stat().st_mtime >= raw.stat().st_mtime:
        return pd.read_parquet(cache)

    df = parse_tskv(raw)
    rating = pd.to_numeric(df["rating"], errors="coerce")
    bad = int((rating.isna() & df["rating"].notna()).sum())
    if bad:
        warnings.warn(f"Не удалось преобразовать rating в число: {bad} строк")
    df["rating"] = rating

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df
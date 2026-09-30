"""
Фаза 2: проверки утечек между train/val/test и отчёт reports/leakage_report.md.

    python -m src.data.leakage
    python -m src.data.leakage --n-query 10000 --threshold 0.8 --num-perm 64

Обязательные проверки (при провале скрипт завершается с ошибкой):
  1. ни одна группа не встречается в двух частях;
  2. точные дубликаты текста не лежат в разных частях;
  3. test-часть не изменилась (хэш совпадает с сохранённым при создании split).
Информационно: дубликаты после нормализации пунктуации, близкие дубликаты
(MinHash), распределения рейтингов и длин по частям.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from src.data.audit import md_table
from src.data.loader import load_config
from src.data.splitting import PARTS, group_column, ids_hash, load_split_data

TOKEN_RE = re.compile(r"\w+")


# ------------------------------------------------------------ обязательные проверки
def group_overlap(df: pd.DataFrame, group_col: str) -> int:
    """Число групп, которые встречаются больше чем в одной части."""
    return int((df.groupby(group_col)["split"].nunique() > 1).sum())


def exact_duplicates_across_parts(df: pd.DataFrame, key: pd.Series) -> int:
    """Число значений key (нормализованных текстов), лежащих в разных частях."""
    dup = key.duplicated(keep=False)
    if not dup.any():
        return 0
    return int((df.loc[dup, "split"].groupby(key[dup]).nunique() > 1).sum())


def verify_test_unchanged(df: pd.DataFrame, cfg: dict) -> bool:
    meta_path = Path(cfg["paths"]["splits_dir"]) / f"split_{cfg['split']['version']}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return ids_hash(df.loc[df["split"] == "test", "row_id"]) == meta["test_ids_sha256"]


# ------------------------------------------------------------ близкие дубликаты
def _minhash(tokens: list[str], num_perm: int, k: int = 3):
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm)
    if len(tokens) < k:  # короткий текст = один шингл (совпадёт только целиком)
        shingles = [" ".join(tokens)]
    else:
        shingles = [" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]
    m.update_batch([s.encode("utf-8") for s in shingles])
    return m


def near_duplicates(df: pd.DataFrame, n_query: int, threshold: float, num_perm: int, seed: int):
    """Доля val/test-отзывов, у которых в train есть близкий дубликат (Жаккар >= threshold).

    Индекс строится по всему train, запросы идут по случайной выборке val и test.
    """
    from datasketch import MinHashLSH

    train = df[df["split"] == "train"]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    with lsh.insertion_session() as session:
        for rid, text in zip(train["row_id"], train["text"]):
            session.insert(int(rid), _minhash(TOKEN_RE.findall(text.casefold()), num_perm))

    text_by_id = df.set_index("row_id")["text"]
    rows, examples = [], []
    for part in ("val", "test"):
        q = df[df["split"] == part]
        q = q.sample(min(n_query, len(q)), random_state=seed)
        for rid, text in zip(q["row_id"], q["text"]):
            tokens = TOKEN_RE.findall(text.casefold())
            hits = lsh.query(_minhash(tokens, num_perm))
            rows.append((part, len(tokens) < 5, bool(hits)))
            if hits and len(examples) < 10 and (len(tokens) >= 5 or len(examples) < 4):
                examples.append((part, int(rid), int(hits[0]), text_by_id[int(rid)], text_by_id[int(hits[0])]))
    res = pd.DataFrame(rows, columns=["part", "short", "has_dup"])
    return res, examples


def snippet(t: str, n: int = 140) -> str:
    t = t.replace("|", "/").replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-query", type=int, default=10000, help="запросов на val и на test")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--num-perm", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["split"]["seed"]
    gcol = group_column(cfg)
    df = load_split_data()
    n = len(df)

    out = ["# Отчёт об утечках: split " + cfg["split"]["version"] + "\n"]
    add = out.append

    # 1. Сводка
    add("## 1. Сводка split\n")
    add(f"group_key: `{cfg['split']['group_key']}` (колонка `{gcol}`), seed {seed}.\n")
    summ = pd.DataFrame({
        "строк": df["split"].value_counts().reindex(PARTS),
        "доля": df["split"].value_counts(normalize=True).reindex(PARTS).round(4),
        "групп": [df.loc[df["split"] == p, gcol].nunique() for p in PARTS],
    })
    add(md_table(summ) + "\n")

    # 2. Обязательные проверки
    add("## 2. Обязательные проверки\n")
    text_key = df["text"].str.casefold()
    g_over = group_overlap(df, gcol)
    ex_dup = exact_duplicates_across_parts(df, text_key)
    unchanged = verify_test_unchanged(df, cfg)
    checks = [
        ("Групп в нескольких частях", g_over, g_over == 0),
        ("Точных дубликатов текста в разных частях", ex_dup, ex_dup == 0),
        ("Test-часть не изменилась (хэш совпадает)", "да" if unchanged else "НЕТ", unchanged),
    ]
    add(md_table(pd.DataFrame([(a, b, "PASS" if ok else "FAIL") for a, b, ok in checks],
                              columns=["проверка", "значение", "результат"]), index=False) + "\n")
    all_ok = all(ok for *_, ok in checks)

    # 3. Дубликаты после нормализации пунктуации
    add("## 3. Дубликаты после удаления пунктуации (информационно)\n")
    alnum = text_key.str.replace(r"[\W_]+", " ", regex=True).str.strip()
    alnum_dup = exact_duplicates_across_parts(df, alnum)
    rows_in_dup = int(alnum.duplicated(keep=False).sum())
    add(f"Строк, чей текст без пунктуации совпадает с другим отзывом: {rows_in_dup:,} "
        f"({rows_in_dup / n:.2%}); из них групп-значений, лежащих в разных частях: {alnum_dup:,}.\n")

    # 4. Близкие дубликаты
    add("## 4. Близкие дубликаты (MinHash)\n")
    t0 = time.time()
    res, examples = near_duplicates(df, args.n_query, args.threshold, args.num_perm, seed)
    add(f"Шингл = 3 слова, порог Жаккара {args.threshold}, {args.num_perm} перестановок. "
        f"Индекс по всему train, запросы: до {args.n_query:,} случайных отзывов из val и из test "
        f"(время {time.time() - t0:.0f} с).\n")
    tbl = res.groupby("part")["has_dup"].agg(["size", "sum", "mean"]).rename(
        columns={"size": "запросов", "sum": "с близким дублем в train", "mean": "доля"})
    add(md_table(tbl.round(4)) + "\n")
    by_len = res.assign(длина=res["short"].map({True: "короткие (< 5 слов)", False: "длинные (>= 5 слов)"})) \
        .groupby("длина")["has_dup"].agg(["size", "sum", "mean"]).rename(
        columns={"size": "запросов", "sum": "с близким дублем", "mean": "доля"})
    add(md_table(by_len.round(4)) + "\n")
    add("Примеры (часть / отзыв / найденный в train дубль):\n")
    for part, rid, hid, t1, t2 in examples:
        add(f"- {part}: «{snippet(t1)}» ~ «{snippet(t2)}»")
    add("")

    # 5. Распределения
    add("## 5. Распределения по частям\n")
    rating = (pd.crosstab(df["split"], df["rating"], normalize="index").reindex(PARTS) * 100).round(2)
    rating.columns = [f"{c}★, %" for c in rating.columns]
    add(md_table(rating) + "\n")
    add(f"Максимальный разброс доли класса между частями: {(rating.max() - rating.min()).max():.2f} п.п.\n")
    words = df["text"].str.split().str.len()
    length = words.groupby(df["split"]).describe(percentiles=[.5, .9, .99]).reindex(PARTS)
    length = length[["mean", "50%", "90%", "99%", "max"]].round(1)
    length.columns = ["слов, среднее", "медиана", "p90", "p99", "максимум"]
    add(md_table(length) + "\n")

    # 6. Решение
    add("## 6. Решение по близким дубликатам\n")
    add("_Заполнить после просмотра раздела 4: нужна ли дополнительная чистка и какая._\n")

    path = Path(cfg["paths"]["reports_dir"]) / "leakage_report.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"Отчёт: {path}")
    print("Обязательные проверки:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

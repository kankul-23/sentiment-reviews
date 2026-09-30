"""
Фаза 1: аудит датасета, чистка и подготовка group_key.

Запуск из корня проекта:
    python -m src.data.audit
    python -m src.data.audit --group-key name_addr --min-words 2 --tokenize-sample 0

Результат:
    reports/data_audit.md                  отчёт с цифрами
    data/processed/reviews_clean.parquet   очищенный датасет с колонкой group_key
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_config, load_raw

EMOJI_PATTERN = "[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]"
URL_PATTERN = r"https?://|www\."

# Прямые упоминания оценки в тексте
WORD2NUM = {"одну": 1, "одна": 1, "две": 2, "три": 3, "четыре": 4, "пять": 5}
PAT_DIGIT = r"(?<!\d)([1-5])\s*(?:-\s*)?(?:зв[её]зд|балл|из\s*5|/\s*5|из\s*пяти)"
PAT_VERB = r"(?:оценк\w*|ставлю|поставил\w*|поставлю|дам|даю)\s*[:\-–]?\s*([1-5])(?!\d)"
PAT_WORD = r"\b(одну|одна|две|три|четыре|пять)\s+зв[её]зд"


# --------------------------------------------------------------------------- utils
def normalize_text(s: pd.Series) -> pd.Series:
    """Литеральные экранирования tskv (\\n, \\r, \\t) -> пробел, схлопнуть пробелы."""
    s = s.fillna("")
    s = s.str.replace(r"\\[nrt]", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def norm_key(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.casefold().str.replace("ё", "е", regex=False)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        cells = [f"{x:,.4g}" if isinstance(x, float) else str(x) for x in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def share(n: int, total: int) -> str:
    return f"{n:,} ({n / total:.2%})"


def describe(s: pd.Series) -> dict:
    qs = {"min": 0, "p50": 50, "p90": 90, "p95": 95, "p99": 99, "p99.9": 99.9, "max": 100}
    return {k: float(np.percentile(s, q)) for k, q in qs.items()}


def snippet(text: str, n: int = 160) -> str:
    t = text.replace("|", "/").replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")


def token_lengths(texts: pd.Series, n: int, seed: int):
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        tok.model_max_length = 10**6
        sample = texts.sample(min(n, len(texts)), random_state=seed).tolist()
        ids = tok(sample, add_special_tokens=True, truncation=False)["input_ids"]
        return pd.Series([len(x) for x in ids])
    except Exception as e:  # нет сети / нет токенайзера
        print(f"[warn] токенайзер недоступен: {e}")
        return None


def mentioned_rating(text: pd.Series) -> pd.Series:
    low = text.str.lower()
    a = pd.to_numeric(low.str.extract(PAT_DIGIT, expand=False), errors="coerce")
    b = pd.to_numeric(low.str.extract(PAT_VERB, expand=False), errors="coerce")
    c = low.str.extract(PAT_WORD, expand=False).map(WORD2NUM)
    return a.fillna(b).fillna(c)


def group_stats(df: pd.DataFrame, key: str) -> dict:
    size = df.groupby(key).size().sort_values(ascending=False)
    n = len(df)
    return {
        "key": key,
        "groups": len(size),
        "reviews/group median": float(size.median()),
        "reviews/group p90": float(size.quantile(0.9)),
        "reviews/group p99": float(size.quantile(0.99)),
        "reviews/group max": int(size.iloc[0]),
        "largest group share": f"{size.iloc[0] / n:.2%}",
        "top-10 groups share": f"{size.iloc[:10].sum() / n:.2%}",
    }


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-key", choices=["name", "name_addr"], default="name")
    ap.add_argument("--min-words", type=int, default=1, help="минимум слов в отзыве")
    ap.add_argument("--tokenize-sample", type=int, default=20000, help="0 = отключить")
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["split"]["seed"]
    proc_dir = Path(cfg["paths"]["processed_dir"])
    rep_dir = Path(cfg["paths"]["reports_dir"])

    df = load_raw()
    df["row_id"] = df.index
    n0 = len(df)

    out: list[str] = []
    add = out.append
    add("# Аудит данных: Geo Reviews Dataset 2023\n")
    add("Файл сгенерирован скриптом `python -m src.data.audit`. "
        "Лицензия датасета: MIT (Copyright 2023 YANDEX LLC).\n")

    # 1. Схема ----------------------------------------------------------------
    add("## 1. Схема и пропуски\n")
    add(f"Строк: **{n0:,}**. Колонки: {', '.join(f'`{c}`' for c in df.columns if c != 'row_id')}.\n")
    miss = pd.DataFrame(
        {
            "dtype": df.drop(columns="row_id").dtypes.astype(str),
            "NaN": df.drop(columns="row_id").isna().sum(),
            "empty": [
                int((df[c].notna() & df[c].astype(str).str.strip().eq("")).sum())
                for c in df.columns if c != "row_id"
            ],
        }
    )
    add(md_table(miss) + "\n")
    n_lit = int(df["text"].str.contains("\\n", regex=False).sum())
    n_real = int(df["text"].str.contains("\n", regex=False).sum())
    add(f"- Отзывов с литеральным `\\n` (экранирование tskv): {share(n_lit, n0)}\n"
        f"- Отзывов с реальным переводом строки: {share(n_real, n0)}\n"
        "- Колонки с датой в данных нет: диапазон дат проверить нельзя "
        "(по описанию датасета отзывы за январь-июль 2023).\n")

    df["text"] = normalize_text(df["text"])  # нормализация до всех остальных проверок

    # 2. Рейтинг --------------------------------------------------------------
    add("## 2. Рейтинговая шкала\n")
    rc = df["rating"].value_counts(dropna=False).sort_index()
    add(md_table(pd.DataFrame({"count": rc, "share": (rc / n0).round(4)})) + "\n")
    valid_mask = df["rating"].between(1, 5)
    p5 = float((df.loc[valid_mask, "rating"] == 5).mean())
    add(f"Среди оценок 1-5 доля пятёрок **{p5:.2%}**. Константный предсказатель «всегда 5» "
        f"даёт accuracy ≈ {p5:.3f} и macro-F1 ≈ {2 * p5 / (1 + p5) / 5:.3f} "
        "(ориентир «дна» для метрик).\n")
    zero = df[df["rating"] == 0]
    if len(zero):
        add(f"Оценка 0 встречается в {share(len(zero), n0)} строк, вне шкалы 1-5. Примеры:\n")
        for t in zero["text"].head(5):
            add(f"- {snippet(t)}")
        add("")

    # 3. Тексты ---------------------------------------------------------------
    add("## 3. Длина и шум в текстах\n")
    chars = df["text"].str.len()
    words = df["text"].str.split().str.len()
    rows = {"символы": describe(chars), "слова": describe(words)}
    tl = None
    if args.tokenize_sample > 0:
        tl = token_lengths(df.loc[df["text"] != "", "text"], args.tokenize_sample, seed)
        if tl is not None:
            rows[f"токены rubert-tiny2 (выборка {len(tl):,})"] = describe(tl)
    add(md_table(pd.DataFrame(rows).T) + "\n")
    if tl is not None:
        add("Доля отзывов длиннее порога в токенах: "
            + ", ".join(f"> {m}: {(tl > m).mean():.2%}" for m in (64, 128, 256, 512)) + "\n")
    add("Короткие отзывы (в словах): "
        + ", ".join(f"< {k}: {share(int((words < k).sum()), n0)}" for k in (2, 3, 5, 10)) + "\n")
    add(f"Шум: эмодзи {share(int(df['text'].str.contains(EMOJI_PATTERN).sum()), n0)}; "
        f"ссылки {share(int(df['text'].str.contains(URL_PATTERN).sum()), n0)}; "
        f"без кириллицы {share(int((~df['text'].str.contains('[а-яА-ЯёЁ]')).sum()), n0)}.\n")

    # 4. Дубликаты ------------------------------------------------------------
    add("## 4. Дубликаты\n")
    nonempty = df[df["text"] != ""]
    tkey = nonempty["text"].str.casefold()
    dup_rows = int(tkey.duplicated().sum())
    in_dup = tkey.duplicated(keep=False)
    only_dup, k2 = nonempty[in_dup], tkey[in_dup]
    conflict_keys = cross_org = 0
    if len(only_dup):
        g = only_dup.groupby(k2).agg(n_ratings=("rating", "nunique"), n_orgs=("name_ru", "nunique"))
        conflict_keys = int((g["n_ratings"] > 1).sum())
        cross_org = int((g["n_orgs"] > 1).sum())
    full_dups = int(df.duplicated(subset=["name_ru", "address", "rating", "text"]).sum())
    add(f"- Повторов текста (без учёта регистра), лишних строк: {share(dup_rows, n0)}\n"
        f"- Текстов с конфликтующими оценками: {conflict_keys:,}\n"
        f"- Текстов, повторяющихся у разных организаций: {cross_org:,}\n"
        f"- Полных дублей строки (организация, адрес, оценка, текст): {share(full_dups, n0)}\n")

    # 5. Чистка ---------------------------------------------------------------
    add("## 5. Решения по чистке\n")
    steps = []

    m = ~df["rating"].between(1, 5)
    steps.append(("оценка вне шкалы 1-5", int(m.sum())))
    df = df[~m]
    m = df["name_ru"].isna() | df["name_ru"].astype(str).str.strip().eq("")
    steps.append(("нет названия организации", int(m.sum())))
    df = df[~m]
    m = df["text"].str.split().str.len().fillna(0) < args.min_words
    steps.append((f"текст короче {args.min_words} сл.", int(m.sum())))
    df = df[~m]
    key = df["text"].str.casefold()
    in_dup = key.duplicated(keep=False)
    m = pd.Series(False, index=df.index)
    if in_dup.any():
        nun = df[in_dup].groupby(key[in_dup])["rating"].nunique()
        m = in_dup & key.isin(nun[nun > 1].index)
    steps.append(("тексты с конфликтующими оценками (удалены все копии)", int(m.sum())))
    df = df[~m]
    m = df["text"].str.casefold().duplicated(keep="first")
    steps.append(("точные дубликаты текста (оставлена первая копия)", int(m.sum())))
    df = df[~m]

    add(md_table(pd.DataFrame(steps, columns=["шаг", "удалено строк"]), index=False) + "\n")
    add(f"Осталось **{len(df):,}** из {n0:,} строк ({len(df) / n0:.2%}).\n")
    rc2 = df["rating"].value_counts().sort_index()
    add(md_table(pd.DataFrame({"count": rc2, "share": (rc2 / len(df)).round(4)})) + "\n")
    add("Выбросы по длине не удаляются: длинные тексты обрезаются параметром max_length.\n")

    df["rating"] = df["rating"].astype("int8")

    # 6. Организации и group_key ------------------------------------------------
    add("## 6. Организации и кандидаты group_key (на очищенных данных)\n")
    df["name_key"] = norm_key(df["name_ru"])
    df["addr_key"] = norm_key(df["address"])
    df["gk_name"] = df["name_key"]
    df["gk_name_addr"] = df["name_key"] + " | " + df["addr_key"]
    n = len(df)
    add(f"Уникальных названий: {df['gk_name'].nunique():,}; адресов: {df['addr_key'].nunique():,}; "
        f"пар название+адрес: {df['gk_name_addr'].nunique():,}; "
        f"строк рубрик: {df['rubrics'].nunique():,}.\n")
    add(md_table(pd.DataFrame([group_stats(df, "gk_name"), group_stats(df, "gk_name_addr")]),
                 index=False) + "\n")

    addr_per_name = df.groupby("gk_name")["addr_key"].nunique()
    chain_names = addr_per_name[addr_per_name > 1].index
    chain_reviews = int(df["gk_name"].isin(chain_names).sum())
    add(f"Названий с более чем одним адресом (сети): {len(chain_names):,} "
        f"из {len(addr_per_name):,}. Отзывов о таких организациях: {share(chain_reviews, n)}. "
        "При группировке по название+адрес отзывы одной сети попадут в разные части split.\n")

    top = df.groupby("gk_name").agg(reviews=("rating", "size"), addresses=("addr_key", "nunique"))
    add("Топ-10 названий по числу отзывов:\n")
    add(md_table(top.sort_values("reviews", ascending=False).head(10)) + "\n")
    first_rubric = df["rubrics"].str.split(";").str[0]
    add("Топ-10 первых рубрик:\n")
    add(md_table(first_rubric.value_counts().head(10).rename("count").to_frame()) + "\n")

    # 7. Прямая утечка цели -----------------------------------------------------
    add("## 7. Прямые упоминания оценки в тексте\n")
    mention = mentioned_rating(df["text"])
    has = mention.notna()
    agree = (mention[has] == df.loc[has, "rating"]).mean() if has.any() else float("nan")
    add(f"Отзывов с явным упоминанием оценки: {share(int(has.sum()), n)}. "
        f"Из них упомянутое число совпадает с реальной оценкой в {agree:.1%} случаев "
        "(остальное - ложные срабатывания регулярок или несогласованность оценки и текста).\n")
    add("Примеры (оценка / упомянуто / текст):\n")
    for idx in mention[has].sample(min(8, int(has.sum())), random_state=seed).index if has.any() else []:
        add(f"- {df.at[idx, 'rating']} / {int(mention[idx])} / {snippet(df.at[idx, 'text'])}")
    add("")

    # 8. Итог ---------------------------------------------------------------------
    add("## 8. Выбранный group_key\n")
    chosen = {"name": "gk_name", "name_addr": "gk_name_addr"}[args.group_key]
    df["group_key"] = df[chosen]
    add(f"В `reviews_clean.parquet` колонка `group_key` = `{chosen}` (параметр `--group-key`). "
        "Обоснование (по EDA): название закрывает утечку через сети (более половины отзывов "
        "относятся к названиям с несколькими адресами), а группы остаются мелкими. "
        "Колонки обоих кандидатов сохранены для сравнения.\n")

    keep = ["row_id", "text", "rating", "name_ru", "address", "rubrics",
            "gk_name", "gk_name_addr", "group_key"]
    proc_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    df[keep].reset_index(drop=True).to_parquet(proc_dir / "reviews_clean.parquet", index=False)
    (rep_dir / "data_audit.md").write_text("\n".join(out), encoding="utf-8")
    print(f"Готово: {rep_dir / 'data_audit.md'}, {proc_dir / 'reviews_clean.parquet'}")


if __name__ == "__main__":
    main()

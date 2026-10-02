"""
Фаза 5: разбор ошибок бейзлайна (TF-IDF + LogReg) и трансформера (rubert-tiny2).

    python -m src.evaluation.error_analysis                 # разбор на val (по умолчанию)
    python -m src.evaluation.error_analysis --split test --allow-test   # только подтверждение выводов

Вход: предсказания из data/predictions (baseline_{split}.parquet, seed{S}_{split}.parquet)
и очищенный датасет data/processed/reviews_clean.parquet (text, rubrics, group_key).
Разбор по умолчанию идёт на val: выводы, которые повлияют на дальнейшие решения (выбор модели
для сервиса, что написать в README), не должны опираться на test. Test допускается только
для подтверждения уже сделанных выводов (флаг --allow-test).

Выход: reports/error_analysis_{split}_tables.md (все таблицы и примеры),
reports/tables/error_*_{split}.csv, reports/figures/error_*_{split}.png,
data/predictions/error_table_{split}.parquet (единая таблица для ноутбуков, вне git).
Выводы пишутся вручную в reports/error_analysis.md.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audit import md_table
from src.data.loader import load_config
from src.evaluation.metrics import LABELS, compute_metrics, confusion_df

PROB_COLS = [f"p{c}" for c in LABELS]

LEN_EDGES = [0, 50, 100, 200, 400, 800, np.inf]
LEN_LABELS = ["<50", "50-99", "100-199", "200-399", "400-799", "800+"]
CONF_EDGES = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]
CONF_LABELS = ["<0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95+"]

NO_MENTION, MENTION_MATCH, MENTION_DIFF = "нет упоминания", "совпадает с оценкой", "не совпадает с оценкой"
AGREE = ["обе верны", "обе ошиблись", "верен только бейзлайн", "верен только трансформер"]

_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2764\uFE0F]")

# Прямое упоминание оценки в тексте. Эвристика с упором на точность, а не на полноту.
_WORD = r"(един\w*|один|двойк\w*|два|тройк\w*|три|четв[её]рк\w*|четыре|пят[её]рк\w*|пять)"
_PATTERNS = [
    (re.compile(r"\b([1-5])\s*(?:/|из)\s*5\b"), "digit"),
    (re.compile(r"\b([1-5])\s*(?:звезд|звёзд|балл|\*)"), "digit"),
    (re.compile(r"(?:оценк\w*|ставлю|поставил\w*|оцениваю|оценил\w*)\s*[:\-–]?\s*([1-5])\b"), "digit"),
    (re.compile(r"(?:\bна|ставлю|поставил\w*|оценк\w*|оценил\w*)\s*[:\-–]?\s*(?:бы\s+|только\s+|твёрдую\s+|твердую\s+)?" + _WORD), "word"),
]
_WORD_PREFIXES = [("един", 1), ("один", 1), ("двойк", 2), ("два", 2), ("тройк", 3), ("три", 3),
                  ("четв", 4), ("четыр", 4), ("пят", 5)]


# ------------------------------------------------------------------------ признаки
def mention_rating(text: str) -> float:
    """Оценка, прямо названная в тексте («5 звёзд», «4 из 5», «ставлю тройку»), иначе NaN.
    При нескольких совпадениях берётся самое раннее."""
    t = str(text).lower()
    best = None
    for pat, kind in _PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        if kind == "digit":
            val = int(m.group(1))
        else:
            val = next(v for pref, v in _WORD_PREFIXES if m.group(1).startswith(pref))
        if best is None or m.start() < best[0]:
            best = (m.start(), val)
    return float(best[1]) if best else np.nan


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["length_chars"] = df["text"].str.len()
    df["length_words"] = df["text"].str.split().str.len()
    df["len_bin"] = pd.cut(df["length_chars"], LEN_EDGES, labels=LEN_LABELS, right=False)
    df["has_emoji"] = np.where(df["text"].map(lambda t: bool(_EMOJI.search(t))), "да", "нет")
    df["mention"] = df["text"].map(mention_rating)
    df["mention_group"] = np.select(
        [df["mention"].isna(), df["mention"] == df["y_true"]],
        [NO_MENTION, MENTION_MATCH], default=MENTION_DIFF)
    df["rubric_main"] = (df["rubrics"].fillna("").astype(str).str.split(";").str[0].str.strip()
                         .replace("", "без рубрики"))
    df["agreement"] = agreement_labels(df["y_true"].to_numpy(), df["pred_base"].to_numpy(),
                                       df["pred_tr"].to_numpy())
    return df


def agreement_labels(y, p_base, p_tr) -> np.ndarray:
    rb, rt = y == p_base, y == p_tr
    return np.select([rb & rt, ~rb & ~rt, rb & ~rt, ~rb & rt], AGREE, default="")


# --------------------------------------------------------------------- данные
def _load_pred(pred_dir: Path, stem: str, split: str) -> pd.DataFrame:
    path = pred_dir / f"{stem}_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Нет {path}. Бейзлайн: python -m src.models.baseline --save-predictions; "
                                "трансформер: --final.")
    return pd.read_parquet(path).sort_values("row_id").reset_index(drop=True)


def build_table(split: str, seed: int, seeds: list, pred_dir: Path, clean_path: Path) -> pd.DataFrame:
    """Единая таблица: истинная оценка, предсказания и уверенность трёх «моделей»
    (бейзлайн, трансформер seed=<seed>, ансамбль трансформера по всем сидам) + поля отзыва."""
    base = _load_pred(pred_dir, "baseline", split)
    trs = {s: _load_pred(pred_dir, f"seed{s}", split) for s in seeds}
    tr = trs[seed]
    for name, other in [("seed" + str(s), d) for s, d in trs.items()]:
        if not (other["row_id"].equals(base["row_id"]) and other["y_true"].equals(base["y_true"])):
            raise ValueError(f"Предсказания {name} и бейзлайна не совпадают по row_id/y_true")

    ens = np.mean([d[PROB_COLS].to_numpy() for d in trs.values()], axis=0)
    df = pd.DataFrame({
        "row_id": base["row_id"].to_numpy(),
        "y_true": base["y_true"].to_numpy().astype(np.int64),
        "pred_base": base["y_pred"].to_numpy().astype(np.int64),
        "pred_tr": tr["y_pred"].to_numpy().astype(np.int64),
        "pred_ens": ens.argmax(axis=1) + 1,
        "conf_base": base[PROB_COLS].to_numpy().max(axis=1),
        "conf_tr": tr[PROB_COLS].to_numpy().max(axis=1),
        "conf_ens": ens.max(axis=1),
    })
    clean = pd.read_parquet(clean_path, columns=["row_id", "text", "rubrics", "group_key"])
    df = df.merge(clean, on="row_id", how="left", validate="one_to_one")
    if df["text"].isna().any():
        raise ValueError("Часть row_id из предсказаний не найдена в reviews_clean.parquet")
    return add_features(df)


# ---------------------------------------------------------------------- метрики
def _f1_from_counts(counts25, present_only: bool = False) -> float:
    cm = np.asarray(counts25, dtype=float).reshape(5, 5)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros(5), where=denom > 0)
    if present_only:
        return float(f1[denom > 0].mean()) if (denom > 0).any() else np.nan
    return float(f1.mean())


def _counts(y, p) -> np.ndarray:
    return np.bincount((y - 1) * 5 + (p - 1), minlength=25)


def cluster_bootstrap(y, preds: dict, groups, n_boot: int = 1000, seed: int = 42,
                      pairs: tuple = ()) -> tuple[dict, dict]:
    """Доверительные интервалы macro-F1 (по всем 5 классам, как в проекте) и разностей между
    моделями. Бутстрэп по группам group_key (отзывы об одном месте зависимы), одни и те же
    выборки групп для всех моделей. Возвращает ({имя: (оценка, low, high)}, {(a, b): (...)} для a - b)."""
    codes, _ = pd.factorize(pd.Series(groups))
    n_groups = int(codes.max()) + 1
    per_group = {}
    for name, p in preds.items():
        c = np.zeros((n_groups, 25))
        np.add.at(c, (codes, (y - 1) * 5 + (p - 1)), 1)
        per_group[name] = c
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_boot) for name in preds}
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        for name, c in per_group.items():
            draws[name][i] = _f1_from_counts(w @ c)

    def ci(d, point):
        return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    single = {n: ci(draws[n], _f1_from_counts(per_group[n].sum(0))) for n in preds}
    diffs = {(a, b): ci(draws[a] - draws[b], single[a][0] - single[b][0]) for a, b in pairs}
    return single, diffs


def slice_table(df: pd.DataFrame, col: str, models: dict, min_n: int = 100) -> pd.DataFrame:
    """Метрики по срезам. macro-F1 считается по классам, присутствующим в срезе, и только
    при n >= min_n (в малых срезах оценка шумит)."""
    rows = []
    for value, g in df.groupby(col, observed=True):
        y = g["y_true"].to_numpy()
        row = {"срез": value, "n": len(g), "доля": round(len(g) / len(df), 4)}
        for name, pcol in models.items():
            p = g[pcol].to_numpy()
            row[f"acc_{name}"] = round(float((y == p).mean()), 4)
            row[f"mae_{name}"] = round(float(np.abs(y - p).mean()), 4)
            row[f"f1_{name}"] = (round(_f1_from_counts(_counts(y, p), present_only=True), 4)
                                 if len(g) >= min_n else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_table(y, preds: dict) -> pd.DataFrame:
    rows = []
    for c in LABELS:
        n = int((y == c).sum())
        row = {"класс": c, "n": n}
        for name, p in preds.items():
            tp = int(((y == c) & (p == c)).sum())
            rec, prec = tp / max(n, 1), tp / max(int((p == c).sum()), 1)
            row[f"recall_{name}"] = round(rec, 4)
            row[f"precision_{name}"] = round(prec, 4)
            row[f"f1_{name}"] = round(2 * rec * prec / (rec + prec), 4) if rec + prec else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def distance_table(y, preds: dict) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        d = np.abs(y - p)
        err = float((d > 0).mean())
        rows.append({"модель": name, "верно": round(float((d == 0).mean()), 4),
                     "ошибка на 1": round(float((d == 1).mean()), 4),
                     "ошибка на 2+": round(float((d >= 2).mean()), 4),
                     "доля 2+ среди ошибок": round(float((d >= 2).mean() / err), 4) if err else np.nan})
    return pd.DataFrame(rows)


def confidence_table(y, p, conf, name: str) -> tuple[pd.DataFrame, float]:
    """Точность по корзинам уверенности (max вероятность) и ECE."""
    bins = pd.cut(pd.Series(conf), CONF_EDGES, labels=CONF_LABELS, right=False)
    g = pd.DataFrame({"bin": bins, "ok": (y == p).astype(float), "conf": conf}).groupby("bin", observed=True)
    out = g.agg(n=("ok", "size"), mean_conf=("conf", "mean"), acc=("ok", "mean")).reset_index()
    ece = float((out["n"] / out["n"].sum() * (out["acc"] - out["mean_conf"]).abs()).sum())
    out.insert(0, "модель", name)
    return out.round(4).rename(columns={"bin": "уверенность"}), round(ece, 4)


# ------------------------------------------------------------------------ примеры
def _clean_text(t: str, n: int = 350) -> str:
    t = " ".join(str(t).replace("|", "/").split())
    return t if len(t) <= n else t[:n] + "…"


def sample_examples(df: pd.DataFrame, n: int = 12, seed: int = 42) -> pd.DataFrame:
    groups = {
        "верен только бейзлайн": df[df["agreement"] == AGREE[2]].sample(
            min(n, (df["agreement"] == AGREE[2]).sum()), random_state=seed),
        "верен только трансформер": df[df["agreement"] == AGREE[3]].sample(
            min(n, (df["agreement"] == AGREE[3]).sum()), random_state=seed),
    }
    both = df[df["agreement"] == AGREE[1]].assign(_c=lambda d: d[["conf_base", "conf_tr"]].min(axis=1))
    groups["обе ошиблись, уверенно"] = both.nlargest(n, "_c")
    parts = [g.assign(группа=name) for name, g in groups.items()]
    cols = ["группа", "row_id", "y_true", "pred_base", "pred_tr", "conf_base", "conf_tr",
            "rubric_main", "mention_group", "text"]
    return pd.concat(parts)[cols].round({"conf_base": 3, "conf_tr": 3}).reset_index(drop=True)


# ------------------------------------------------------------------------ графики
def plot_reliability(tables: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0.2, 1], [0.2, 1], "k--", lw=1, label="идеальная калибровка")
    for name, t in tables.items():
        ax.plot(t["mean_conf"], t["acc"], marker="o", label=name)
    ax.set_xlabel("средняя уверенность в корзине")
    ax.set_ylabel("доля верных ответов")
    ax.set_title("Калибровка")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_by_length(tbl: pd.DataFrame, models: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(tbl))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(models):
        ax.bar(x + i * w, tbl[f"acc_{m}"], w, label=m)
    ax.set_xticks(x + w * (len(models) - 1) / 2, tbl["срез"])
    ax.set_xlabel("длина отзыва, символов")
    ax.set_ylabel("accuracy")
    ax.set_title("Точность по длине отзыва")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------- запуск
def _md(df: pd.DataFrame) -> str:
    return md_table(df, index=False)


def _confusion_md(y, p) -> str:
    cm = confusion_df(y, p, normalize=True).round(3).reset_index().rename(columns={"index": "истина"})
    return _md(cm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--allow-test", action="store_true",
                    help="разрешить разбор на test (только для подтверждения выводов, сделанных на val)")
    ap.add_argument("--seed", type=int, default=42, help="сид трансформера для сравнения с бейзлайном")
    ap.add_argument("--top-rubrics", type=int, default=15)
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--clean", default="data/processed/reviews_clean.parquet")
    args = ap.parse_args()

    if args.split == "test" and not args.allow_test:
        raise PermissionError("Разбор на test только с --allow-test и только для подтверждения выводов, "
                              "сделанных на val.")

    cfg = load_config()
    tr_cfg = load_config("configs/transformer.yaml")
    seeds = list(tr_cfg["seeds"])
    if args.seed not in seeds:
        raise ValueError(f"--seed {args.seed} нет в configs/transformer.yaml: {seeds}")
    pred_dir = Path(tr_cfg["output"]["predictions_dir"])
    reports = Path(cfg["paths"]["reports_dir"])
    split = args.split
    tag = f"{split}"

    df = build_table(split, args.seed, seeds, pred_dir, Path(args.clean))
    df.drop(columns=["text"]).to_parquet(pred_dir / f"error_table_{split}.parquet", index=False)
    y = df["y_true"].to_numpy()
    preds = {"baseline": df["pred_base"].to_numpy(), "transformer": df["pred_tr"].to_numpy(),
             "ensemble": df["pred_ens"].to_numpy()}
    short = {"base": "pred_base", "tr": "pred_tr"}
    tdir = reports / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    md = [f"# Разбор ошибок ({split}, n={len(df):,})\n",
          f"Бейзлайн: TF-IDF word+char + LogReg. Трансформер: rubert-tiny2, seed {args.seed}; "
          f"ансамбль: среднее вероятностей по сидам {seeds}.\n"]

    # 1. общие метрики с интервалами
    single, diffs = cluster_bootstrap(y, preds, df["group_key"], args.n_boot,
                                      pairs=(("transformer", "baseline"), ("ensemble", "transformer"),
                                             ("ensemble", "baseline")))
    overall = []
    for name, p in preds.items():
        m = compute_metrics(y, p)
        _, lo, hi = single[name]
        overall.append({"модель": name, "macro_f1": round(m["macro_f1"], 4), "ci95_low": round(lo, 4),
                        "ci95_high": round(hi, 4), "mae": round(m["mae"], 4),
                        "accuracy": round(m["accuracy"], 4), "qwk": round(m["qwk"], 4)})
    overall = pd.DataFrame(overall)
    diff_tbl = pd.DataFrame([{"разность macro-F1": f"{a} - {b}", "оценка": round(v[0], 4),
                              "ci95_low": round(v[1], 4), "ci95_high": round(v[2], 4),
                              "значима": "да" if v[1] > 0 or v[2] < 0 else "нет"}
                             for (a, b), v in diffs.items()])
    md += [f"## 1. Общие метрики (бутстрэп по group_key, {args.n_boot} повторов)\n", _md(overall), "",
           _md(diff_tbl), ""]
    overall.to_csv(tdir / f"error_overall_{tag}.csv", index=False)

    # 2. по классам и матрицы ошибок
    pc = per_class_table(y, preds)
    md += ["## 2. По классам\n", _md(pc), ""]
    pc.to_csv(tdir / f"error_per_class_{tag}.csv", index=False)
    for name in ("baseline", "transformer"):
        md += [f"Матрица ошибок {name} (доля от истинного класса):\n", _confusion_md(y, preds[name]), ""]

    # 3. расстояние ошибки
    dist = distance_table(y, preds)
    md += ["## 3. Насколько ошибаются\n", _md(dist), ""]
    dist.to_csv(tdir / f"error_distance_{tag}.csv", index=False)

    # 4. уверенность и калибровка
    conf_parts, ece_rows, rel = [], [], {}
    for name, c in (("baseline", "conf_base"), ("transformer", "conf_tr"), ("ensemble", "conf_ens")):
        t, ece = confidence_table(y, preds[name], df[c].to_numpy(), name)
        conf_parts.append(t)
        ece_rows.append({"модель": name, "ECE": ece})
        rel[name] = t
    conf = pd.concat(conf_parts)
    md += ["## 4. Уверенность и калибровка\n", _md(pd.DataFrame(ece_rows)), "", _md(conf), ""]
    conf.to_csv(tdir / f"error_confidence_{tag}.csv", index=False)
    plot_reliability(rel, reports / "figures" / f"error_reliability_{tag}.png")

    # 5. срезы
    md.append("## 5. Срезы\n")
    top = df["rubric_main"].value_counts().head(args.top_rubrics).index
    df["rubric_slice"] = np.where(df["rubric_main"].isin(top), df["rubric_main"], "прочие")
    for title, col in (("Длина отзыва, символов", "len_bin"), ("Эмодзи в тексте", "has_emoji"),
                       ("Прямое упоминание оценки в тексте", "mention_group"),
                       (f"Основная рубрика (топ-{args.top_rubrics})", "rubric_slice")):
        t = slice_table(df, col, short)
        if col == "rubric_slice":
            t = t.sort_values("n", ascending=False)
        if col == "len_bin":
            plot_by_length(t, list(short), reports / "figures" / f"error_by_length_{tag}.png")
        t.to_csv(tdir / f"error_slice_{col}_{tag}.csv", index=False)
        md += [f"### {title}\n", _md(t), ""]
    with_m = df["mention_group"] != NO_MENTION
    md.append(f"Доля отзывов с прямым упоминанием оценки: {with_m.mean():.1%}; среди них оценка в тексте "
              f"не совпадает с меткой в {(df.loc[with_m, 'mention_group'] == MENTION_DIFF).mean():.1%} "
              "случаев (оценка в тексте найдена эвристикой, это грубая оценка шума меток).\n")

    # 6. согласие моделей
    ag = pd.crosstab(df["y_true"], df["agreement"]).reindex(columns=AGREE, fill_value=0)
    ag.loc["все"] = ag.sum()
    ag = ag.reset_index().rename(columns={"y_true": "истина"})
    md += ["## 6. Согласие моделей\n", _md(ag), ""]
    ag.to_csv(tdir / f"error_agreement_{tag}.csv", index=False)

    # 7. примеры
    ex = sample_examples(df, args.examples)
    ex.to_csv(tdir / f"error_examples_{tag}.csv", index=False)
    ex_md = ex.assign(text=ex["text"].map(_clean_text))
    md += ["## 7. Примеры\n", _md(ex_md), ""]

    out = reports / f"error_analysis_{split}_tables.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(overall.to_string(index=False))
    print("\n" + diff_tbl.to_string(index=False))
    print(f"\nОтчёт: {out}")


if __name__ == "__main__":
    main()

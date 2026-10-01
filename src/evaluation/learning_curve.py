"""
Фаза 3: learning curve бейзлайна (подвыборки только из train, метрики на val).

    python -m src.evaluation.learning_curve                 # полная кривая по baseline.yaml
    python -m src.evaluation.learning_curve --sizes 10000 --repeats 1 --no-log   # замер времени

Для каждого размера делается n_repeats независимых подвыборок (seed = base + номер повтора);
каждый запуск пишется отдельной строкой в experiments.csv. Размер выбирается по правилу из
baseline.yaml (learning_curve.min_gain), заданному ДО запуска.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audit import md_table
from src.data.loader import load_config
from src.data.splitting import load_split_data
from src.experiments.tracking import set_seed
from src.models.baseline import parse_class_weight, run_experiment, select_eval


def choose_size(summary: pd.DataFrame, min_gain: float):
    """Наименьший размер, после которого следующий шаг даёт прирост macro-F1 меньше
    max(шум, min_gain), где шум = наибольшее std между повторами двух соседних точек.
    Если такого нет, берётся наибольший размер (кривая не вышла на плато)."""
    sizes = list(summary.index)
    for a, b in zip(sizes[:-1], sizes[1:]):
        gain = summary.at[b, "f1_mean"] - summary.at[a, "f1_mean"]
        noise = max(summary.at[a, "f1_std"], summary.at[b, "f1_std"])
        if gain < max(noise, min_gain):
            return a
    return sizes[-1]


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby("size")
    s = pd.DataFrame({
        "f1_mean": g["macro_f1"].mean(), "f1_std": g["macro_f1"].std(),
        "mae_mean": g["mae"].mean(), "mae_std": g["mae"].std(),
        "acc_mean": g["accuracy"].mean(),
        "train_time": g["train_time"].mean(), "infer_time": g["inference_time"].mean(),
    })
    return s.fillna({"f1_std": 0.0, "mae_std": 0.0})


def plot_curve(summary: pd.DataFrame, chosen, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = summary.index.to_numpy()
    ax[0].errorbar(x, summary["f1_mean"], yerr=summary["f1_std"], marker="o", capsize=3)
    ax[0].set_ylabel("macro-F1 (val)")
    ax[1].errorbar(x, summary["mae_mean"], yerr=summary["mae_std"], marker="o", capsize=3, color="tab:red")
    ax[1].set_ylabel("MAE (val)")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("размер train")
        a.set_xticks(x, [f"{int(v / 1000)}k" for v in x])
        a.axvline(chosen, ls="--", color="gray", label=f"выбрано: {int(chosen / 1000)}k")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Learning curve: TF-IDF + LogReg (среднее ± std по независимым подвыборкам)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--no-log", action="store_true", help="замер: ничего не писать на диск")
    args = ap.parse_args()

    cfg = load_config()
    bl = load_config("configs/baseline.yaml")
    lc = bl["learning_curve"]
    sel = bl["selected"]
    if sel["variant"] in (None, "TODO"):
        raise ValueError("Сначала выбери конфигурацию: заполни selected.variant и selected.class_weight "
                         "в configs/baseline.yaml по результатам сравнения.")
    variant, cw = sel["variant"], parse_class_weight(sel["class_weight"])
    sizes = args.sizes or lc["sizes"]
    repeats = args.repeats or lc["n_repeats"]
    seed = bl["seed"]
    set_seed(seed)

    data = load_split_data()
    pool = data[data["split"] == "train"]
    val = select_eval(data, "val")
    print(f"Кривая: {variant}, class_weight={sel['class_weight']}, размеры {sizes}, повторов {repeats}")

    rows = []
    for size in sizes:
        for rep in range(repeats):
            sub_seed = seed + rep
            train = pool.sample(size, random_state=sub_seed)
            r = run_experiment(train, val, variant, cw, bl, cfg["paths"]["experiments_csv"],
                               cfg["split"]["version"], "val", sub_seed,
                               notes=f"learning curve size={size} repeat={rep}", log=not args.no_log)
            rows.append({"size": size, "repeat": rep, "seed": sub_seed, **{
                k: r[k] for k in ("macro_f1", "mae", "accuracy", "train_time", "inference_time")}})
            print(f"  {size:>7,} повтор {rep}: macro-F1 {r['macro_f1']:.4f} MAE {r['mae']:.4f} "
                  f"({r['train_time']:.0f} c)")

    runs = pd.DataFrame(rows)
    summary = summarize(runs)
    chosen = choose_size(summary, lc["min_gain"])
    print("\n", summary.round(4).to_string())
    print(f"\nВыбранный размер по правилу (min_gain={lc['min_gain']}): {chosen:,}")
    if args.no_log:
        return

    out = Path(cfg["paths"]["reports_dir"])
    (out / "tables").mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "tables" / "learning_curve_runs.csv", index=False)
    table = pd.DataFrame({
        "Размер train": [f"{int(s / 1000)}k" for s in summary.index],
        "Macro-F1": [f"{m:.4f} ± {d:.4f}" for m, d in zip(summary["f1_mean"], summary["f1_std"])],
        "MAE": [f"{m:.4f} ± {d:.4f}" for m, d in zip(summary["mae_mean"], summary["mae_std"])],
        "Время обучения, с": summary["train_time"].round(1).to_numpy(),
        "Инференс на val, с": summary["infer_time"].round(1).to_numpy(),
    })
    text = (f"# Learning curve: TF-IDF + LogReg ({variant}, class_weight={sel['class_weight']})\n\n"
            f"Метрики на val, среднее ± std по {repeats} независимым подвыборкам train.\n\n"
            + md_table(table, index=False)
            + f"\n\n**Выбранный размер: {chosen:,}.** Правило (задано до запуска): наименьший размер, "
              f"после которого следующий шаг даёт прирост macro-F1 меньше max(std между повторами, "
              f"{lc['min_gain']}).\n\n"
              "Оговорка: кривая построена для TF-IDF + LogReg и не доказывает оптимальность размера "
              "для трансформера.\n")
    (out / "tables" / "learning_curve.md").write_text(text, encoding="utf-8")
    plot_curve(summary, chosen, out / "figures" / "learning_curve.png")
    print(f"Таблица и график: {out / 'tables' / 'learning_curve.md'}, {out / 'figures' / 'learning_curve.png'}")


if __name__ == "__main__":
    main()

"""
Фаза 5: температурное масштабирование вероятностей (калибровка).

    python -m src.evaluation.calibration                  # подбор T на val, метрики на val
    python -m src.evaluation.calibration --allow-test     # + проверка на test (T уже зафиксирована по val)

Температура подбирается ТОЛЬКО на val минимизацией log-loss (один параметр на модель).
softmax(log p / T) эквивалентен масштабированию логитов, поэтому сохранённых вероятностей достаточно.
Преобразование монотонно: argmax, а значит accuracy, MAE и macro-F1, не меняются; меняются
вероятности (log-loss, Brier, ECE). На test T только применяется, не подбирается.

Выход: reports/calibration.json (T для сервиса), reports/tables/calibration_{val|val_test}.csv,
reports/figures/calibration_reliability_{split}.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.data.audit import md_table
from src.data.loader import load_config
from src.evaluation.error_analysis import PROB_COLS, _load_pred, confidence_table, plot_reliability

EPS = 1e-12


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(probs, EPS, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y - 1], EPS, 1.0)).mean())


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y - 1]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """T > 1 смягчает переуверенную модель, T < 1 заостряет недоуверенную."""
    res = minimize_scalar(lambda u: log_loss(apply_temperature(probs, float(np.exp(u))), y),
                          bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def evaluate(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(axis=1) + 1
    _, ece = confidence_table(y, pred, probs.max(axis=1), "m")
    return {"log_loss": round(log_loss(probs, y), 4), "brier": round(brier(probs, y), 4), "ECE": ece,
            "mean_conf": round(float(probs.max(axis=1).mean()), 4),
            "accuracy": round(float((pred == y).mean()), 4)}


def _probs(pred_dir: Path, split: str, seeds: list, seed: int) -> tuple[np.ndarray, dict]:
    base = _load_pred(pred_dir, "baseline", split)
    trs = {s: _load_pred(pred_dir, f"seed{s}", split) for s in seeds}
    for d in trs.values():
        if not (d["row_id"].equals(base["row_id"]) and d["y_true"].equals(base["y_true"])):
            raise ValueError("Предсказания моделей не совпадают по row_id/y_true")
    y = base["y_true"].to_numpy().astype(np.int64)
    probs = {"baseline": base[PROB_COLS].to_numpy(), "transformer": trs[seed][PROB_COLS].to_numpy(),
             "ensemble": np.mean([d[PROB_COLS].to_numpy() for d in trs.values()], axis=0)}
    return y, probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="сид трансформера")
    ap.add_argument("--allow-test", action="store_true",
                    help="проверить откалиброванные вероятности на test (T фиксирована по val)")
    args = ap.parse_args()

    cfg = load_config()
    tr_cfg = load_config("configs/transformer.yaml")
    seeds = list(tr_cfg["seeds"])
    if args.seed not in seeds:
        raise ValueError(f"--seed {args.seed} нет в configs/transformer.yaml: {seeds}")
    pred_dir = Path(tr_cfg["output"]["predictions_dir"])
    reports = Path(cfg["paths"]["reports_dir"])

    splits = ["val"] + (["test"] if args.allow_test else [])
    data = {s: _probs(pred_dir, s, seeds, args.seed) for s in splits}

    temps, rows = {}, []
    for name in ("baseline", "transformer", "ensemble"):
        y_val, p_val = data["val"][0], data["val"][1][name]
        T = fit_temperature(p_val, y_val)
        temps[name] = round(T, 4)
        for split in splits:
            y, p = data[split][0], data[split][1][name]
            cal = apply_temperature(p, T)
            assert (cal.argmax(axis=1) == p.argmax(axis=1)).all(), "температура изменила argmax"
            for label, probs in (("до", p), ("после", cal)):
                rows.append({"модель": name, "T": temps[name], "split": split, "калибровка": label,
                             **evaluate(probs, y)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    tag = "val_test" if args.allow_test else "val"
    (reports / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "tables" / f"calibration_{tag}.csv", index=False)
    (reports / "tables" / f"calibration_{tag}.md").write_text(md_table(table, index=False), encoding="utf-8")
    out = {"method": "temperature scaling: softmax(log p / T), T fitted on val by log-loss",
           "transformer_seed": args.seed, "temperature": temps}
    (reports / "calibration.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_split = splits[-1]
    y, p = data[eval_split][0], data[eval_split][1]["transformer"]
    rel = {}
    for label, probs in (("до калибровки", p), ("после калибровки", apply_temperature(p, temps["transformer"]))):
        rel[label], _ = confidence_table(y, probs.argmax(axis=1) + 1, probs.max(axis=1), label)
    fig = reports / "figures" / f"calibration_reliability_{eval_split}.png"
    plot_reliability(rel, fig)
    print(f"\nT: {temps}\nСохранено: {reports / 'calibration.json'}, {fig}")


if __name__ == "__main__":
    main()

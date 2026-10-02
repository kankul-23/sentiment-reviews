"""
Фаза 4: агрегация по seed'ам и итоговая таблица «качество против стоимости».

    python -m src.evaluation.comparison        # собрать reports/tables/model_comparison.md

Читает:
    reports/final_test_eval.json               бейзлайн (Фаза 3)
    reports/final_test_eval_transformer.json   rubert-tiny2, 3 seed'а (Фаза 4)
    reports/tables/transformer_cost.json       размер, задержка (python -m src.models.transformer --cost)
Любой из файлов может отсутствовать: тогда соответствующие ячейки помечаются «—».
"""
import json
from pathlib import Path

import numpy as np

from src.data.audit import md_table  # noqa: F401  (оставлено для единого стиля таблиц)
from src.data.loader import load_config

METRIC_KEYS = ("macro_f1", "mae", "accuracy", "qwk", "f1_1", "f1_2", "f1_3", "f1_4", "f1_5")


def mean_std(values) -> tuple[float, float]:
    """Среднее и выборочное std (ddof=1). Для одного значения std = 0."""
    arr = np.asarray(list(values), dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


def aggregate(per_seed: list[dict], split: str = "test", keys=METRIC_KEYS) -> dict:
    """per_seed: список записей {"seed": ..., "test": {metric: value}, "val": {...}}."""
    return {k: dict(zip(("mean", "std"), mean_std(r[split][k] for r in per_seed))) for k in keys}


def _pm(stat: dict, digits: int = 4) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def comparison_markdown(baseline: dict | None, transformer: dict | None, cost: dict | None,
                        n_test: int | None = None) -> str:
    """Таблица сравнения на test. baseline: содержимое final_test_eval.json;
    transformer: final_test_eval_transformer.json; cost: transformer_cost.json."""
    header = ["Модель", "Macro-F1", "MAE", "Accuracy", "QWK",
              "Обучение, мин", "Инференс, с / 1000 отзывов", "Размер, МБ"]
    rows = []
    if baseline:
        m = baseline["metrics"]
        n = n_test or baseline.get("n_test") or (transformer or {}).get("n_test")
        per_1k = f"{m['inference_time'] / n * 1000:.2f} (CPU)" if n else "—"
        rows.append(["TF-IDF + LogReg (1 запуск, seed 42)",
                     f"{m['macro_f1']:.4f}", f"{m['mae']:.4f}", f"{m['accuracy']:.4f}",
                     f"{m['qwk']:.4f}", f"{m['train_time'] / 60:.1f}", per_1k, "— (не сохранён)"])
    if transformer:
        agg = transformer["aggregate"]
        seeds = transformer["seeds"]
        train_min = np.mean([r["train_time"] for r in transformer["per_seed"]]) / 60
        n = n_test or transformer.get("n_test")
        inf = np.mean([r["test"]["inference_time"] for r in transformer["per_seed"]])
        dev = {"cuda": "GPU", "cpu": "CPU"}.get(transformer.get("device"), "устройство не записано")
        per_1k = f"{inf / n * 1000:.2f} ({dev})" if n else "—"
        if cost:  # то же на CPU: батч 32, измерено в --cost на 1000 отзывах val
            per_1k += f"; {1000 / cost['throughput_texts_per_s']:.2f} (CPU, batch 32)"
        size = f"{cost['size_mb']:.0f}" if cost else "—"
        rows.append([f"rubert-tiny2 ({len(seeds)} seed'а, mean ± std)",
                     _pm(agg["macro_f1"]), _pm(agg["mae"]), _pm(agg["accuracy"]), _pm(agg["qwk"]),
                     f"{train_min:.1f}", per_1k, size])

    def line(cells):
        return "| " + " | ".join(cells) + " |"

    out = [line(header), "|" + "---|" * len(header)] + [line(r) for r in rows]
    text = "\n".join(out)

    if transformer:
        agg = transformer["aggregate"]
        per_class = ["", "F1 по классам (test, mean ± std):", ""]
        per_class += [f"- {c}★: {_pm(agg[f'f1_{c}'])}" for c in range(1, 6)]
        text += "\n" + "\n".join(per_class)
    if cost:
        text += (f"\n\nЗадержка одного запроса на CPU (batch 1, {cost['torch_threads']} потоков): "
                 f"медиана {cost['latency_ms_p50']:.1f} мс, p95 {cost['latency_ms_p95']:.1f} мс; "
                 f"пропускная способность (batch {cost['throughput_batch']}): "
                 f"{cost['throughput_texts_per_s']:.0f} отзывов/с; параметров: {cost['n_params'] / 1e6:.1f} млн.")
    text += ("\n\nИнференс: бейзлайн измерен на CPU по всей test-части. Для трансформера первое число — "
             "по всей test-части на устройстве финального прогона, "
             "второе — на CPU по 1000 отзывам val (transformer_cost.json). Сравнивать скорость нужно "
             "по значениям на CPU. Время включает токенизацию/векторизацию. Различия меньше разброса "
             "по seed'ам не считаются улучшением.\n")
    return text


def main() -> None:
    cfg = load_config()
    rep = Path(cfg["paths"]["reports_dir"])
    baseline = _load_json(rep / "final_test_eval.json")
    transformer = _load_json(rep / "final_test_eval_transformer.json")
    cost = _load_json(rep / "tables" / "transformer_cost.json")
    text = "# Сравнение моделей на test\n\n" + comparison_markdown(baseline, transformer, cost)
    out = rep / "tables" / "model_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nТаблица: {out}")


if __name__ == "__main__":
    main()

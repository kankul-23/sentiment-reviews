"""Метрики для 5-классовой классификации рейтинга (Фаза 3).

Основная метрика macro-F1 считается по всем пяти классам 1-5 (класс, которого нет
ни в ответах, ни в предсказаниях, даёт F1 = 0). MAE считается по исходной шкале 1-5.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

LABELS = (1, 2, 3, 4, 5)


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per_class = f1_score(y_true, y_pred, labels=list(LABELS), average=None, zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(LABELS))
    out = {
        "macro_f1": float(per_class.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "qwk": float(np.nan_to_num(qwk)),
    }
    out.update({f"f1_{label}": float(v) for label, v in zip(LABELS, per_class)})
    return out


def confusion_df(y_true, y_pred, normalize: bool = False) -> pd.DataFrame:
    """Матрица ошибок 5x5. normalize=True: по строкам (доля от истинного класса = recall)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS)).astype(float)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    else:
        cm = cm.astype(int)
    return pd.DataFrame(cm, index=[f"true {c}" for c in LABELS], columns=[f"pred {c}" for c in LABELS])


def plot_confusion(y_true, y_pred, path, title: str = "Матрица ошибок") -> None:
    """Тепловая карта: цвет = доля от истинного класса, подписи = число отзывов."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = confusion_df(y_true, y_pred).to_numpy()
    share = confusion_df(y_true, y_pred, normalize=True).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5), [str(c) for c in LABELS])
    ax.set_yticks(range(5), [str(c) for c in LABELS])
    ax.set_xlabel("предсказано")
    ax.set_ylabel("истинная оценка")
    ax.set_title(title)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{counts[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="доля от истинного класса")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

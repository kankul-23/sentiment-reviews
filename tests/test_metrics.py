import numpy as np
import pytest

from src.evaluation.metrics import LABELS, compute_metrics, confusion_df


def test_perfect_prediction():
    y = np.array([1, 2, 3, 4, 5, 5])
    m = compute_metrics(y, y)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["qwk"] == pytest.approx(1.0)


def test_known_values():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 4])
    m = compute_metrics(y_true, y_pred)
    # F1 по классам: 1, 1, 1, 2/3, 0
    assert m["macro_f1"] == pytest.approx((3 + 2 / 3) / 5)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["mae"] == pytest.approx(0.2)


def test_mae_uses_rating_scale():
    m = compute_metrics([1, 5], [5, 1])
    assert m["mae"] == pytest.approx(4.0)


def test_constant_five_baseline():
    rng = np.random.default_rng(0)
    y = rng.choice(LABELS, size=20000, p=[0.07, 0.02, 0.04, 0.08, 0.79])
    m = compute_metrics(y, np.full_like(y, 5))
    p5 = (y == 5).mean()
    assert m["accuracy"] == pytest.approx(p5)
    assert m["macro_f1"] == pytest.approx(2 * p5 / (1 + p5) / 5, abs=1e-6)  # ~0.175
    assert m["f1_1"] == 0.0


def test_missing_class_counts_as_zero():
    # класс 3 нигде не встречается: macro-F1 всё равно по 5 классам
    m = compute_metrics([1, 2, 4, 5], [1, 2, 4, 5])
    assert m["macro_f1"] == pytest.approx(4 / 5)


def test_confusion_shape_and_rows():
    cm = confusion_df([1, 1, 5, 5, 5], [1, 5, 5, 5, 4])
    assert cm.shape == (5, 5)
    assert cm.loc["true 5"].sum() == 3
    assert cm.loc["true 1", "pred 5"] == 1
    norm = confusion_df([1, 1, 5], [1, 5, 5], normalize=True)
    assert norm.loc["true 1"].sum() == pytest.approx(1.0)

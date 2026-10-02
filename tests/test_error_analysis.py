import numpy as np
import pandas as pd

from src.evaluation.error_analysis import (
    AGREE,
    NO_MENTION,
    MENTION_DIFF,
    MENTION_MATCH,
    _counts,
    _f1_from_counts,
    add_features,
    agreement_labels,
    cluster_bootstrap,
    confidence_table,
    distance_table,
    mention_rating,
    per_class_table,
    slice_table,
)
from src.evaluation.metrics import compute_metrics


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(1, 6, n)
    noisy = np.clip(y + rng.integers(-1, 2, n), 1, 5)
    return y, noisy


def test_mention_rating_digits_and_words():
    assert mention_rating("Ставлю 5 звёзд, всё отлично") == 5
    assert mention_rating("Итого 4 из 5") == 4
    assert mention_rating("Оценка: 3") == 3
    assert mention_rating("поставил бы тройку") == 3
    assert mention_rating("на пятёрку") == 5
    assert np.isnan(mention_rating("Были в 5 часов вечера, очень вкусно"))
    assert np.isnan(mention_rating("10 из 10 рекомендую"))


def test_mention_rating_takes_earliest():
    assert mention_rating("сначала 2 звезды, потом поставил 5") == 2


def test_f1_from_counts_matches_compute_metrics():
    y, p = _toy()
    assert abs(_f1_from_counts(_counts(y, p)) - compute_metrics(y, p)["macro_f1"]) < 1e-9


def test_f1_present_only_ignores_absent_classes():
    y = np.array([1, 1, 2, 2])
    assert _f1_from_counts(_counts(y, y)) == 2 / 5  # absent classes count as 0
    assert _f1_from_counts(_counts(y, y), present_only=True) == 1.0


def test_agreement_labels():
    y = np.array([1, 2, 3, 4])
    base = np.array([1, 1, 3, 1])
    tr = np.array([1, 2, 1, 1])
    assert list(agreement_labels(y, base, tr)) == [AGREE[0], AGREE[3], AGREE[2], AGREE[1]]


def test_add_features_groups():
    df = pd.DataFrame({
        "text": ["Супер, 5 звёзд 😀", "Так себе", "Ставлю 2, не понравилось"],
        "rubrics": ["Кафе;Бар", None, "Отель"],
        "y_true": [5, 3, 4], "pred_base": [5, 3, 4], "pred_tr": [5, 2, 4],
    })
    out = add_features(df)
    assert list(out["mention_group"]) == [MENTION_MATCH, NO_MENTION, MENTION_DIFF]
    assert list(out["has_emoji"]) == ["да", "нет", "нет"]
    assert list(out["rubric_main"]) == ["Кафе", "без рубрики", "Отель"]


def test_slice_table_and_min_n():
    y, p = _toy()
    df = pd.DataFrame({"y_true": y, "pred_base": p, "pred_tr": y, "g": np.where(np.arange(len(y)) < 50, "a", "b")})
    t = slice_table(df, "g", {"base": "pred_base", "tr": "pred_tr"}, min_n=100)
    assert set(t["срез"]) == {"a", "b"}
    assert t.loc[t["срез"] == "a", "f1_base"].isna().all()
    assert (t["acc_tr"] == 1.0).all()
    assert abs(t["доля"].sum() - 1) < 1e-3


def test_per_class_and_distance_tables():
    y = np.array([1, 2, 3, 4, 5, 5])
    p = np.array([1, 2, 3, 4, 3, 5])
    pc = per_class_table(y, {"m": p})
    assert pc.loc[pc["класс"] == 5, "recall_m"].iloc[0] == 0.5
    d = distance_table(y, {"m": p}).iloc[0]
    assert abs(d["верно"] - 5 / 6) < 1e-3 and abs(d["ошибка на 2+"] - 1 / 6) < 1e-3
    assert d["доля 2+ среди ошибок"] == 1.0


def test_confidence_table_ece_perfect_calibration():
    y = np.array([1] * 10)
    p = np.array([1] * 8 + [2] * 2)
    conf = np.full(10, 0.8)
    t, ece = confidence_table(y, p, conf, "m")
    assert t["n"].sum() == 10 and ece < 1e-9


def test_cluster_bootstrap_detects_difference_and_orders_interval():
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.integers(1, 6, n)
    good = np.where(rng.random(n) < 0.9, y, rng.integers(1, 6, n))
    bad = np.where(rng.random(n) < 0.4, y, rng.integers(1, 6, n))
    groups = rng.integers(0, 300, n)
    single, diffs = cluster_bootstrap(y, {"good": good, "bad": bad}, groups, n_boot=200, seed=0,
                                      pairs=(("good", "bad"),))
    pt, lo, hi = diffs[("good", "bad")]
    assert lo <= pt <= hi and lo > 0
    for pt, lo, hi in single.values():
        assert lo <= pt <= hi


def test_cluster_bootstrap_is_reproducible():
    y, p = _toy(400)
    g = np.arange(400) // 4
    a = cluster_bootstrap(y, {"m": p}, g, n_boot=50, seed=3)[0]["m"]
    b = cluster_bootstrap(y, {"m": p}, g, n_boot=50, seed=3)[0]["m"]
    assert a == b

import numpy as np
import pandas as pd
import pytest

from src.evaluation.learning_curve import choose_size
from src.models.baseline import build_pipeline, parse_class_weight, select_eval

BL = {
    "tfidf": {
        "word": {"analyzer": "word", "ngram_range": [1, 2], "min_df": 1, "sublinear_tf": True},
        "char": {"analyzer": "char_wb", "ngram_range": [2, 4], "min_df": 1, "sublinear_tf": True},
    },
    "model": {"C": 1.0, "max_iter": 100, "solver": "lbfgs"},
}


def tiny_df():
    texts = ["отлично вкусно", "ужасно грязно", "нормально", "хорошо быстро", "плохо долго", "отлично"] * 5
    ratings = [5, 1, 3, 4, 2, 5] * 5
    return pd.DataFrame({"text": texts, "rating": ratings,
                         "split": (["train"] * 3 + ["val"] * 2 + ["test"]) * 5})


def test_test_split_is_protected():
    df = tiny_df()
    with pytest.raises(PermissionError):
        select_eval(df, "test")
    assert set(select_eval(df, "test", allow_test=True)["split"]) == {"test"}
    assert set(select_eval(df, "val")["split"]) == {"val"}


@pytest.mark.parametrize("variant", ["word", "char", "word+char"])
def test_pipeline_fits_and_predicts(variant):
    df = tiny_df()
    pipe = build_pipeline(variant, BL, None, seed=0)
    pipe.fit(df["text"], df["rating"])
    pred = pipe.predict(df["text"])
    assert len(pred) == len(df)
    assert set(pred) <= {1, 2, 3, 4, 5}


def test_class_weight_parsing():
    assert parse_class_weight("none") is None
    assert parse_class_weight(None) is None
    assert parse_class_weight("balanced") == "balanced"
    with pytest.raises(ValueError):
        parse_class_weight("TODO")


def _summary(f1_means, f1_stds, sizes=(10000, 25000, 50000, 100000)):
    return pd.DataFrame({"f1_mean": f1_means, "f1_std": f1_stds}, index=list(sizes))


def test_choose_size_stops_at_plateau():
    s = _summary([0.30, 0.36, 0.375, 0.378], [0.004, 0.004, 0.004, 0.004])
    # 10k->25k: +0.06 (растёт), 25k->50k: +0.015 (растёт), 50k->100k: +0.003 < 0.005 -> берём 50k
    assert choose_size(s, min_gain=0.005) == 50000


def test_choose_size_noise_can_stop_earlier():
    s = _summary([0.30, 0.31, 0.33, 0.36], [0.02, 0.02, 0.02, 0.02])
    # прирост 10k->25k равен 0.01 < шума 0.02, значит достаточно 10k
    assert choose_size(s, min_gain=0.005) == 10000


def test_choose_size_no_plateau_takes_largest():
    s = _summary([0.30, 0.34, 0.38, 0.42], [0.002] * 4)
    assert choose_size(s, min_gain=0.005) == 100000

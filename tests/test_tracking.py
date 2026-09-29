import csv

import numpy as np
import pytest

from src.experiments.tracking import FIELDS, log_experiment, set_seed


def test_log_experiment_writes_row(tmp_path):
    path = tmp_path / "experiments.csv"
    exp_id = log_experiment(
        {"phase": 0, "model": "dummy", "seed": 42, "macro_f1": 0.5,
         "model_params": {"C": 1.0}},
        path=path,
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == exp_id
    assert rows[0]["model"] == "dummy"
    assert list(rows[0].keys()) == FIELDS


def test_log_experiment_appends_without_duplicate_header(tmp_path):
    path = tmp_path / "experiments.csv"
    log_experiment({"model": "a"}, path=path)
    log_experiment({"model": "b"}, path=path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_unknown_field_raises(tmp_path):
    with pytest.raises(ValueError):
        log_experiment({"bogus": 1}, path=tmp_path / "e.csv")


def test_set_seed_reproducible():
    set_seed(42)
    a = np.random.rand(3)
    set_seed(42)
    b = np.random.rand(3)
    assert (a == b).all()
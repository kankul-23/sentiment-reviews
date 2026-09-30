import numpy as np
import pandas as pd
import pytest

from src.data.leakage import exact_duplicates_across_parts, group_overlap
from src.data.splitting import ids_hash, make_split


def make_df(n_groups: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 8, n_groups)
    gk = np.repeat([f"g{i}" for i in range(n_groups)], sizes)
    rating = rng.choice([1, 2, 3, 4, 5], size=len(gk), p=[0.07, 0.03, 0.05, 0.10, 0.75])
    return pd.DataFrame({
        "row_id": np.arange(len(gk)),
        "gk_name": gk,
        "rating": rating,
        "text": [f"text {i}" for i in range(len(gk))],
    })


@pytest.fixture(scope="module")
def df():
    d = make_df()
    d["split"] = make_split(d, seed=42)
    return d


def test_no_group_overlap(df):
    assert group_overlap(df, "gk_name") == 0


def test_all_rows_assigned(df):
    assert set(df["split"]) == {"train", "val", "test"}
    assert df["split"].notna().all()


def test_proportions(df):
    shares = df["split"].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.03)
    assert shares["val"] == pytest.approx(0.15, abs=0.03)
    assert shares["test"] == pytest.approx(0.15, abs=0.03)


def test_rating_distribution_is_close(df):
    # На маленьких данных StratifiedGroupKFold из scikit-learn 1.7 стратифицирует хуже,
    # чем 1.8, поэтому берём достаточно большой датасет и допуск 3 п.п. (на реальных
    # данных разброс между частями заметно меньше: см. reports/leakage_report.md).
    overall = df["rating"].value_counts(normalize=True)
    per_part = pd.crosstab(df["split"], df["rating"], normalize="index")
    assert (per_part - overall).abs().to_numpy().max() < 0.02


def test_reproducible_and_seed_sensitive():
    d = make_df(1500)
    a = make_split(d, seed=42)
    b = make_split(d, seed=42)
    c = make_split(d, seed=7)
    assert (a == b).all()
    assert not (a == c).all()


def test_overlap_check_detects_leak(df):
    broken = df.copy()
    g = broken.loc[broken["split"] == "train", "gk_name"].iloc[0]
    idx = broken.index[broken["gk_name"] == g][0]
    broken.loc[idx, "split"] = "test" if broken.at[idx, "split"] != "test" else "train"
    assert group_overlap(broken, "gk_name") == 1


def test_exact_duplicate_check():
    d = pd.DataFrame({"split": ["train", "test", "train", "val"], "text": ["a", "a", "b", "c"]})
    assert exact_duplicates_across_parts(d, d["text"]) == 1
    assert exact_duplicates_across_parts(d[d["text"] != "a"], d.loc[d["text"] != "a", "text"]) == 0


def test_ids_hash_is_order_independent():
    assert ids_hash([3, 1, 2]) == ids_hash([1, 2, 3])
    assert ids_hash([1, 2, 3]) != ids_hash([1, 2, 4])


def test_invalid_fractions_raise():
    with pytest.raises(ValueError):
        make_split(make_df(200), seed=1, fractions=(0.7, 0.2, 0.2))

import numpy as np
import pytest

from src.evaluation.comparison import METRIC_KEYS, aggregate, comparison_markdown, mean_std


def record(seed, f1, mae=0.27):
    m = {k: 0.5 for k in METRIC_KEYS}
    m.update({"macro_f1": f1, "mae": mae, "inference_time": 40.0})
    return {"seed": seed, "train_time": 600.0, "val": dict(m), "test": dict(m)}


def test_mean_std_uses_sample_std():
    mean, std = mean_std([0.50, 0.52, 0.54])
    assert mean == pytest.approx(0.52)
    assert std == pytest.approx(np.std([0.50, 0.52, 0.54], ddof=1))


def test_mean_std_single_value():
    assert mean_std([0.5]) == (0.5, 0.0)


def test_aggregate_over_seeds():
    agg = aggregate([record(42, 0.50), record(123, 0.52), record(2026, 0.54)], "test")
    assert agg["macro_f1"]["mean"] == pytest.approx(0.52)
    assert agg["macro_f1"]["std"] == pytest.approx(0.02)
    assert set(agg) == set(METRIC_KEYS)


def _transformer():
    recs = [record(42, 0.50), record(123, 0.52), record(2026, 0.54)]
    return {"seeds": [42, 123, 2026], "n_test": 74752, "per_seed": recs, "aggregate": aggregate(recs)}


def _baseline():
    return {"metrics": {"macro_f1": 0.5147, "mae": 0.2706, "accuracy": 0.8061, "qwk": 0.8214,
                        "train_time": 428.0, "inference_time": 37.1}}


def test_table_has_both_models_and_per_1000_inference():
    text = comparison_markdown(_baseline(), _transformer(), None)
    assert "TF-IDF + LogReg" in text and "rubert-tiny2" in text
    assert "0.5147" in text and "0.5200 ± 0.0200" in text
    assert "0.50" in text   # 37.1 / 74752 * 1000 = 0.496 -> 0.50 c на 1000 отзывов


def test_table_without_cost_and_baseline_does_not_crash():
    assert "—" in comparison_markdown(None, _transformer(), None)
    assert "TF-IDF" in comparison_markdown(_baseline(), None, None)


def test_table_with_cost():
    cost = {"size_mb": 116.3, "torch_threads": 4, "latency_ms_p50": 8.2, "latency_ms_p95": 14.0,
            "throughput_batch": 32, "throughput_texts_per_s": 250.0, "n_params": 29_200_000}
    text = comparison_markdown(_baseline(), _transformer(), cost)
    assert "116" in text and "8.2 мс" in text


def test_table_separates_gpu_and_cpu_inference():
    cost = {"size_mb": 119.2, "torch_threads": 24, "latency_ms_p50": 5.4, "latency_ms_p95": 8.3,
            "throughput_batch": 32, "throughput_texts_per_s": 911.0, "n_params": 29_195_333}
    tr = {**_transformer(), "device": "cuda"}
    text = comparison_markdown(_baseline(), tr, cost)
    assert "(CPU)" in text                      # бейзлайн
    assert "(GPU)" in text                      # финальный прогон трансформера на cuda
    assert "1.10 (CPU, batch 32)" in text       # 1000 / 911 = 1.10 с на 1000 отзывов
    old = comparison_markdown(_baseline(), _transformer(), cost)   # json без поля device
    assert "устройство не записано" in old

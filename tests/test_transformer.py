"""Тесты Фазы 4. Реальный rubert-tiny2 не скачивается: маленькая случайная BERT-модель и
игрушечный токенайзер. Проверяется механика (батчи, веса, обучение, предсказания), а не качество."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import BertConfig, BertForSequenceClassification  # noqa: E402

from src.models.baseline import select_eval  # noqa: E402
from src.models.transformer import (  # noqa: E402
    balanced_weights,
    collate,
    encode,
    inference_batches,
    make_batches,
    predict_proba,
    subsample,
    to_labels,
    to_ratings,
    train_model,
)


class FakeTokenizer:
    """Каждое новое слово получает следующий свободный id (без коллизий хэша);
    [CLS]=2 в начале, [SEP]=3 в конце; pad=0."""
    pad_token_id = 0
    unk_token_id = 1

    def __init__(self):
        self.vocab = {}

    def __call__(self, texts, truncation=True, max_length=None, add_special_tokens=True):
        out = []
        for t in texts:
            ids = [self.vocab.setdefault(w, 5 + len(self.vocab)) for w in t.split()]
            if add_special_tokens:
                ids = [2] + ids + [3]
            if truncation and max_length:
                ids = ids[:max_length]
            out.append(ids)
        return {"input_ids": out}


def tiny_model():
    cfg = BertConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=64, num_labels=5)
    return BertForSequenceClassification(cfg)


TC = {"max_length": 32, "batch_size": 8, "learning_rate": 5e-3, "epochs": 12, "warmup_ratio": 0.0,
      "weight_decay": 0.0, "max_grad_norm": 1.0, "length_bucketing": True, "class_weight": "none",
      "log_every": 0}


def toy_data(n_per_class=12):
    words = {1: "ужасно грязно", 2: "плохо долго", 3: "нормально средне", 4: "хорошо быстро", 5: "отлично супер"}
    texts, ratings = [], []
    for r, w in words.items():
        for i in range(n_per_class):
            texts.append(f"{w} {'и' * (i % 3)}")
            ratings.append(r)
    return texts, np.array(ratings)


def test_label_mapping_roundtrip():
    r = np.array([1, 2, 3, 4, 5])
    assert list(to_labels(r)) == [0, 1, 2, 3, 4]
    assert list(to_ratings(to_labels(r))) == [1, 2, 3, 4, 5]


def test_balanced_weights_match_sklearn():
    from sklearn.utils.class_weight import compute_class_weight

    y = np.array([0] * 50 + [1] * 10 + [2] * 5 + [3] * 20 + [4] * 15)
    expected = compute_class_weight("balanced", classes=np.arange(5), y=y)
    assert balanced_weights(y) == pytest.approx(expected)


def test_make_batches_cover_each_index_once():
    rng = np.random.default_rng(0)
    lengths = rng.integers(3, 100, size=203)
    for bucketing in (True, False):
        batches = make_batches(lengths, 16, np.random.default_rng(1), bucketing)
        flat = np.concatenate(batches)
        assert sorted(flat) == list(range(203))
        assert all(len(b) <= 16 for b in batches)


def test_bucketing_reduces_padding():
    rng = np.random.default_rng(0)
    lengths = rng.integers(3, 256, size=3200)

    def padded_tokens(batches):
        return sum(lengths[b].max() * len(b) for b in batches)

    plain = make_batches(lengths, 32, np.random.default_rng(1), bucketing=False)
    bucket = make_batches(lengths, 32, np.random.default_rng(1), bucketing=True)
    assert padded_tokens(bucket) < 0.7 * padded_tokens(plain)


def test_inference_batches_sorted_and_complete():
    lengths = [5, 50, 7, 30, 9]
    batches = inference_batches(lengths, 2)
    flat = np.concatenate(batches)
    assert sorted(flat) == list(range(5))
    assert [lengths[i] for i in flat] == sorted(lengths)


def test_collate_pads_and_masks():
    ids, mask = collate([[2, 7, 3], [2, 3]], pad_id=0)
    assert ids.tolist() == [[2, 7, 3], [2, 3, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]


def test_encode_truncates():
    out = encode(FakeTokenizer(), ["а б в г д е ж з"], max_length=4)
    assert len(out[0]) == 4


def test_subsample_is_deterministic_and_same_as_baseline_logic():
    df = pd.DataFrame({"text": list("abcdefghij"), "rating": [5] * 10})
    a, b = subsample(df, 4, 42), subsample(df, 4, 42)
    assert a.index.tolist() == b.index.tolist() == df.sample(4, random_state=42).index.tolist()
    assert len(subsample(df, "full", 42)) == 10
    assert len(subsample(df, 100, 42)) == 10


def _cross_entropy(probs, labels):
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-9).mean())


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_training_reduces_loss(seed):
    """Случайная крошечная BERT обучается нестабильно (на части seed'ов застревает на ~60% точности),
    поэтому проверяем не точность, а то, что обучение уверенно снижает кросс-энтропию относительно
    начальной ln(5) ≈ 1.61 (на всех проверенных seed'ах итог 0.5-1.0)."""
    texts, ratings = toy_data()
    y = to_labels(ratings)
    torch.manual_seed(seed)
    tok, model = FakeTokenizer(), tiny_model()
    before = _cross_entropy(predict_proba(model, tok, texts, TC["max_length"]), y)
    t = train_model(model, tok, texts, y, TC, seed=seed)
    probs = predict_proba(model, tok, texts, TC["max_length"], batch_size=7)
    after = _cross_entropy(probs, y)
    assert t > 0
    assert probs.shape == (len(texts), 5)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert set(probs.argmax(axis=1) + 1) <= {1, 2, 3, 4, 5}
    assert after < 0.8 * before


def test_predict_independent_of_batch_size_and_order():
    texts, _ = toy_data(6)
    tok, model = FakeTokenizer(), tiny_model()
    p1 = predict_proba(model, tok, texts, 32, batch_size=1)
    p2 = predict_proba(model, tok, texts, 32, batch_size=16)
    assert np.allclose(p1, p2, atol=1e-4)


def test_training_is_reproducible_for_same_seed():
    texts, ratings = toy_data(6)
    tc = {**TC, "epochs": 2}

    def run():
        torch.manual_seed(123)  # одинаковая инициализация модели; seed обучения задаёт train_model
        m = tiny_model()
        train_model(m, FakeTokenizer(), texts, to_labels(ratings), tc, seed=5)
        return predict_proba(m, FakeTokenizer(), texts, 32)

    assert np.allclose(run(), run(), atol=1e-5)


def test_balanced_class_weight_runs():
    texts, ratings = toy_data(6)
    tc = {**TC, "epochs": 1, "class_weight": "balanced"}
    train_model(tiny_model(), FakeTokenizer(), texts, to_labels(ratings), tc, seed=0)


def test_invalid_class_weight_rejected():
    texts, ratings = toy_data(3)
    with pytest.raises(ValueError):
        train_model(tiny_model(), FakeTokenizer(), texts, to_labels(ratings),
                    {**TC, "class_weight": "TODO"}, seed=0)


def test_test_split_still_protected():
    df = pd.DataFrame({"text": ["a"] * 3, "rating": [5] * 3, "split": ["train", "val", "test"]})
    with pytest.raises(PermissionError):
        select_eval(df, "test")


def test_run_one_logs_rows_and_saves_artifacts(tmp_path, monkeypatch):
    import src.models.transformer as T

    class Tok(FakeTokenizer):
        def save_pretrained(self, path):
            (tmp_path / "tok_saved").write_text("x")

    monkeypatch.setattr(T, "load_model_and_tokenizer", lambda name: (tiny_model(), Tok()))
    texts, ratings = toy_data(6)
    df = pd.DataFrame({"row_id": range(len(texts)), "text": texts, "rating": ratings})
    csv = tmp_path / "experiments.csv"
    cfg = {"paths": {"experiments_csv": str(csv)}, "split": {"version": "v1"}}
    tr = {"model_name": "x/tiny", "eval": {"batch_size": 8}}
    tc = {**TC, "epochs": 2, "eval_each_epoch": True}
    r = T.run_one(df, {"val": df}, cfg, tr, tc, seed=0, notes="unit", pred_dir=tmp_path / "pred",
                  save_dir=tmp_path / "model")
    log = pd.read_csv(csv)
    assert len(log) == 2                                   # промежуточная эпоха + итог
    assert set(log["phase"]) == {4} and set(log["eval_split"]) == {"val"}
    assert (tmp_path / "model" / "config.json").exists()
    pred = pd.read_parquet(tmp_path / "pred" / "seed0_val.parquet")
    assert len(pred) == len(df) and {"p1", "p5", "y_pred"} <= set(pred.columns)
    assert r["splits"]["val"]["metrics"]["inference_time"] >= 0

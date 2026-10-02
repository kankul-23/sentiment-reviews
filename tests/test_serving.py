import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.inference import NUM_LABELS, apply_temperature, build_prediction
from src.serving.schemas import MAX_BATCH, MAX_TEXT_CHARS
from src.serving.settings import load_settings


class FakeModel:
    """Заглушка: «плохо» -> класс 1, «отлично» -> класс 5, иначе 3."""

    def info(self):
        return {"model_name": "fake", "temperature": 1.0, "max_length": 256, "device": "cpu"}

    def predict(self, texts):
        out = []
        for t in texts:
            cls = 0 if "плохо" in t else 4 if "отлично" in t else 2
            p = np.full(NUM_LABELS, 0.05)
            p[cls] = 0.8
            out.append(build_prediction(p, truncated=len(t) > 1000))
        return out


@pytest.fixture()
def client():
    with TestClient(create_app(model_loader=FakeModel)) as c:
        yield c


# ------------------------------------------------------------------------------ API
def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/info").json()
    assert info["model_name"] == "fake" and "version" in info


def test_predict_response_shape(client):
    r = client.post("/predict", json={"text": "Всё отлично"})
    assert r.status_code == 200
    body = r.json()
    assert body["rating"] == 5 and body["truncated"] is False
    assert set(body["probabilities"]) == {"1", "2", "3", "4", "5"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert 1 <= body["expected_rating"] <= 5


def test_batch_keeps_order(client):
    texts = ["плохо", "отлично", "нормально", "плохо"]
    r = client.post("/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    assert [p["rating"] for p in r.json()["predictions"]] == [1, 5, 3, 1]


def test_truncated_flag_is_passed_through(client):
    assert client.post("/predict", json={"text": "а" * 1500}).json()["truncated"] is True


@pytest.mark.parametrize("payload", [
    {}, {"text": ""}, {"text": "   \n\t"}, {"text": 123}, {"text": "а" * (MAX_TEXT_CHARS + 1)}, {"txt": "x"},
])
def test_predict_rejects_bad_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("payload", [
    {"texts": []}, {"texts": ["ok", " "]}, {"texts": ["x"] * (MAX_BATCH + 1)}, {"texts": "строка"}, {},
])
def test_batch_rejects_bad_input(client, payload):
    assert client.post("/predict/batch", json=payload).status_code == 422


def test_batch_accepts_max_size(client):
    r = client.post("/predict/batch", json={"texts": ["плохо"] * MAX_BATCH})
    assert r.status_code == 200 and len(r.json()["predictions"]) == MAX_BATCH


# ------------------------------------------------------------- вспомогательные функции
def test_build_prediction():
    p = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    out = build_prediction(p, truncated=False)
    assert out["rating"] == 5 and out["confidence"] == 0.5
    assert abs(out["expected_rating"] - 3.9) < 1e-6


def test_apply_temperature_matches_calibration_module():
    from src.evaluation.calibration import apply_temperature as reference

    p = np.random.default_rng(0).dirichlet(np.ones(5), 200)
    for T in (0.8, 1.0, 1.35):
        assert np.allclose(apply_temperature(p, T), reference(p, T), atol=1e-12)
    assert (apply_temperature(p, 1.35).argmax(1) == p.argmax(1)).all()


def test_load_settings_from_config_files(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("model_name: m\nnum_threads: null\noutput: {models_dir: models/x}\ntrain: {max_length: 128}\n",
                   encoding="utf-8")
    cal = tmp_path / "c.json"
    cal.write_text(json.dumps({"transformer_seed": 7, "temperature": {"transformer": 1.5}}), encoding="utf-8")
    s = load_settings({"CONFIG_PATH": str(cfg), "CALIBRATION_PATH": str(cal)})
    assert s.max_length == 128 and s.temperature == 1.5 and s.calibration_seed == 7
    assert s.model_dir.as_posix() == "models/x/seed_7" and s.num_threads is None
    s2 = load_settings({"CONFIG_PATH": str(cfg), "CALIBRATION_PATH": str(cal), "MODEL_DIR": "/m",
                        "MAX_LENGTH": "64", "TORCH_THREADS": "2"})
    assert s2.model_dir.as_posix() == "/m" and s2.max_length == 64 and s2.num_threads == 2


def test_load_settings_rejects_nonpositive_temperature(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("model_name: m\noutput: {models_dir: m}\ntrain: {max_length: 8}\n", encoding="utf-8")
    cal = tmp_path / "c.json"
    cal.write_text(json.dumps({"transformer_seed": 1, "temperature": {"transformer": 0}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings({"CONFIG_PATH": str(cfg), "CALIBRATION_PATH": str(cal)})


# ------------------------------------------- настоящая (крошечная) модель, без скачивания
@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    path = tmp_path_factory.mktemp("tiny")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "хорошо", "плохо", "очень", "и", "!", "место"]
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = transformers.BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True)
    torch.manual_seed(0)
    cfg = transformers.BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
                                  num_labels=5)
    transformers.BertForSequenceClassification(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def _load_tiny(tiny_dir, max_length=16, temperature=1.0):
    from src.serving.inference import SentimentModel
    from src.serving.settings import Settings

    return SentimentModel.load(Settings(model_dir=tiny_dir, model_name="tiny", max_length=max_length,
                                        temperature=temperature, calibration_seed=0))


TEXTS = ["хорошо", "очень плохо и очень хорошо!", "место", "плохо " * 30, "хорошо очень"]


def test_real_model_probs_are_valid_and_batch_equals_single(tiny_dir):
    m = _load_tiny(tiny_dir)
    raw, _ = m.predict_proba_raw(TEXTS)
    assert raw.shape == (5, 5) and np.allclose(raw.sum(1), 1, atol=1e-5)
    for i, t in enumerate(TEXTS):
        single, _ = m.predict_proba_raw([t])
        assert np.allclose(single[0], raw[i], atol=1e-5)


def test_real_model_truncation_matches_tokenizer(tiny_dir):
    m = _load_tiny(tiny_dir, max_length=7)
    ids, flags = m.encode(TEXTS)
    ref = m.tokenizer(TEXTS, truncation=True, max_length=7)["input_ids"]
    assert ids == ref
    assert flags == [False, True, False, True, False]
    assert max(len(s) for s in ids) == 7


def test_real_model_temperature_keeps_rating(tiny_dir):
    a = _load_tiny(tiny_dir, temperature=1.0).predict(TEXTS)
    b = _load_tiny(tiny_dir, temperature=2.0).predict(TEXTS)
    assert [x["rating"] for x in a] == [x["rating"] for x in b]
    assert all(y["confidence"] <= x["confidence"] + 1e-6 for x, y in zip(a, b))


def test_real_model_matches_training_predict_proba(tiny_dir):
    """Сервис не должен расходиться с кодом, которым считались метрики."""
    train_mod = pytest.importorskip("src.models.transformer")
    m = _load_tiny(tiny_dir, max_length=16)
    ours, _ = m.predict_proba_raw(TEXTS)
    ref = train_mod.predict_proba(m.model, m.tokenizer, TEXTS, max_length=16, batch_size=2)
    assert np.allclose(ours, ref, atol=1e-5)


def test_real_model_end_to_end_via_api(tiny_dir):
    m = _load_tiny(tiny_dir)
    with TestClient(create_app(model_loader=lambda: m)) as c:
        r = c.post("/predict/batch", json={"texts": TEXTS})
        assert r.status_code == 200 and len(r.json()["predictions"]) == len(TEXTS)
        assert c.get("/info").json()["model_name"] == "tiny"

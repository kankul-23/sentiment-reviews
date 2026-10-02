"""Инференс rubert-tiny2 для сервиса: токенизация, батчи по длине, softmax, температурное масштабирование.

Не импортирует код обучения (pandas, sklearn): образу сервиса нужны только torch и transformers.
Совпадение с src.models.transformer.predict_proba проверяется тестом (tests/test_serving.py).
"""
import threading
from pathlib import Path

import numpy as np

NUM_LABELS = 5


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """softmax(log p / T). Та же формула, что в src/evaluation/calibration.py (проверяется тестом)."""
    z = np.log(np.clip(probs, 1e-12, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def build_prediction(probs_row: np.ndarray, truncated: bool) -> dict:
    p = np.asarray(probs_row, dtype=float)
    rating = int(p.argmax()) + 1
    return {
        "rating": rating,
        "confidence": round(float(p.max()), 4),
        "expected_rating": round(float((p * np.arange(1, NUM_LABELS + 1)).sum()), 3),
        "probabilities": {str(k + 1): round(float(v), 4) for k, v in enumerate(p)},
        "truncated": bool(truncated),
    }


class SentimentModel:
    def __init__(self, model, tokenizer, max_length: int, temperature: float, device="cpu",
                 batch_size: int = 32, meta: dict | None = None):
        import torch

        self._torch = torch
        self.model = model.to(torch.device(device)).eval()
        self.tokenizer = tokenizer
        # обрезку делаем сами (чтобы знать, была ли она), предупреждение о длине >512 не нужно
        self.tokenizer.model_max_length = int(1e9)
        self.max_length = int(max_length)
        self.temperature = float(temperature)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.meta = meta or {}
        self._lock = threading.Lock()  # один прямой проход за раз: иначе потоки спорят за ядра CPU

    @classmethod
    def load(cls, settings) -> "SentimentModel":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(settings.model_dir)
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"Нет модели в {path}. Ожидаются config.json, веса и файлы токенайзера.")
        if settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda, но CUDA недоступна")
        if settings.num_threads:
            torch.set_num_threads(settings.num_threads)
        tok = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        if model.config.num_labels != NUM_LABELS:
            raise ValueError(f"Ожидалось {NUM_LABELS} классов, в модели {model.config.num_labels}")
        meta = {"model_name": settings.model_name, "model_dir": str(path),
                "calibration_seed": settings.calibration_seed, "temperature": settings.temperature,
                "max_length": settings.max_length, "device": settings.device,
                "torch_threads": torch.get_num_threads()}
        return cls(model, tok, settings.max_length, settings.temperature, settings.device, meta=meta)

    def info(self) -> dict:
        return dict(self.meta)

    def encode(self, texts: list[str]) -> tuple[list[list[int]], list[bool]]:
        """Токены без паддинга; длиннее max_length обрезаются так же, как у токенайзера
        ([CLS] ... [SEP] ровно max_length). Второй результат: был ли отзыв обрезан."""
        full = self.tokenizer(list(texts), truncation=False)["input_ids"]
        sep = self.tokenizer.sep_token_id
        ids, truncated = [], []
        for s in full:
            cut = len(s) > self.max_length
            ids.append(s[: self.max_length - 1] + [sep] if cut else s)
            truncated.append(cut)
        return ids, truncated

    def predict_proba_raw(self, texts: list[str]) -> tuple[np.ndarray, list[bool]]:
        """Вероятности без калибровки (N, 5) в порядке входа и признаки обрезки."""
        torch = self._torch
        ids, truncated = self.encode(texts)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        order = np.argsort([len(s) for s in ids], kind="stable")
        probs = np.zeros((len(ids), NUM_LABELS), dtype=np.float32)
        with self._lock, torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                width = max(len(ids[i]) for i in idx)
                batch = torch.full((len(idx), width), pad, dtype=torch.long)
                mask = torch.zeros((len(idx), width), dtype=torch.long)
                for row, i in enumerate(idx):
                    batch[row, :len(ids[i])] = torch.tensor(ids[i], dtype=torch.long)
                    mask[row, :len(ids[i])] = 1
                logits = self.model(input_ids=batch.to(self.device), attention_mask=mask.to(self.device)).logits
                probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return probs, truncated

    def predict(self, texts: list[str]) -> list[dict]:
        raw, truncated = self.predict_proba_raw(texts)
        cal = apply_temperature(raw, self.temperature)
        return [build_prediction(cal[i], truncated[i]) for i in range(len(texts))]

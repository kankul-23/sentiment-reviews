"""Сквозная проверка: сервис отдаёт то же, что считалось при оценке модели.

    python -m src.serving.check_parity --url http://localhost:8001

Берёт случайные отзывы из val, получает предсказания от сервиса и сравнивает с сохранёнными
data/predictions/seed{S}_val.parquet (с применённой температурой из /info). Расхождения возможны
из-за точности вычислений (обучение шло на GPU со смешанной точностью, сервис считает на CPU в fp32),
поэтому проверяются доля совпавших оценок и медианная разница вероятностей, а не побитовое равенство.
Только val; test не используется. Нужны httpx и pandas (в образ не входят).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROB_COLS = [f"p{c}" for c in range(1, 6)]


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--min-agreement", type=float, default=0.97)
    ap.add_argument("--max-median-diff", type=float, default=0.01)
    args = ap.parse_args()

    import httpx
    import pandas as pd

    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        r = client.get("/info")
        if r.status_code != 200:
            raise SystemExit(f"/info вернул HTTP {r.status_code}: {r.text[:300]!r}. "
                             f"Это сервис из этого репозитория? Проверьте URL и логи контейнера.")
        info = r.json()
        seed, T = int(info["calibration_seed"]), float(info["temperature"])
        pred = pd.read_parquet(Path(args.pred_dir) / f"seed{seed}_val.parquet")
        clean = pd.read_parquet("data/processed/reviews_clean.parquet", columns=["row_id", "text"])
        df = pred.merge(clean, on="row_id").sample(min(args.n, len(pred)), random_state=42).reset_index(drop=True)

        served = []
        texts = df["text"].str.slice(0, 20000).tolist()
        for i in range(0, len(texts), 32):
            r = client.post("/predict/batch", json={"texts": texts[i:i + 32]})
            r.raise_for_status()
            served.extend(r.json()["predictions"])

    served_p = np.array([[p["probabilities"][str(c)] for c in range(1, 6)] for p in served])
    served_rating = np.array([p["rating"] for p in served])
    ref_p = apply_temperature(df[PROB_COLS].to_numpy(), T)
    ref_rating = ref_p.argmax(axis=1) + 1

    agree = float((served_rating == ref_rating).mean())
    diff = np.abs(served_p - ref_p).max(axis=1)
    print(f"Сервис: {args.url}, seed {seed}, T = {T}, отзывов: {len(df)}")
    print(f"Совпадение оценок: {agree:.4f} ({int((served_rating != ref_rating).sum())} расхождений)")
    print(f"Макс. разница вероятностей: медиана {np.median(diff):.5f}, p95 {np.percentile(diff, 95):.5f}, "
          f"максимум {diff.max():.5f}")
    bad = df.loc[served_rating != ref_rating, "row_id"].tolist()[:10]
    if bad:
        print(f"row_id с расхождением (до 10): {bad}")
    ok = agree >= args.min_agreement and float(np.median(diff)) <= args.max_median_diff
    print("ПРОВЕРКА ПРОЙДЕНА" if ok else "ПРОВЕРКА НЕ ПРОЙДЕНА")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

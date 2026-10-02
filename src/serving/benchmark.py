"""Замер работающего сервиса по HTTP (CPU): задержка одиночных запросов и пропускная способность батчей.

    python -m uvicorn src.serving.app:app --port 8000        # или docker run -p 8000:8000 sentiment-reviews
    python -m src.serving.benchmark --url http://localhost:8000

Тексты берутся из val (не из test). Нужны httpx и pandas (есть в окружении разработки, в образ не входят).
Результат: reports/tables/serving_benchmark.json
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n-single", type=int, default=300)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--batches-per-size", type=int, default=10)
    ap.add_argument("--out", default="reports/tables/serving_benchmark.json")
    args = ap.parse_args()

    import httpx
    import pandas as pd

    clean = pd.read_parquet("data/processed/reviews_clean.parquet", columns=["row_id", "text"])
    split = pd.read_parquet("data/splits/split_v1.parquet")
    val = clean.merge(split[split["split"] == "val"], on="row_id")
    texts = val.sample(min(len(val), args.n_single + 2000), random_state=42)["text"].str.slice(0, 20000).tolist()

    with httpx.Client(base_url=args.url, timeout=60, trust_env=False) as client:
        r = client.get("/info")
        if r.status_code != 200:
            raise SystemExit(f"/info вернул HTTP {r.status_code}: {r.text[:300]!r}. "
                             f"Это сервис из этого репозитория? Проверьте URL и логи контейнера.")
        info = r.json()
        for t in texts[:10]:                                    # прогрев
            client.post("/predict", json={"text": t}).raise_for_status()

        lat = []
        for t in texts[:args.n_single]:
            t0 = time.perf_counter()
            client.post("/predict", json={"text": t}).raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)

        batches = {}
        pool = texts[args.n_single:]
        for bs in args.batch_sizes:
            times = []
            for k in range(args.batches_per_size):
                chunk = pool[k * bs:(k + 1) * bs] or pool[:bs]
                t0 = time.perf_counter()
                client.post("/predict/batch", json={"texts": chunk}).raise_for_status()
                times.append(time.perf_counter() - t0)
            batches[str(bs)] = {"latency_s_median": float(np.median(times)),
                                "texts_per_s": float(bs / np.median(times))}

    res = {"url": args.url, "model_info": info, "n_single": len(lat),
           "single_ms_p50": float(np.percentile(lat, 50)), "single_ms_p95": float(np.percentile(lat, 95)),
           "batch": batches, "client_cpu": platform.processor() or platform.machine(),
           "note": "задержка включает HTTP, токенизацию, инференс и калибровку; клиент и сервер на одной машине"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "model_info"}, ensure_ascii=False, indent=2))
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()

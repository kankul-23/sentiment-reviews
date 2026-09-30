"""
Фаза 2: smoke test. TF-IDF + LogReg на подвыборке train, метрика на val.

    python -m src.experiments.smoke_test                 # честный group split
    python -m src.experiments.smoke_test --leaky         # случайный split (для оценки утечки)

Цель: убедиться, что пайплайн работает от данных до метрики и пишет строку в
reports/experiments.csv. Запуск с --leaky нужен только чтобы измерить, насколько
метрика завышается без группового split; строка помечается split_version=random-leaky.
"""
import argparse
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error

from src.data.loader import load_config
from src.data.splitting import load_split_data
from src.experiments.tracking import log_experiment, set_seed


def random_split(n: int, seed: int, fractions=(0.70, 0.15, 0.15)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr, n_va = int(n * fractions[0]), int(n * fractions[1])
    part = np.empty(n, dtype=object)
    part[perm[:n_tr]] = "train"
    part[perm[n_tr:n_tr + n_va]] = "val"
    part[perm[n_tr + n_va:]] = "test"
    return part


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--leaky", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    scfg = cfg["split"]
    seed = scfg["seed"]
    set_seed(seed)

    df = load_split_data()
    if args.leaky:
        df["split"] = random_split(len(df), seed, (scfg["train"], scfg["val"], scfg["test"]))

    train = df[df["split"] == "train"].sample(min(args.n_train, (df["split"] == "train").sum()),
                                              random_state=seed)
    val = df[df["split"] == "val"]  # test не используем

    tfidf_params = {"analyzer": "word", "ngram_range": [1, 2], "min_df": 3,
                    "sublinear_tf": True, "max_features": 200000}
    model_params = {"C": 1.0, "max_iter": 1000, "class_weight": None}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=200000)

    t0 = time.time()
    x_train = vec.fit_transform(train["text"])
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(x_train, train["rating"])
    train_time = time.time() - t0

    t0 = time.time()
    pred = clf.predict(vec.transform(val["text"]))
    infer_time = time.time() - t0

    y = val["rating"].to_numpy()
    macro_f1 = f1_score(y, pred, average="macro")
    mae = mean_absolute_error(y, pred)
    acc = accuracy_score(y, pred)
    print(classification_report(y, pred, digits=3, zero_division=0))
    print(f"macro-F1 {macro_f1:.4f} | MAE {mae:.4f} | accuracy {acc:.4f}")

    exp_id = log_experiment({
        "phase": 2,
        "model": "tfidf-word12+logreg (smoke)",
        "dataset_size": len(train),
        "split_version": "random-leaky" if args.leaky else scfg["version"],
        "seed": seed,
        "tfidf_params": tfidf_params,
        "model_params": model_params,
        "eval_split": "val",
        "macro_f1": round(macro_f1, 4),
        "mae": round(mae, 4),
        "accuracy": round(acc, 4),
        "train_time": round(train_time, 2),
        "inference_time": round(infer_time, 2),
        "notes": "LEAKY: случайный split, только для оценки утечки" if args.leaky else "smoke test",
    }, path=cfg["paths"]["experiments_csv"])
    print("experiment_id:", exp_id)


if __name__ == "__main__":
    main()

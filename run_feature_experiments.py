"""Compare feature configs on Validation (+ grouped CV on train). The Test set is NOT touched.

Usage:
    python run_feature_experiments.py [--data-dir data] [--reports-dir reports]

Needs data/train.csv and data/val.csv from the Phase 1 group-aware split
(columns: subject, body, queue, group_id).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from triageai.mlflow_utils import RANDOM_STATE, log_experiment, per_class_metrics, setup_mlflow
from triageai.model import FEATURE_CONFIGS, build_pipeline
from triageai.preprocessing import EMAIL_RE, URL_RE

REQUIRED_COLS = {"subject", "body", "queue", "group_id"}
CLF_PARAMS = dict(C=10, class_weight="balanced")


def load_splits(data_dir):
    train_path, val_path = data_dir / "train.csv", data_dir / "val.csv"
    tr, va = pd.read_csv(train_path), pd.read_csv(val_path)
    for name, d in (("train.csv", tr), ("val.csv", va)):
        missing = REQUIRED_COLS - set(d.columns)
        if missing:
            raise SystemExit(f"{name} is missing columns {sorted(missing)}. "
                             "Use the group-aware split files from Phase 1.")
    shared = set(tr["group_id"]) & set(va["group_id"])
    if shared:
        raise SystemExit(f"{len(shared)} duplicate groups appear in BOTH train and val (leakage). "
                         "Regenerate the split before running experiments.")
    return tr, va, [train_path, val_path]


def get_labels(data_dir, y):
    f = data_dir / "label_classes.json"
    return json.loads(f.read_text()) if f.exists() else sorted(y.unique())


def cleaning_stats(tr):
    text = tr["subject"].fillna("") + " " + tr["body"].fillna("")
    n = len(text)
    row = {"tickets": n,
           "with_url": int(text.str.contains(URL_RE).sum()),
           "with_email_address": int(text.str.contains(EMAIL_RE).sum())}
    return pd.DataFrame([row])


def diagnose_best(name, yva, pred, labels, reports_dir):
    """Per-class table, row-normalised confusion matrix and top confusions for the best config."""
    p, r, f, s = precision_recall_fscore_support(yva, pred, labels=labels, zero_division=0)
    pd.DataFrame({"queue": labels, "precision": p, "recall": r, "f1": f, "support": s}) \
        .round(4).to_csv(reports_dir / "per_class_best.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(yva, pred, labels=labels, normalize="true",
                                            xticks_rotation=45, values_format=".2f",
                                            ax=ax, colorbar=False)
    ax.set_title(f"Validation confusion matrix (row-normalised) - {name}")
    plt.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix_best.png", dpi=150)
    plt.close(fig)

    cm = pd.DataFrame(confusion_matrix(yva, pred, labels=labels), index=labels, columns=labels)
    pairs = [(a, b, int(cm.loc[a, b])) for a in labels for b in labels if a != b and cm.loc[a, b] > 0]
    pd.DataFrame(sorted(pairs, key=lambda t: -t[2])[:10],
                 columns=["true_queue", "predicted_as", "tickets"]) \
        .to_csv(reports_dir / "top_confusions_best.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()
    data_dir, reports_dir = Path(args.data_dir), Path(args.reports_dir)
    reports_dir.mkdir(exist_ok=True)

    tr, va, split_files = load_splits(data_dir)
    cols = ["subject", "body"]
    Xtr, ytr, Xva, yva = tr[cols], tr["queue"], va[cols], va["queue"]
    labels = get_labels(data_dir, ytr)
    cleaning_stats(tr).to_csv(reports_dir / "cleaning_stats.csv", index=False)
    setup_mlflow()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, preds = [], {}
    for name, cfg in FEATURE_CONFIGS.items():
        t0 = time.time()
        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **CLF_PARAMS)
        pipe = build_pipeline(clf, **cfg)
        scores = cross_val_score(pipe, Xtr, ytr, groups=tr["group_id"], cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xva)
        preds[name] = pred
        m = dict(val_macro_f1=f1_score(yva, pred, average="macro"),
                 weighted_f1=f1_score(yva, pred, average="weighted"),
                 accuracy=accuracy_score(yva, pred),
                 cv_macro_f1_mean=float(scores.mean()), cv_macro_f1_std=float(scores.std()))
        log_experiment(name, "LogisticRegression", params=CLF_PARAMS,
                       metrics={**m, **per_class_metrics(yva, pred, labels)},
                       pipeline_config={k: str(v) for k, v in cfg.items()},
                       split_files=split_files,
                       cv_info=dict(splitter="StratifiedGroupKFold", n_folds=5, groups="group_id"),
                       tags={"phase": "2-feature-comparison"})
        rows.append(dict(config=name, **{k: round(float(v), 4) for k, v in m.items()},
                         sec=round(time.time() - t0)))
        print(rows[-1], flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(reports_dir / "feature_comparison.csv", index=False)
    best = result.loc[result["val_macro_f1"].idxmax(), "config"]
    diagnose_best(best, yva, preds[best], labels, reports_dir)
    print(f"\nBest on Validation: {best}. Reports written to {reports_dir}/")


if __name__ == "__main__":
    main()

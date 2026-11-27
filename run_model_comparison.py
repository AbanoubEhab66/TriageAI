"""Phase 3 - Model development: compare classifiers on grouped CV (train) and Validation.

The Test set is NOT read anywhere in this script.

Usage:
    python run_model_comparison.py [--data-dir data] [--reports-dir reports]
                                   [--artifacts-dir artifacts] [--grid small|large]
                                   [--n-jobs 2] [--quick]

Needs data/train.csv and data/val.csv from the Phase 1 group-aware split
(columns: subject, body, queue, group_id) and data/label_classes.json.

Protocol (fixed before running, so Validation is not used to tune anything inside a family):
  1. Inside each model family, hyper-parameters / n-gram setting are chosen by 5-fold
     StratifiedGroupKFold CV macro-F1 on TRAIN only.
  2. The families are then compared with each other on VALIDATION (macro-F1 first).
  3. The best config of every family is saved as a candidate (uncalibrated pipeline, joblib).
     Calibration + threshold are Phase 4, they are not done here.
"""
import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from triageai.mlflow_utils import RANDOM_STATE, log_experiment, per_class_metrics, setup_mlflow
from triageai.model import build_pipeline, save_pipeline
from triageai.validation import is_valid_input

REQUIRED_COLS = {"subject", "body", "queue", "group_id"}
N_FOLDS = 5

# Only the n-gram range is compared (same stop-word setting for both), as the plan asks:
# Unigrams vs Unigrams+Bigrams. stop_words="english" = RECOMMENDED_CONFIG from Phase 2.
FEATURES = {
    "uni":    dict(ngram_range=(1, 1), stop_words="english"),
    "uni_bi": dict(ngram_range=(1, 2), stop_words="english"),
}

GRIDS = {
    "small": {
        "NaiveBayes":     [dict(alpha=a) for a in (0.01, 0.1, 1.0)],
        "LogReg":         [dict(C=c) for c in (1, 10, 100)],
        "LogReg_balanced": [dict(C=c) for c in (1, 10, 100)],
        "LinearSVC":      [dict(C=c) for c in (0.1, 0.5, 1.0)],
    },
    "large": {
        "NaiveBayes":     [dict(alpha=a) for a in (0.001, 0.003, 0.01, 0.03, 0.1)],
        "LogReg":         [dict(C=c) for c in (10, 30, 100, 300, 1000)],
        "LogReg_balanced": [dict(C=c) for c in (10, 30, 100, 300, 1000)],
        "LinearSVC":      [dict(C=c) for c in (0.5, 1.0, 3.0, 10.0, 30.0)],
    },
}

PROBA_SUPPORT = {
    "Dummy":           ("predict_proba (class priors only)", "no calibration needed - baseline"),
    "NaiveBayes":      ("predict_proba", "has probabilities but they are usually over-confident; check ECE, calibrate"),
    "LogReg":          ("predict_proba", "usually the best-calibrated raw model; still check ECE"),
    "LogReg_balanced": ("predict_proba", "class_weight shifts probabilities toward small classes; check ECE, calibrate"),
    "LinearSVC":       ("decision_function only", "NO predict_proba: must go through CalibratedClassifierCV"),
}


def make_clf(family, params):
    if family == "Dummy":
        return DummyClassifier(strategy="most_frequent")
    if family == "NaiveBayes":
        return MultinomialNB(**params)
    if family == "LogReg":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **params)
    if family == "LogReg_balanced":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                  class_weight="balanced", **params)
    if family == "LinearSVC":
        return LinearSVC(max_iter=5000, random_state=RANDOM_STATE, **params)
    raise ValueError(family)


def load_splits(data_dir):
    train_path, val_path = data_dir / "train.csv", data_dir / "val.csv"
    tr, va = pd.read_csv(train_path), pd.read_csv(val_path)
    for name, d in (("train.csv", tr), ("val.csv", va)):
        missing = REQUIRED_COLS - set(d.columns)
        if missing:
            raise SystemExit(f"{name} is missing columns {sorted(missing)}.")
        bad = ~np.array([is_valid_input(s, b) for s, b in zip(d["subject"], d["body"])])
        if bad.any():
            raise SystemExit(f"{name}: {int(bad.sum())} tickets have no usable text.")
    shared = set(tr["group_id"]) & set(va["group_id"])
    if shared:
        raise SystemExit(f"{len(shared)} duplicate groups appear in BOTH train and val (leakage).")
    return tr, va, [train_path, val_path]


def get_labels(data_dir, y):
    f = data_dir / "label_classes.json"
    return json.loads(f.read_text()) if f.exists() else sorted(y.unique())


def scores_of(pipe, X):
    """(scores, kind): probabilities if the model has them, else decision_function."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X), "proba"
    return pipe.decision_function(X), "decision"


def top2_accuracy(scores, classes, y_true):
    top2 = np.argsort(-scores, axis=1)[:, :2]
    classes = np.asarray(classes)
    return float(np.mean([t in classes[row] for t, row in zip(y_true, top2)]))


def ece(proba, classes, y_true, n_bins=15):
    """Expected Calibration Error of the top-label confidence (equal-width bins).
    Only a quick diagnostic for choosing candidates; Phase 4 owns the official ECE."""
    conf = proba.max(axis=1)
    correct = (np.asarray(classes)[proba.argmax(axis=1)] == np.asarray(y_true)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def run_name(family, feat, params):
    p = "_".join(f"{k}{v}" for k, v in params.items())
    return f"{family}__{feat}" + (f"__{p}" if p else "")


def diagnose(name, yva, pred, labels, reports_dir):
    """Row-normalised confusion matrix + top confusions for one model."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(yva, pred, labels=labels, normalize="true",
                                            xticks_rotation=45, values_format=".2f",
                                            ax=ax, colorbar=False)
    ax.set_title(f"Validation confusion matrix (row-normalised) - {name}")
    plt.tight_layout()
    png = reports_dir / f"confusion_matrix_{name}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    cm = pd.DataFrame(confusion_matrix(yva, pred, labels=labels), index=labels, columns=labels)
    pairs = [(a, b, int(cm.loc[a, b])) for a in labels for b in labels if a != b and cm.loc[a, b] > 0]
    top = pd.DataFrame(sorted(pairs, key=lambda t: -t[2])[:10],
                       columns=["true_queue", "predicted_as", "tickets"])
    top.to_csv(reports_dir / f"top_confusions_{name}.csv", index=False)
    return png, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--grid", choices=list(GRIDS), default="small")
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel CV folds (-1 = all cores)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test on a small sample of train (results are NOT meaningful)")
    args = ap.parse_args()
    data_dir, reports_dir = Path(args.data_dir), Path(args.reports_dir)
    artifacts_dir = Path(args.artifacts_dir)
    cand_dir = artifacts_dir / "candidates"
    reports_dir.mkdir(exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    tr, va, split_files = load_splits(data_dir)
    if args.quick:
        tr = tr.sample(frac=1, random_state=RANDOM_STATE).groupby("queue").head(150).reset_index(drop=True)
        print(f"[quick] train reduced to {len(tr)} rows - smoke test only", flush=True)
    cols = ["subject", "body"]
    Xtr, ytr, Xva, yva = tr[cols], tr["queue"], va[cols], va["queue"]
    labels = get_labels(data_dir, ytr)
    n_folds = 3 if args.quick else N_FOLDS

    # Same folds for every experiment, computed once; check no group sits in both sides (test #18).
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(Xtr, ytr, tr["group_id"]))
    g = tr["group_id"].to_numpy()
    for tri, vai in folds:
        assert set(g[tri]).isdisjoint(g[vai]), "group leaked between CV folds"
    cv_info = dict(splitter="StratifiedGroupKFold", n_folds=n_folds, groups="group_id")
    setup_mlflow()

    # experiment list: Dummy baseline + 4 families x features x grid
    grid = GRIDS[args.grid]
    experiments = [("Dummy", "uni", {})]
    for family, plist in grid.items():
        for feat, params in itertools.product(FEATURES, plist):
            experiments.append((family, feat, params))
    print(f"{len(experiments)} experiments, {n_folds}-fold grouped CV each", flush=True)

    rows, fitted = [], {}
    out_csv = reports_dir / "model_comparison.csv"
    for i, (family, feat, params) in enumerate(experiments, 1):
        name = run_name(family, feat, params)
        t0 = time.time()
        pipe = build_pipeline(make_clf(family, params), **FEATURES[feat])
        cv_scores = cross_val_score(pipe, Xtr, ytr, cv=folds, scoring="f1_macro", n_jobs=args.n_jobs)
        pipe.fit(Xtr, ytr)
        assert list(pipe.classes_) == list(labels), "class order differs from label_classes.json"
        pred = pipe.predict(Xva)
        sc, kind = scores_of(pipe, Xva)
        m = dict(val_macro_f1=f1_score(yva, pred, average="macro"),
                 weighted_f1=f1_score(yva, pred, average="weighted"),
                 accuracy=accuracy_score(yva, pred),
                 top2_accuracy=top2_accuracy(sc, pipe.classes_, yva),
                 cv_macro_f1_mean=float(cv_scores.mean()), cv_macro_f1_std=float(cv_scores.std()))
        if kind == "proba":
            m["val_ece_15bin_uncalibrated"] = ece(sc, pipe.classes_, yva)
            m["val_mean_confidence"] = float(sc.max(axis=1).mean())
        log_experiment(name, family, params=params,
                       metrics={**m, **per_class_metrics(yva, pred, labels)},
                       pipeline_config={k: str(v) for k, v in FEATURES[feat].items()},
                       split_files=split_files, cv_info=cv_info, calibration_method="none",
                       tags={"phase": "3-model-comparison", "family": family, "features": feat,
                             "probability_support": PROBA_SUPPORT[family][0]})
        row = dict(run=name, family=family, features=feat,
                   params=json.dumps(params), **{k: round(float(v), 4) for k, v in m.items()},
                   sec=round(time.time() - t0))
        rows.append(row)
        fitted[name] = (pipe, pred, sc, kind)
        pd.DataFrame(rows).to_csv(out_csv, index=False)          # saved after every run
        print(f"[{i}/{len(experiments)}] {name}: cv={m['cv_macro_f1_mean']:.4f} "
              f"val={m['val_macro_f1']:.4f} ({row['sec']}s)", flush=True)

    res = pd.DataFrame(rows)

    # ---- 1) pick the best config of each family by grouped CV macro-F1 (train only) ----
    best_rows = res[res.family != "Dummy"].sort_values("cv_macro_f1_mean", ascending=False) \
                   .groupby("family", as_index=False).head(1)
    # ---- 2) compare the families on Validation ----
    best_rows = best_rows.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    best_rows.insert(0, "rank_val", range(1, len(best_rows) + 1))

    cands = []
    for _, r in best_rows.iterrows():
        pipe, pred, sc, kind = fitted[r["run"]]
        path = save_pipeline(pipe, cand_dir / f"{r['run']}.joblib",
                             metadata=dict(labels=labels, family=r["family"], features=r["features"],
                                           params=json.loads(r["params"]), calibrated=False,
                                           val_macro_f1=float(r["val_macro_f1"]),
                                           cv_macro_f1_mean=float(r["cv_macro_f1_mean"])))
        sup, note = PROBA_SUPPORT[r["family"]]
        cands.append(dict(rank_val=int(r["rank_val"]), run=r["run"], family=r["family"],
                          features=r["features"], params=json.loads(r["params"]),
                          val_macro_f1=float(r["val_macro_f1"]),
                          cv_macro_f1_mean=float(r["cv_macro_f1_mean"]),
                          top2_accuracy=float(r["top2_accuracy"]),
                          probability_support=sup, calibration_note=note,
                          file=str(path), calibrated=False))
    (artifacts_dir / "candidates.json").write_text(json.dumps(
        dict(labels=labels, note="Uncalibrated pipelines fitted on train only. "
             "Calibrate on train with group-aware CV in Phase 4.", candidates=cands), indent=2))

    best = best_rows.iloc[0]
    save_pipeline(fitted[best["run"]][0], artifacts_dir / "best_model.joblib",
                  metadata=dict(labels=labels, run=best["run"], calibrated=False,
                                params=json.loads(best["params"]), features=best["features"]))
    (artifacts_dir / "labels.json").write_text(json.dumps(labels, indent=2))

    # ---- reports ----
    best_rows.drop(columns=["sec"]).to_csv(reports_dir / "candidates.csv", index=False)
    pd.DataFrame([dict(model=k, probability_support=v[0], calibration=v[1])
                  for k, v in PROBA_SUPPORT.items()]).to_csv(reports_dir / "probability_support.csv", index=False)

    pc = []
    for _, r in best_rows.iterrows():
        pred = fitted[r["run"]][1]
        p, rc, f, s = precision_recall_fscore_support(yva, pred, labels=labels, zero_division=0)
        for lab, pi, ri, fi, si in zip(labels, p, rc, f, s):
            pc.append(dict(run=r["run"], queue=lab, precision=round(pi, 4), recall=round(ri, 4),
                           f1=round(fi, 4), support=int(si)))
    pc = pd.DataFrame(pc)
    pc.to_csv(reports_dir / "per_class_candidates.csv", index=False)
    pc.pivot(index="queue", columns="run", values="f1").to_csv(reports_dir / "per_class_f1_wide.csv")

    png, top = diagnose(best["run"], yva, fitted[best["run"]][1], labels, reports_dir)

    # validation scores of candidates that have probabilities (useful for Phase 4 right away)
    for _, r in best_rows.iterrows():
        pipe, pred, sc, kind = fitted[r["run"]]
        if kind != "proba":
            continue
        order = np.argsort(-sc, axis=1)[:, :2]
        cls = np.asarray(pipe.classes_)
        pd.DataFrame(dict(ticket_id=va["ticket_id"], true_queue=yva.values, pred=pred,
                          confidence=sc.max(axis=1).round(6),
                          top1=cls[order[:, 0]], top2=cls[order[:, 1]])
                     ).to_csv(reports_dir / f"val_scores_{r['run']}.csv", index=False)

    log_experiment("best_candidate_summary", best["family"], params=json.loads(best["params"]),
                   metrics={k: best[k] for k in ("val_macro_f1", "weighted_f1", "accuracy",
                                                 "top2_accuracy", "cv_macro_f1_mean", "cv_macro_f1_std")},
                   pipeline_config={k: str(v) for k, v in FEATURES[best["features"]].items()},
                   split_files=split_files, cv_info=cv_info, calibration_method="none",
                   tags={"phase": "3-model-comparison", "role": "best_on_validation"},
                   artifacts=[str(png), str(reports_dir / "candidates.csv")])

    print("\n=== Best config per family (chosen by CV, ranked on Validation) ===")
    print(best_rows[["rank_val", "run", "cv_macro_f1_mean", "val_macro_f1", "top2_accuracy"]]
          .to_string(index=False))
    print("\n=== Top confusions of the best model on Validation ===")
    print(top.to_string(index=False))
    print(f"\nSaved: {cand_dir}/ , {artifacts_dir}/best_model.joblib , {reports_dir}/")
    if args.quick:
        print("\n[quick] smoke test only - do not use these numbers.")


if __name__ == "__main__":
    main()

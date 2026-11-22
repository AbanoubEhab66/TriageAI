"""One logging format for the whole team. Everyone calls log_experiment()."""
import hashlib
import os
import re
import subprocess
from pathlib import Path

import mlflow
from sklearn.metrics import precision_recall_fscore_support

EXPERIMENT_NAME = "TriageAI"
RANDOM_STATE = 42
SPLIT_POLICY = "train/val/test 70/15/15, group-aware (StratifiedGroupKFold), rs=42"

# <repo>/src/triageai/mlflow_utils.py -> repo root, so the DB is the same wherever you run from
_DEFAULT_URI = "sqlite:///" + (Path(__file__).resolve().parents[2] / "mlflow.db").as_posix()


def setup_mlflow(tracking_uri=None, experiment=EXPERIMENT_NAME):
    """Tracking URI order: argument > MLFLOW_TRACKING_URI env var > <repo>/mlflow.db."""
    mlflow.set_tracking_uri(tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or _DEFAULT_URI)
    mlflow.set_experiment(experiment)


def dataset_fingerprint(*paths):
    """Short SHA-256 over the given files (in order). Same files -> same value."""
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:12]


def git_commit():
    """Short commit hash of the current repo, or None if git is not available."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None
    except Exception:
        return None


def metric_key(name):
    """MLflow keys allow letters, digits, _ - . space /  -> replace anything else with '_'."""
    return re.sub(r"[^0-9A-Za-z_\-. /]", "_", str(name))


def per_class_metrics(y_true, y_pred, labels):
    """Standard per-class names: precision__<queue>, recall__<queue>, f1__<queue>."""
    labels = list(labels)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {}
    for lab, pi, ri, fi in zip(labels, p, r, f):
        k = metric_key(lab)
        out[f"precision__{k}"], out[f"recall__{k}"], out[f"f1__{k}"] = float(pi), float(ri), float(fi)
    return out


def log_experiment(run_name, model_name, params=None, metrics=None, pipeline_config=None,
                   split_ref=None, split_files=None, cv_info=None, calibration_method=None,
                   tags=None, artifacts=None, model=None):
    """params: classifier hyperparameters
    pipeline_config: dict, e.g. {'ngram_range': '(1, 2)', 'stop_words': None, 'min_df': 2}
    split_files: the train/val(/test) CSV paths used -> a content fingerprint is logged as
        split_sha256, so you can tell exactly which split a run used (it changes if the split changes)
    split_ref: optional human label for the split (free text)
    calibration_method: 'none' | 'sigmoid' | 'isotonic'
    metrics: flat dict, standard names:
      macro_f1, weighted_f1, accuracy, val_macro_f1, top2_accuracy, cv_macro_f1_mean, cv_macro_f1_std,
      per class: precision__<queue>, recall__<queue>, f1__<queue>   (see per_class_metrics)
      calibration: raw_ece, calibrated_ece
      routing: threshold, coverage, routing_error_rate, routing_error_ci_low, routing_error_ci_high,
               human_review_rate, auto_routed_count, auto_routed_errors
    cv_info: {'splitter': 'StratifiedGroupKFold', 'n_folds': 5, 'groups': 'group_id'}
    artifacts: list of file paths (reliability diagram, confusion matrix, ...)
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("split_policy", SPLIT_POLICY)
        mlflow.log_param("split_reference", split_ref or "unspecified")
        if split_files:
            mlflow.log_param("split_sha256", dataset_fingerprint(*split_files))
        commit = git_commit()
        if commit:
            mlflow.log_param("git_commit", commit)
        if calibration_method:
            mlflow.log_param("calibration_method", calibration_method)
        for k, v in (params or {}).items():
            mlflow.log_param(f"clf__{k}", v)
        for k, v in (pipeline_config or {}).items():
            mlflow.log_param(f"feat__{k}", v)
        for k, v in (cv_info or {}).items():
            mlflow.log_param(f"cv__{k}", v)
        for k, v in (metrics or {}).items():
            mlflow.log_metric(metric_key(k), float(v))
        if tags:
            mlflow.set_tags(tags)
        for path in artifacts or []:
            mlflow.log_artifact(path)
        if model is not None:
            import mlflow.sklearn as mlflow_sklearn  # alias: do not shadow the module-level 'mlflow'
            mlflow_sklearn.log_model(model, "model")

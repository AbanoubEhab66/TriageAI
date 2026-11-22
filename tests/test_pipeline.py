import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from triageai.mlflow_utils import (dataset_fingerprint, log_experiment, metric_key,
                                   per_class_metrics, setup_mlflow)
from triageai.model import FEATURE_CONFIGS, build_pipeline, load_pipeline, save_pipeline


def data():
    rows = [("I was charged twice", "billing payment invoice refund"),
            ("Server down", "technical outage error crash"),
            ("Invoice wrong", "billing payment charged"),
            ("App crash", "technical error bug crash")] * 5
    df = pd.DataFrame(rows, columns=["subject", "body"])
    y = ["Billing", "Technical", "Billing", "Technical"] * 5
    return df, y


def fitted():
    X, y = data()
    p = build_pipeline(LogisticRegression(max_iter=500, random_state=42), min_df=1)
    return p.fit(X, y), X, y


def test_pipeline_fit_predict_proba():
    p, X, _ = fitted()
    assert p.predict_proba(X.head(2)).shape == (2, 2)
    p.predict(pd.DataFrame({"subject": [None], "body": ["billing invoice"]}))   # NaN subject
    p.predict(["billing invoice"])                                               # plain strings
    p.predict("billing invoice")                                                 # one string


def test_all_feature_configs_build():
    for cfg in FEATURE_CONFIGS.values():
        build_pipeline(LogisticRegression(), **cfg)


def test_vectorizer_is_fitted_on_train_only():
    p, _, _ = fitted()
    vocab_before = dict(p.named_steps["vectorizer"].vocabulary_)
    assert "zzuniquezz" not in vocab_before
    p.predict(pd.DataFrame({"subject": ["zzuniquezz"], "body": ["zzuniquezz refund"]}))
    assert p.named_steps["vectorizer"].vocabulary_ == vocab_before   # predict never changes the vocabulary


def test_group_cv_has_no_group_overlap():
    X, y = data()
    y, groups = np.array(y), np.arange(len(X)) % 10
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in cv.split(X, y, groups):
        assert set(groups[tr]).isdisjoint(groups[va])


def test_save_load_roundtrip(tmp_path):
    p, X, _ = fitted()
    path = save_pipeline(p, tmp_path / "artifacts" / "m.joblib")
    p2, meta = load_pipeline(path, return_metadata=True)
    assert (p2.predict(X) == p.predict(X)).all()
    assert meta["classes"] == ["Billing", "Technical"]          # class order is stored with the model
    assert "sklearn_version" in meta


def test_load_missing_file_gives_clear_error(tmp_path):
    try:
        load_pipeline(tmp_path / "nope.joblib")
    except FileNotFoundError as e:
        assert "Model file not found" in str(e)
    else:
        raise AssertionError("FileNotFoundError expected")


def test_dataset_fingerprint_changes_with_content(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("x\n1\n"); b.write_text("x\n1\n")
    assert dataset_fingerprint(a) == dataset_fingerprint(b)
    b.write_text("x\n2\n")
    assert dataset_fingerprint(a) != dataset_fingerprint(b)


def test_metric_names_and_per_class_metrics():
    assert metric_key("Billing and Payments") == "Billing and Payments"
    assert metric_key("A:B*") == "A_B_"
    out = per_class_metrics(["a", "b", "a"], ["a", "a", "a"], ["a", "b"])
    assert set(out) == {"precision__a", "recall__a", "f1__a", "precision__b", "recall__b", "f1__b"}
    assert out["recall__b"] == 0.0


def test_mlflow_logging(tmp_path):
    train = tmp_path / "train.csv"
    train.write_text("x\n1\n")
    setup_mlflow(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")     # as_posix: works on Windows too
    log_experiment("t", "LogReg", params={"C": 1}, metrics={"macro_f1": 0.9, "f1__Billing and Payments": 0.5},
                   pipeline_config={"ngram_range": "(1, 2)"}, split_files=[train], calibration_method="none")

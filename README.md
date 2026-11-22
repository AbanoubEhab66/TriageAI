# Source code (`src/triageai`)

| File | What it does |
|---|---|
| `validation.py` | Input validation (`is_valid_input`, `validate_ticket`): a ticket is valid if Subject **or** Body has text |
| `preprocessing.py` | `combine_text`, `clean_text`, `TextPreprocessor` (sklearn transformer, same code in training and prediction) |
| `model.py` | `build_pipeline(classifier, ...)`, `FEATURE_CONFIGS`, `save_pipeline` / `load_pipeline` (joblib) |
| `mlflow_utils.py` | `setup_mlflow()`, `log_experiment()`, `per_class_metrics()`: one logging format for everyone |

## Setup
```bash
pip install -r requirements.txt
pip install -e .          # optional: makes `import triageai` work from anywhere
pytest                    # tests find src/ automatically (see pyproject.toml)
```

## Use
```python
from sklearn.linear_model import LogisticRegression
from triageai.model import build_pipeline, RECOMMENDED_CONFIG, save_pipeline, load_pipeline
from triageai.mlflow_utils import setup_mlflow, log_experiment, per_class_metrics

pipe = build_pipeline(LogisticRegression(max_iter=2000, C=10, class_weight="balanced"),
                      **RECOMMENDED_CONFIG)
pipe.fit(train[["subject", "body"]], train["queue"])          # TF-IDF is fitted on train only
proba = pipe.predict_proba(val[["subject", "body"]])

setup_mlflow()
log_experiment("my_run", "LogisticRegression",
               params={"C": 10}, metrics={"macro_f1": 0.5, **per_class_metrics(y, pred, labels)},
               pipeline_config={"ngram_range": "(1, 2)"},
               split_files=["data/train.csv", "data/val.csv"])  # logs a fingerprint of the split

save_pipeline(pipe, "artifacts/model.joblib")                 # everyone saves/loads the same way
pipe = load_pipeline("artifacts/model.joblib")
```

## Rules for the whole team
* The vectorizer is **inside** the pipeline. Never fit TF-IDF on all data before splitting.
* Cross-validation on train = `StratifiedGroupKFold` with `groups=train["group_id"]`.
* The test set is used once, at the very end.
* MLflow names (use exactly these): `macro_f1`, `weighted_f1`, `accuracy`, `val_macro_f1`, `top2_accuracy`,
  `f1__<queue>`, `precision__<queue>`, `recall__<queue>`, `raw_ece`, `calibrated_ece`, `threshold`, `coverage`,
  `routing_error_rate`, `routing_error_ci_low`, `routing_error_ci_high`, `human_review_rate`,
  `auto_routed_count`, `auto_routed_errors`.
* Tracking DB: `<repo>/mlflow.db` (local to each person) unless `MLFLOW_TRACKING_URI` is set. Do not commit it.

## Re-run the feature comparison
```bash
python run_feature_experiments.py --data-dir data --reports-dir reports
```
Writes `feature_comparison.csv`, `per_class_best.csv`, `confusion_matrix_best.png`,
`top_confusions_best.csv`, `cleaning_stats.csv`. It stops with an error if any duplicate group
appears in both train and val.

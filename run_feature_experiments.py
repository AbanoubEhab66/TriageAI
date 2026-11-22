"""Compare feature configs on Validation (+ grouped CV on train). Test set is NOT touched."""
import json, time, joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from src.pipeline import build_pipeline
from src.mlflow_utils import setup_mlflow, log_experiment, RANDOM_STATE

tr, va = pd.read_csv("data/train.csv"), pd.read_csv("data/val.csv")
cols = ["subject", "body"]
Xtr, ytr, Xva, yva = tr[cols], tr["queue"], va[cols], va["queue"]
setup_mlflow()

CONFIGS = {
    "tfidf_uni":               dict(ngram_range=(1, 1)),
    "tfidf_uni_bi":            dict(ngram_range=(1, 2)),
    "tfidf_uni_bi_stopwords":  dict(ngram_range=(1, 2), stop_words="english"),
    "tfidf_uni_stopwords":     dict(ngram_range=(1, 1), stop_words="english"),
    "bow_uni_bi":              dict(ngram_range=(1, 2), use_bow=True),
}
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows = []
for name, cfg in CONFIGS.items():
    t = time.time()
    clf = LogisticRegression(max_iter=2000, C=10, class_weight="balanced", random_state=RANDOM_STATE)
    pipe = build_pipeline(clf, **cfg)
    cvs = cross_val_score(pipe, Xtr, ytr, groups=tr["group_id"], cv=cv, scoring="f1_macro", n_jobs=-1)
    pipe.fit(Xtr, ytr)
    p = pipe.predict(Xva)
    m = dict(val_macro_f1=f1_score(yva, p, average="macro"), weighted_f1=f1_score(yva, p, average="weighted"),
             accuracy=accuracy_score(yva, p), cv_macro_f1_mean=cvs.mean(), cv_macro_f1_std=cvs.std())
    log_experiment(name, "LogisticRegression", params=dict(C=10, class_weight="balanced"),
                   metrics=m, pipeline_config={k: str(v) for k, v in cfg.items()},
                   cv_info=dict(splitter="StratifiedGroupKFold", n_folds=5, groups="group_id"))
    rows.append(dict(config=name, **{k: round(v, 4) for k, v in m.items()}, sec=round(time.time() - t)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("feature_comparison.csv", index=False)

"""build_pipeline(classifier): the shared interface for the whole team,
plus save/load helpers so everybody persists the model the same way (joblib)."""
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from .preprocessing import TextPreprocessor

# Feature configurations compared in Phase 2 (single source of truth; reuse them in Phase 3).
FEATURE_CONFIGS = {
    "tfidf_uni":                      dict(ngram_range=(1, 1)),
    "tfidf_uni_bi":                   dict(ngram_range=(1, 2)),
    "tfidf_uni_bi_stopwords":         dict(ngram_range=(1, 2), stop_words="english"),
    "tfidf_uni_stopwords":            dict(ngram_range=(1, 1), stop_words="english"),
    "bow_uni_bi":                     dict(ngram_range=(1, 2), use_bow=True),
    "tfidf_uni_bi_stopwords_noclean": dict(ngram_range=(1, 2), stop_words="english", clean=False),
}

# Best on Validation and grouped CV with LogReg C=10 balanced (reports/feature_comparison.csv).
# Bigrams are a clear gain; stop-word removal is within noise (see CHANGES.md), kept as default.
RECOMMENDED_CONFIG = dict(ngram_range=(1, 2), stop_words="english")


def build_pipeline(classifier, ngram_range=(1, 2), stop_words=None, min_df=2,
                   max_df=0.95, sublinear_tf=True, max_features=None,
                   clean=True, use_bow=False):
    """Preprocessor -> vectorizer -> classifier.

    The vectorizer lives INSIDE the pipeline, so it is fitted on training data only
    (no leakage in train / CV). stop_words: None or 'english'.
    use_bow=True -> CountVectorizer (sublinear_tf is ignored).
    """
    common = dict(ngram_range=ngram_range, stop_words=stop_words, min_df=min_df,
                  max_df=max_df, max_features=max_features)
    vec = (CountVectorizer(**common) if use_bow
           else TfidfVectorizer(sublinear_tf=sublinear_tf, **common))
    return Pipeline([
        ("preprocess", TextPreprocessor(clean=clean)),
        ("vectorizer", vec),
        ("clf", classifier),
    ])


def save_pipeline(pipeline, path="artifacts/model.joblib", metadata=None):
    """Save a fitted pipeline with joblib, plus metadata (class order, library versions)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = getattr(pipeline, "classes_", None)
    meta = {
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": [str(c) for c in classes] if classes is not None else None,
        **(metadata or {}),
    }
    joblib.dump({"pipeline": pipeline, "metadata": meta}, path)
    return path


def load_pipeline(path="artifacts/model.joblib", return_metadata=False):
    """Load a pipeline saved with save_pipeline(). Warns if scikit-learn versions differ."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. "
                                "Train and save a model first (save_pipeline).")
    obj = joblib.load(path)
    pipeline, meta = obj["pipeline"], obj.get("metadata", {})
    saved = meta.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        warnings.warn(f"Model was saved with scikit-learn {saved} but {sklearn.__version__} "
                      "is installed; predictions may differ or fail.", stacklevel=2)
    return (pipeline, meta) if return_metadata else pipeline

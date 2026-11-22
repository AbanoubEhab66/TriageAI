"""build_pipeline(classifier): the shared interface for the whole team."""
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from .preprocessing import TextPreprocessor

# Ready-made TF-IDF configs to compare in MLflow
TFIDF_CONFIGS = {
    "tfidf_uni": dict(ngram_range=(1, 1)),
    "tfidf_uni_bi": dict(ngram_range=(1, 2)),
}


def build_pipeline(classifier, ngram_range=(1, 2), stop_words=None, min_df=2,
                   max_df=0.95, sublinear_tf=True, max_features=None,
                   clean=True, use_bow=False):
    """Preprocessor -> vectorizer -> classifier.
    The vectorizer lives INSIDE the pipeline, so it is fitted on training data only
    (no leakage in train/CV). stop_words: None or 'english'. use_bow=True -> CountVectorizer.
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


# Best config on Validation + grouped CV (see feature_comparison.csv), LogReg C=10 balanced.
RECOMMENDED_CONFIG = dict(ngram_range=(1, 2), stop_words="english")

"""Text preprocessing for TriageAI. Used identically in training and prediction."""
import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .validation import is_valid_input, safe_str  # noqa: F401  (re-exported for convenience)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WS_RE = re.compile(r"\s+")


def combine_text(subject, body) -> str:
    """Merge subject + body into one string (same rule everywhere)."""
    return f"{safe_str(subject).strip()} {safe_str(body).strip()}".strip()


def clean_text(text) -> str:
    """Light cleaning: lowercase, mask URLs/emails, collapse whitespace.
    Numbers and punctuation inside words are kept (they can help classification)."""
    text = safe_str(text).lower()
    text = URL_RE.sub(" url ", text)
    text = EMAIL_RE.sub(" email ", text)
    return WS_RE.sub(" ", text).strip()


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn transformer. Accepts:
      - DataFrame with 'subject' and/or 'body' columns
      - DataFrame with a single 'text' column
      - list / Series / array of strings, or a single string
    Stateless, so it is safe inside Pipeline / cross-validation.
    """

    def __init__(self, clean=True):
        self.clean = clean

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, str):            # a single ticket; do not iterate over characters
            X = [X]
        if isinstance(X, pd.DataFrame):
            if "subject" in X.columns or "body" in X.columns:
                subj = X["subject"] if "subject" in X.columns else [""] * len(X)
                body = X["body"] if "body" in X.columns else [""] * len(X)
                texts = [combine_text(s, b) for s, b in zip(subj, body)]
            elif "text" in X.columns:
                texts = [safe_str(t) for t in X["text"]]
            else:
                raise ValueError("DataFrame needs 'subject'/'body' or 'text' column")
        else:
            texts = [safe_str(t) for t in X]
        return [clean_text(t) for t in texts] if self.clean else texts

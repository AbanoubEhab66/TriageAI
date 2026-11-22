"""Text preprocessing for TriageAI. Used identically in training and prediction."""
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WS_RE = re.compile(r"\s+")


def _safe(x) -> str:
    """None / NaN -> empty string, everything else -> str."""
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


def combine_text(subject, body) -> str:
    """Merge subject + body into one string (same rule everywhere)."""
    return f"{_safe(subject).strip()} {_safe(body).strip()}".strip()


def is_valid_input(subject, body) -> bool:
    """Valid if Subject OR Body has non-empty text (team agreement #12)."""
    return bool(_safe(subject).strip() or _safe(body).strip())


def clean_text(text: str) -> str:
    """Light cleaning: lowercase, mask URLs/emails, collapse whitespace.
    Numbers and punctuation inside words are kept (they can help classification)."""
    text = _safe(text).lower()
    text = URL_RE.sub(" url ", text)
    text = EMAIL_RE.sub(" email ", text)
    return WS_RE.sub(" ", text).strip()


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn transformer. Accepts:
      - DataFrame with 'subject' and/or 'body' columns
      - DataFrame with a single 'text' column
      - list / Series of strings
    Stateless, so it is safe to place inside Pipeline / cross-validation.
    """

    def __init__(self, clean=True):
        self.clean = clean

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if "subject" in X.columns or "body" in X.columns:
                subj = X["subject"] if "subject" in X.columns else [""] * len(X)
                body = X["body"] if "body" in X.columns else [""] * len(X)
                texts = [combine_text(s, b) for s, b in zip(subj, body)]
            elif "text" in X.columns:
                texts = [_safe(t) for t in X["text"]]
            else:
                raise ValueError("DataFrame needs 'subject'/'body' or 'text' column")
        else:
            texts = [_safe(t) for t in X]
        return [clean_text(t) for t in texts] if self.clean else texts

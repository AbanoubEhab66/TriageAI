"""Input validation, shared by training, inference and the app.

Team agreement #12: a ticket is valid if the Subject OR the Body has non-empty text.
Only a ticket with both fields empty / whitespace-only is rejected.
"""
import numpy as np
import pandas as pd


def safe_str(x) -> str:
    """None / NaN / pd.NA -> empty string; everything else -> str."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    missing = pd.isna(x)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return ""
    return str(x)


def is_valid_input(subject, body) -> bool:
    """True if Subject or Body contains non-whitespace text."""
    return bool(safe_str(subject).strip() or safe_str(body).strip())


def validate_ticket(subject, body) -> None:
    """Raise ValueError with a clear message when the ticket has no usable text."""
    if not is_valid_input(subject, body):
        raise ValueError("Empty ticket: please provide a subject or a body.")

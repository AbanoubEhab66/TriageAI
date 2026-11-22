import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.pipeline import build_pipeline
from src.preprocessing import clean_text, combine_text, is_valid_input
from src.mlflow_utils import setup_mlflow, log_experiment

def data():
    rows = [("I was charged twice", "billing payment invoice refund"),
            ("Server down", "technical outage error crash"),
            ("Invoice wrong", "billing payment charged"),
            ("App crash", "technical error bug crash"),] * 5
    df = pd.DataFrame(rows, columns=["subject", "body"])
    y = ["Billing", "Technical", "Billing", "Technical"] * 5
    return df, y

def test_basic():
    assert clean_text(None) == ""
    assert combine_text(None, "hi") == "hi"
    assert is_valid_input("", "x") and not is_valid_input(" ", None)
    assert "url" in clean_text("see https://a.com")

def test_pipeline_and_mlflow(tmp_path):
    X, y = data()
    p = build_pipeline(LogisticRegression(max_iter=500, random_state=42), min_df=1)
    p.fit(X, y)
    proba = p.predict_proba(X.head(2))
    assert proba.shape == (2, 2)
    # works with NaN subject and plain strings
    p.predict(pd.DataFrame({"subject": [None], "body": ["billing invoice"]}))
    p.predict(["billing invoice"])
    setup_mlflow(f"sqlite:///{tmp_path}/mlflow.db")
    log_experiment("t", "LogReg", params={"C": 1}, metrics={"macro_f1": 0.9},
                   pipeline_config={"ngram_range": "(1, 2)"})

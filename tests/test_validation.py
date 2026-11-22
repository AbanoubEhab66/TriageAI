import numpy as np
import pandas as pd

from triageai.preprocessing import TextPreprocessor, clean_text, combine_text
from triageai.validation import is_valid_input, safe_str, validate_ticket


def test_safe_str_handles_all_missing_types():
    assert safe_str(None) == ""
    assert safe_str(np.nan) == ""
    assert safe_str(pd.NA) == ""
    assert safe_str(12) == "12"
    assert safe_str("x") == "x"


def test_input_validation_rule():
    # valid if subject OR body has text (agreement #12)
    assert is_valid_input("", "x")
    assert is_valid_input("x", None)
    assert not is_valid_input(" ", None)
    assert not is_valid_input("", "")
    assert not is_valid_input(None, np.nan)


def test_validate_ticket_raises_clear_error():
    validate_ticket("only subject", "")          # no error
    try:
        validate_ticket("  ", None)
    except ValueError as e:
        assert "Empty ticket" in str(e)
    else:
        raise AssertionError("ValueError expected")


def test_cleaning_and_combining():
    assert clean_text(None) == ""
    assert combine_text(None, "hi") == "hi"
    assert combine_text(" a ", " b ") == "a b"
    assert clean_text("See  HTTPS://a.com  now") == "see url now"
    assert clean_text("mail me: a.b@c.com") == "mail me: email"


def test_preprocessor_accepts_all_input_shapes():
    tp = TextPreprocessor()
    assert tp.transform("Hello  World") == ["hello world"]                       # single string
    assert tp.transform(["A", None]) == ["a", ""]                                 # list
    assert tp.transform(pd.Series(["A b"])) == ["a b"]                            # Series
    df = pd.DataFrame({"subject": [None, "S"], "body": ["B", None]})
    assert tp.transform(df) == ["b", "s"]                                         # subject/body
    assert tp.transform(pd.DataFrame({"text": ["T x"]})) == ["t x"]               # text column
    assert TextPreprocessor(clean=False).transform(["A  B"]) == ["A  B"]
    try:
        tp.transform(pd.DataFrame({"other": [1]}))
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError expected")

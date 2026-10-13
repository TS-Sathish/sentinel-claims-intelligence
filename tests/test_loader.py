import pandas as pd
from src.data.loader import find_target, normalize_target, excluded_columns

def test_target_detection():
    assert find_target(pd.DataFrame({"x":[1,2],"FraudFound_P":[0,1]})) == "FraudFound_P"

def test_yes_no_labels():
    assert normalize_target(pd.Series(["Y","N"])).tolist() == [1,0]

def test_target_variants_and_invalid_labels():
    assert normalize_target(pd.Series(["Fraudulent", "Genuine"])).tolist() == [1, 0]
    try:
        normalize_target(pd.Series(["maybe", "no"]))
    except ValueError as exc:
        assert "Unrecognized labels" in str(exc)
    else:
        raise AssertionError("unknown target labels should fail")

def test_leakage_exclusions():
    found = excluded_columns(["PolicyNumber", "Age", "Fault", "Make", "FraudFound_P"], "FraudFound_P")
    assert set(found) == {"PolicyNumber", "Age", "Fault", "FraudFound_P"}
    assert "sensitive" in found["Age"]

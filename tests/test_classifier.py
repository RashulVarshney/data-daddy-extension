import numpy as np
import pytest

from datadaddy_ai.classifier.features import (
    REGEX_FEATURE_NAMES,
    ColumnFeaturizer,
    dense_features,
    is_generic_header,
    luhn_valid,
    normalise_header,
    regex_features,
    sample_values,
)
from datadaddy_ai.classifier.labels import LABELS
from datadaddy_ai.classifier.model import ColumnClassifier, header_dropout
from datadaddy_ai.classifier.scan import scan_dataframe
from datadaddy_ai.classifier.synth import SyntheticGenerator, make_hardneg_set, make_pii_set


def test_generator_deterministic_under_seed():
    a = make_pii_set(7, ["en_US", "en_IN"], 3) + make_hardneg_set(7, ["en_US"], 5)
    b = make_pii_set(7, ["en_US", "en_IN"], 3) + make_hardneg_set(7, ["en_US"], 5)
    assert [(c.header, c.label, c.values) for c in a] == [(c.header, c.label, c.values) for c in b]


def test_generator_differs_across_seeds():
    a = make_pii_set(1, ["en_US"], 2)
    b = make_pii_set(2, ["en_US"], 2)
    assert [c.values for c in a] != [c.values for c in b]


def test_synthetic_govt_ids_are_invalid_by_construction():
    g = SyntheticGenerator(3)
    for v in g.gen_govt_id(100, "en_US"):
        assert v.replace("-", "")[:3] in {"000", "666"} or v.startswith("9")
    for v in g.gen_govt_id(100, "en_GB"):
        assert v[:2] in {"BG", "GB", "NK", "KN", "TN", "NT", "ZZ"}


def test_normalise_and_generic_header():
    assert normalise_header("userEmail") == "user email"
    assert normalise_header(None) == "__none__"
    assert is_generic_header("col_3") and is_generic_header("f12") and is_generic_header(None)
    assert not is_generic_header("email")


def test_luhn():
    assert luhn_valid("79927398713") and not luhn_valid("79927398710")


def test_sample_values_caps_and_missing():
    vals = [str(i) for i in range(500)] + [None, "", "N/A"]
    s, null_rate, _ = sample_values(vals, seed=1)
    assert len(s) <= 200 and 0 <= null_rate < 1
    s2, nr2, ws = sample_values([" a ", "N/A", None, "b"])
    assert s2 == ["a", "b"] and nr2 == 0.5 and ws == 0.5


def test_regex_features_email_and_ip():
    f = dict(zip(REGEX_FEATURE_NAMES, regex_features(["a@b.com", "x@y.org"]), strict=True))
    assert f["email"] == 1.0 and f["ipv4"] == 0.0
    f = dict(zip(REGEX_FEATURE_NAMES, regex_features(["10.0.0.1", "8.8.8.8"]), strict=True))
    assert f["ipv4"] == 1.0


def test_dense_features_finite_on_weird_input():
    x = dense_features(["inf", "1e999", "nan", "-5", "abc"], 0.1, 0.0)
    assert np.isfinite(x).all()


def test_header_dropout_rate_and_determinism():
    cols = make_pii_set(1, ["en_US"], 20, header_mode="clear")
    a = header_dropout(cols, 0.5, seed=3)
    b = header_dropout(cols, 0.5, seed=3)
    assert a == b
    changed = sum(1 for (_, h), c in zip(a, cols, strict=True) if h != c.header)
    assert 0.25 < changed / len(cols) < 0.75


@pytest.fixture(scope="module")
def tiny_model():
    tr = make_pii_set(1, ["en_US", "en_IN"], 25) + make_hardneg_set(2, ["en_US"], 150)
    return ColumnClassifier("logreg", seed=1).fit(tr)


def test_featurizer_shapes_consistent():
    cols = make_pii_set(1, ["en_US"], 10)
    f = ColumnFeaturizer()
    pairs = [(c.values, c.header) for c in cols]
    X = f.fit_transform(pairs)
    assert f.transform(pairs[:3]).shape[1] == X.shape[1]


def test_tiny_model_learns_obvious_columns(tiny_model):
    te = make_pii_set(99, ["en_US", "en_IN"], 8, header_mode="none")
    pred = tiny_model.predict_instances(te)
    acc = np.mean([p == c.label for p, c in zip(pred, te, strict=True)])
    assert acc > 0.8


def test_scan_dataframe_report(tiny_model):
    import pandas as pd

    g = SyntheticGenerator(5)
    df = pd.DataFrame({"email": g.gen_email(50, "en_US"), "n": [str(i % 7) for i in range(50)]})
    rep = scan_dataframe(df, tiny_model)
    assert rep["columns"][0]["predicted_class"] == "EMAIL"
    assert rep["risk"]["level"] in {"MEDIUM", "HIGH", "CRITICAL", "LOW"}
    assert set(rep["columns"][0]["top3"]) <= set(LABELS)

import pytest

from datadaddy_ai.classifier.model import ColumnClassifier
from datadaddy_ai.classifier.synth import make_hardneg_set, make_pii_set


@pytest.fixture(scope="session")
def tiny_clf():
    tr = make_pii_set(1, ["en_US", "en_IN"], 25) + make_hardneg_set(2, ["en_US"], 150)
    return ColumnClassifier("logreg", seed=1).fit(tr)

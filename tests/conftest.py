import numpy as np
import pytest

from dre.data import m5, stockouts
from dre.features import FeatureBuilder


@pytest.fixture(scope="session")
def panel():
    return m5.synthetic(n_items=30, stores=("CA_1", "TX_1"), days=520, seed=3)


@pytest.fixture(scope="session")
def fb(panel):
    return FeatureBuilder(panel, stockouts.detect(panel.sales))


@pytest.fixture(scope="session")
def cfg():
    import yaml

    return yaml.safe_load(open("config/dre.yaml"))


@pytest.fixture
def rng():
    return np.random.default_rng(0)

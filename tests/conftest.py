import os
import pickle
import pytest

@pytest.fixture
def wf_network_fixture():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "wf_network.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

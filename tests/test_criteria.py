from utils import least_confidence, margin_sampling, entropy

import pytest
import numpy as np
from scipy.special import softmax


@pytest.fixture
def generate_pred_prob():
    x = np.random.rand(10, 100)
    if np.random.rand() < 0.5:
        x = softmax(x)
    return x


def test_least_confidence(generate_pred_prob):

    k = np.random.randint(0, 120)
    with pytest.raises(AssertionError):
        least_confidence(pred_prob=generate_pred_prob, k=k)


def test_margin_sampling(generate_pred_prob):
    k = np.random.randint(0, 120)
    with pytest.raises(AssertionError):
        margin_sampling(pred_prob=generate_pred_prob, k=k)


def test_entropy(generate_pred_prob):
    k = np.random.randint(0, 120)
    with pytest.raises(AssertionError):
        entropy(pred_prob=generate_pred_prob, k=k)

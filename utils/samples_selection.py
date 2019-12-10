from utils.criteria import least_confidence, entropy, margin_sampling
import numpy as np


def get_high_confidence_samples(pred_prob: np.ndarray,
                                delta: float) -> np.ndarray:
    """
    Select high confidence samples from `D^U` whose entropy is smaller than
     the threshold
    `delta`.

    Parameters
    ----------
    pred_prob : np.ndarray
        prediction probability of x_i with dimension (batch x n_class)
    delta : float
        threshold

    Returns
    -------
    np.array  with dimension (None, 3). The highest confidence samples.
        column 1: index original of the sample.
        column 2: predicted class.
        column 3: entropy
    """
    eni = entropy(pred_prob=pred_prob, k=len(pred_prob))
    hcs = eni[eni[:, 2] < delta]
    return hcs


def get_uncertain_samples(pred_prob: np.ndarray, k: int, criteria: str):
    """
    Get the K most informative samples based on the criteria
    Parameters
    ----------
    pred_prob : np.ndarray
        prediction probability of x_i with dimension (batch x n_class)
    k: int
    criteria: str
        `cl` : least_confidence()
        `ms` : margin_sampling()
        `en` : entropy

    Returns
    -------
    np.ndarray with dimension (K x 3)
    """
    if criteria == 'cl':
        uncertain_samples = least_confidence(pred_prob=pred_prob, k=k)
    elif criteria == 'ms':
        uncertain_samples = margin_sampling(pred_prob=pred_prob, k=k)
    elif criteria == 'en':
        uncertain_samples = entropy(pred_prob=pred_prob, k=k)
    else:
        raise ValueError('criteria {} not found !'.format(criteria))
    return uncertain_samples

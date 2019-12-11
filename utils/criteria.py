# Authors rafik gouiaa <rafikgouiaaphd@gmail.com>, ...
from typing import Tuple
import numpy as np


def least_confidence(pred_prob: np.ndarray, k: int) -> Tuple[np.ndarray,
                                                             np.ndarray]:
    f"""
    Rank all the unlabeled samples in an ascending order according to
    equation 2

    Parameters
    ----------
    pred_prob : prediction probability of x_i with dimension (batch x n_class)
    k : int
        most informative samples
    Returns
    -------
    np.array with dimension (K x 1) containing the indices of the K
        most informative samples.
    np.array with dimension (K x 3) containing the indices, the predicted class
        and the `lc` of the k most informative samples
        column 1: indices
        column 2: predicted class.
        column 3: lc
    """
    assert np.round(pred_prob.sum(1).sum()) == pred_prob.shape[
        0], "pred_prob is not " \
            "a probability" \
            " distribution"
    assert 0 < k <= pred_prob.shape[0], "invalid k value k should be >0 &" \
                                        "k <=  pred_prob.shape[0"
    # Get max probabilities prediction and its corresponding classes
    most_pred_prob, most_pred_class = np.max(pred_prob, axis=1), np.argmax(
        pred_prob, axis=1)
    size = len(pred_prob)
    lc_i = np.column_stack(
        (list(range(size)), most_pred_class, most_pred_prob))
    # sort lc_i in ascending order
    lc_i = lc_i[lc_i[:, -1].argsort()]

    return lc_i[:k, 0].astype(np.int32), lc_i[:k]


def margin_sampling(pred_prob: np.ndarray, k: int) -> Tuple[np.ndarray,
                                                            np.ndarray]:
    f"""
    Rank all the unlabeled samples in an ascending order according to the
    equation 3
    ----------
    pred_prob : np.ndarray
        prediction probability of x_i with dimension (batch x n_class)
    k : int
        most informative samples

    Returns
    -------
    np.array with dimension (K x 1)  containing the indices of the K
        most informative samples.
    np.array with dimension (K x 3) containing the indices, the predicted class
        and the `ms_i` of the k most informative samples
        column 1: indices
        column 2: predicted class.
        column 3: margin sampling
    """
    assert np.round(pred_prob.sum(1).sum()) == pred_prob.shape[
        0], "pred_prob is not " \
            "a probability" \
            " distribution"
    assert 0 < k <= pred_prob.shape[0], "invalid k value k should be >0 &" \
                                        "k <=  pred_prob.shape[0"
    # Sort pred_prob to get j1 and j2
    size = len(pred_prob)
    margin = np.diff(np.abs(np.sort(pred_prob, axis=1)[:, ::-1][:, :2]))
    pred_class = np.argmax(pred_prob, axis=1)
    ms_i = np.column_stack((list(range(size)), pred_class, margin))

    # sort ms_i in ascending order according to margin
    ms_i = ms_i[ms_i[:, 2].argsort()]

    # the smaller the margin  means the classifier is more
    # uncertain about the sample
    return ms_i[:k, 0].astype(np.int32), ms_i[:k]


def entropy(pred_prob: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    f"""
    Rank all the unlabeled samples in an descending order according to
    the equation 4

    Parameters
    ----------
    pred_prob : np.ndarray
        prediction probability of x_i with dimension (batch x n_class)
    k : int

    Returns
    -------
    np.array with dimension (K x 1)  containing the indices of the K
        most informative samples.
    np.array with dimension (K x 3) containing the indices, the predicted class
        and the `en_i` of the k most informative samples
        column 1: indices
        column 2: predicted class.
        column 3: entropy

    """
    # calculate the entropy for the pred_prob
    assert np.round(pred_prob.sum(1).sum()) == pred_prob.shape[
        0], "pred_prob is not " \
            "a probability" \
            " distribution"
    assert 0 < k <= pred_prob.shape[0], "invalid k value k should be >0 &" \
                                        "k <=  pred_prob.shape[0"
    size = len(pred_prob)
    entropy_ = - np.nansum(pred_prob * np.log(pred_prob), axis=1)
    pred_class = np.argmax(pred_prob, axis=1)
    en_i = np.column_stack((list(range(size)), pred_class, entropy_))

    # Sort en_i in descending order
    en_i = en_i[(-1 * en_i[:, 2]).argsort()]
    return en_i[:k, 0].astype(np.int32), en_i[:k]

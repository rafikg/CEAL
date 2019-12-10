# Authors rafik gouiaa <rafikgouiaaphd@gmail.com>, ...


import numpy as np


def least_confidence(pred_prob: np.ndarray, k: int) -> np.ndarray:
    f"""
    Rank all the unlabeled samples in an ascending order according to
    ..math::
    \\displaymath lc_i = \max p(y_i = j| x_i; W)

    Parameters
    ----------
    pred_prob : prediction probability of x_i with dimension (batch x n_class)
    k : int
        most informative samples
    Returns
    -------
    np.array with dimension (K x 3) containing the K most informative samples.
        column 1: index original of the sample.
        column 2: predicted class.
        column 3: lc
    """
    # Get max probabilities prediction and its corresponding classes
    most_pred_prob, most_pred_class = np.max(pred_prob, axis=1), np.argmax(
        pred_prob, axis=1)
    l = len(pred_prob)
    lc_i = np.column_stack((list(range(l)), most_pred_class, most_pred_prob))
    # sort lc_i in ascending order
    lc_i = lc_i[lc_i[:, -1].argsort()]

    return lc_i[:k]


def margin_sampling(pred_prob: np.ndarray, k: int) -> np.ndarray:
    f"""
    Rank all the unlabeled samples in an ascending order according to the
    ..math::
    \\displaymath ms_i = p(y_i = j_1| x_i;W) - p(y_i = j_2|x_i;W)
    Parameters
    ----------
    pred_prob : np.ndarray
        prediction probability of x_i with dimension (batch x n_class)
    k : int
        most informative samples

    Returns
    -------
    np.array with dimension (K x 3) containing the K most informative samples.
        column 1: index original of the sample.
        column 2: predicted class.
        column 3: margin
    """
    # Sort pred_prob to get j1 and j2
    l = len(pred_prob)
    margin = np.diff(np.abs(np.sort(pred_prob, axis=1)[:, ::-1][:, :2]))
    pred_class = np.argmax(pred_prob, axis=1)
    ms_i = np.column_stack((list(range(l)), pred_class, margin))

    # sort ms_i in ascending order according to margin
    ms_i = ms_i[ms_i[:, 2].argsort()]

    # the smaller the margin  means the classifier is more
    # uncertain about the sample
    return ms_i[:k]


def entropy(pred_prob: np.ndarray, k: int) -> np.ndarray:
    f"""
    Rank all the unlabeled samples in an descending order according to their
    
    ..math::

    Parameters
    ----------
    pred_prob : np.ndarray
        prediction probability of x_i with dimension (batch x n_class)
    k : int

    Returns
    -------
    np.array with dimension (K x 3) containing the K most informative samples.
        column 1: index original of the sample.
        column 2: predicted class.
        column 3: entropy
    """
    # calculate the entropy for the pred_prob
    l = len(pred_prob)
    entropy = - np.nansum(pred_prob * np.log(pred_prob), axis=0)
    pred_class = np.argmax(pred_prob, axis=1)
    en_i = np.column_stack((list(range(l)), pred_class, entropy))

    # Sort en_i in descending order
    en_i = en_i[(-1 * en_i[:, 2]).argsort()]
    return en_i[:k]

# Authors rafik gouiaa <rafikgouiaaphd@gmail.com>, ...


import numpy as np


def least_confidence(pred_prob: np.ndarray, k: int) -> np.ndarray:
    f"""
    Rank all the unlabeled samples in an ascending order according to
    ..math::
    lc_i = \max p(y_i = j| x_i; W)

    Parameters
    ----------
    pred_prob : prediction probability of x_i with dimension (batch x n_class)
    k : int
        most informative samples
    Returns
    -------
    K most informative samples based on the l_ci
    """
    # Get max probabilities prediction and its corresponding classes
    most_pred_prob, most_pred_class = np.max(pred_prob, axis=1), np.argmax(pred_prob, axis=1)
    l = len(pred_prob)
    lc_i = np.column_stack((list(range(l)), most_pred_class, most_pred_prob))
    # sort lc_i in ascending order
    lc_i = lc_i[lc_i[:, -1].argsort()]

    return lc_i[:k]


def margin_sampling(pred_prob: np.ndarray, k: int)->np.ndarray:
    """
    Rank all the unlabeled samples in an ascending order according to the
    ..math::
    ms_i = p(y_i = j_1| x_i;W) - p(y_i = j_2|x_i;W)
    Parameters
    ----------
    pred_prob : prediction probability of x_i with dimension (batch x n_class)
    k : int

    Returns
    -------
    K most informative samples based on the l_ci.
    """
    # Sort pred_prob to get j1 and j2
    l = len(pred_prob)
    margin = np.diff(np.abs(np.sort(pred_prob, axis=1)[:, ::-1][:, :2]))
    most_pred_class = np.argmax(pred_prob, axis=1)
    ms_i = np.column_stack((list(range(l)), most_pred_class, margin))

    # sort ms_i in ascending order according to margin
    ms_i = ms_i[ms_i[:, 2].argsort()]

    # the smaller the margin  means the classifier is more
    # uncertain about the sample
    return ms_i[:k]






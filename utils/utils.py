def update_threshold(delta: float, dr: float, t: int) -> float:
    """
    Update the selection threshold of high confidence samples
    Parameters
    ----------
    delta
    dr
    t

    Returns
    -------

    """
    if t > 0:
        delta = delta - dr * t
    return delta

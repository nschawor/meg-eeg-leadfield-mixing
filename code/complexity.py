import numpy as np


def compute_norm_coefficients(weighted_patterns, nr_components):
    """Computes normalized spatial pattern coefficients.

    Parameters
    ----------
        weighted_patterns : array, patterns weighted by amplitude.
        nr_components : measure is calculated using this many components.

    Returns
    -------
        M : array, normalized spatial pattern coefficient.
    """

    weighted_patterns = weighted_patterns.astype("float32")

    # take absolute value
    M = np.abs(weighted_patterns)
    M = M[:, :nr_components]

    # normalize across dipoles
    norm_M = np.sum(M, axis=1)
    M = (M.T / norm_M).T
    return M


def compute_sensor_complexity(weighted_patterns, nr_components):
    """Computes sensor complexity as a proxy for assessing spatial mixing.

    Parameters
    ----------
        weighted_patterns : array, patterns weighted by amplitude.
        nr_components : measure is calcualted using this many components.

    Returns
    -------
        sensor_complexity : array, sensor complexity for each sensor.
    """
    M = compute_norm_coefficients(weighted_patterns, nr_components)
    sensor_complexity = -np.sum(M * np.log(M), axis=1)

    return sensor_complexity

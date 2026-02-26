import numpy as np
from typing import Optional

def compute_sequence_separation_mask(
    seq_len: int,
    minimum_sequence_separation: Optional[int] = 1,
) -> np.ndarray:
    """
    Create a boolean mask enforcing |i - j| >= minimum_sequence_separation.

    Parameters
    ----------
    seq_len : int
        Length of the sequence.
    minimum_sequence_separation : int, optional
        Minimum |i - j| required. If None or <= 0, no restriction.

    Returns
    -------
    np.ndarray (bool, shape = (seq_len, seq_len))
    """
    if minimum_sequence_separation is None or minimum_sequence_separation <= 0:
        return np.ones((seq_len, seq_len), dtype=bool)

    idx = np.arange(seq_len)
    return np.abs(idx[:, None] - idx[None, :]) >= minimum_sequence_separation

def compute_distance_mask(
    distance_matrix: np.ndarray,
    maximum_contact_distance: Optional[float] = None,
) -> np.ndarray:
    """
    Create a boolean mask enforcing distance <= maximum_contact_distance.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Square (N, N) distance matrix.
    maximum_contact_distance : float, optional
        Distance cutoff. If None, no restriction.

    Returns
    -------
    np.ndarray (bool, same shape as distance_matrix)
    """
    dm = np.asarray(distance_matrix)

    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError(f"distance_matrix must be square, got shape {dm.shape}")

    if maximum_contact_distance is None:
        return np.ones_like(dm, dtype=bool)

    return dm <= maximum_contact_distance

# Copied directly from FrustratometeR's repo because I can't find how to access it from the package 
def compute_mask(
    distance_matrix: np.ndarray,
    maximum_contact_distance: Optional[float] = 9.5,
    minimum_sequence_separation: Optional[int] = 1,
) -> np.ndarray:

    n = distance_matrix.shape[0]

    seq_mask = compute_sequence_separation_mask(
        n, minimum_sequence_separation
    )

    dist_mask = compute_distance_mask(
        distance_matrix, maximum_contact_distance
    )

    return seq_mask & dist_mask
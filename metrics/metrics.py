"""
Basic metrics functions to use with pytorch.
"""
import numpy as np


def expected_calibration_error(data_class, data_probability, k=100):
    n_samples = data_class.shape[0]
    # Sort data:
    idx = np.argsort(data_probability)
    sorted_class = data_class[idx]
    sorted_probability = data_probability[idx]
    score = 0.0
    for i in range(k):
        # Rate of positives in slice
        o_i = np.mean(sorted_class[int(i * n_samples / k): int((i + 1) * n_samples / k)])
        # Average predicted probability
        e_i = np.mean(sorted_probability[int(i * n_samples / k): int((i + 1) * n_samples / k)])
        score += np.abs(o_i - e_i) / k
    return score


def ece_torch(data_class, data_probability, k=100):
    # Same as expected_calibration_error() but for torch tensors.
    import torch
    n_samples = data_class.data.shape[0]
    # Sort data:
    sorted_probability, idx = torch.sort(data_probability.data, 0)
    # Make into vector
    idx = idx.view(n_samples)
    # Sort also class-vector:
    sorted_class = data_class.data[idx]
    score = 0.0
    for i in range(k):
        # Rate of positives in slice
        o_i = torch.mean(sorted_class[int(i * n_samples / k): int((i + 1) * n_samples / k)])
        # Average predicted probability
        e_i = torch.mean(sorted_probability[int(i * n_samples / k): int((i + 1) * n_samples / k)])
        score += np.abs(o_i - e_i) / k
    return(score)

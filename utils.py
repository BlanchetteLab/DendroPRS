"""
General utilities
"""
import os
import shutil
import numpy as np


def create_directory(path, remove_curr):
    if os.path.exists(path):
        if remove_curr:
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def split_indices(idx_list, seed=0, train_percentage=0.7):
    """
    param idx_list: a list of indices (or IDs) to be split into train and test sets
    param seed: random seed for repeatability
    train_percentage: portion of the data allocated to train set
    returns: set of train indices and set of test indices
    """
    np.random.seed(seed=seed)
    train_idx = list()
    test_idx = list()
    for idx in idx_list:
        if np.random.uniform(0.0, 1.0) < train_percentage:
            train_idx.append(idx)
        else:
            test_idx.append(idx)
    return train_idx, test_idx

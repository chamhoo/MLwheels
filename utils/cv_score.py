"""
auther: leechh
"""
import numpy as np
from time import time
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def cv(model, x, y, split_method=StratifiedKFold, metrics_function=accuracy_score, nfolds=5, seed=2020):
    """
    :param model: sklearn-like model
    :param x: numpy.narray, shape: [n_samples, n_features]
    :param y: numpy.narray, shape: [n_samples, ]
    :param split_method: sklearn.model_selection
    :param metrics_function: metrics function, Take two parameter (y_true, y_pred)
    :param nfolds: Number of folds. Must be at least 2.
    :param seed: int, RandomState instance, None.
    :return: score
    """
    start_time = time()
    score_lst = list()
    # split dataset and K-Folds cross-validation
    split = split_method(n_splits=nfolds, shuffle=True, random_state=seed)
    for train_idx, valid_idx in split.split(x, y):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_valid = x[valid_idx]
        y_valid = y[valid_idx]
        # copy model & fit
        sub_model = deepcopy(model)
        sub_model.fit(x_train, y_train)
        y_pred = sub_model.predict(x_valid)
        subscore = metrics_function(y_valid, y_pred)
        score_lst.append(subscore)
    score = np.mean(score_lst)
    print(f"Total use time: {round(time() - start_time, 4)}s, score: {score}")
    return score

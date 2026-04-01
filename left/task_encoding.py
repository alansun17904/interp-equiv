

import cProfile
import itertools
import os
import pickle
import random
import re
import sys

import fire
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from sklearn.model_selection import train_test_split
import tqdm
import torch


def encoder_fit(X, Y):
    # lin = Ridge(solver="lsqr")
    # lin = LinearRegression()
    X=np.array(X)
    Y=np.array(Y)
    lin = RidgeCV(alpha_per_target=False, fit_intercept=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    lin.fit(X_train, Y_train)
    return np.sqrt(max(lin.score(X_test, Y_test), 0))


def post_intervention(reprs1, reprs1_post, reprs2):
    tot = 0
    for i in range(20):
        rnd_idx = random.randint(0, len(reprs1) - 1)

        other_idx = random.randint(0, len(reprs2) - 1)

        base, inter = reprs1[rnd_idx], reprs1_post[rnd_idx]
        target = reprs2[other_idx]

        first = encoder_fit(base, target)
        secon = encoder_fit(inter, target)

        if first > secon:
            tot += 1
    return tot / 20


def pre_intervention(reprs1, reprs2):
    tot = 0
    for i in range(20):
        rnd_idx = random.randint(0, len(reprs1) - 1)
        other_idx = random.randint(0, len(reprs2) - 1)

        base, inter = reprs1[rnd_idx], reprs1[other_idx]
        target = reprs2[other_idx]

        first = encoder_fit(base, inter)
        secon = encoder_fit(base, target)

        if first > secon:
            tot += 1
    return tot / 20


def reorder(reprs):
    k = reprs.keys()
    cases = list({v[0] for v in k})
    d = dict()
    for c in cases:
        d[c] = []
        for k2 in filter(lambda x: x[0] == c, k):
            d[c].append(reprs[k2])
    return d


def main(mode:str="intervention", n_comps:int=1, fw=True):
    fw = "fw" if fw else "bw"

    null_reprs = pickle.load(open(f"task-compressed-inter-0-{fw}.pkl", "rb"))
    alt_reprs = pickle.load(open(f"task-compressed-inter-{n_comps}-{fw}.pkl", "rb"))

    null_reprs = reorder(null_reprs)
    alt_reprs = reorder(alt_reprs)

    grid = dict()

    for i in tqdm.tqdm(null_reprs.keys()):
        for j in tqdm.tqdm(alt_reprs.keys()):
            if mode == "intervention":
                grid[(i,j)] = post_intervention(null_reprs[i], alt_reprs[i], null_reprs[j])
            else:
                grid[(i,j)] = pre_intervention(null_reprs[i], null_reprs[j])

    pickle.dump(grid, open(f"selectivity-task-{fw}-{mode}-{n_comps}.pkl", "wb"))


if __name__ == "__main__":
    fire.Fire(main)

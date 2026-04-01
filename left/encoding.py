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


def compress(X):
    pca = PCA()
    return pca.fit_transform(X)


def encoder_fit(X, Y):
    # lin = Ridge(solver="lsqr")
    # lin = LinearRegression()
    lin = RidgeCV(alpha_per_target=False, fit_intercept=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    lin.fit(X_train, Y_train)
    return lin.score(X_test, Y_test)


def branches(algo_repr):
    max_reprs = [len(v) for v in algo_repr]

    bf = max_reprs[0]
    coors = []
    for i in range(max_reprs[-1]):
        coors.append(
            [i // bf ** (len(max_reprs) - j - 1) for j in range(len(max_reprs))]
        )
    return coors


def collate_reprs(algo_repr):
    max_reprs = [len(v) for v in algo_repr]

    all_reprs = []
    coors = branches(algo_repr)

    for coor in tqdm.tqdm(coors, desc="preparing reprs"):
        repr_path = [v[coor[i]] for i, v in enumerate(algo_repr)]
        compressed = np.array(torch.cat(repr_path, dim=1))
        all_reprs.append(compressed)
    return all_reprs


def fit_across_group(g1_reprs, g2_reprs):
    s = []
    for i in tqdm.tqdm(range(len(g1_reprs))):
        s.append(encoder_fit(g1_reprs[i], g2_reprs[max(len(g2_reprs) - i - 1, 0)]))
        # for j in range(max(len(g2_reprs), 5)):
        #     s.append(encoder_fit(g1_reprs[i], g2_reprs[j]))
    return s


def permutation_test(prim_reprs, other_reprs, perms, comps=20):
    """
    Args:
        prim_reprs: Primary representations
        other_reprs: Other representations
        perms: Total number of across group permutations
        comps: Total number of comparisons to make across group
    """

    r1 = np.random.choice(np.arange(len(prim_reprs)), size=comps)
    r2 = np.random.choice(np.arange(len(other_reprs)), size=comps)

    # get reprs
    g1_reprs = [prim_reprs[v] for v in r1]
    g2_reprs = [other_reprs[v] for v in r2]

    # compute default across group comparison

    Tobs = fit_across_group(g1_reprs, g2_reprs)

    print(Tobs)

    combined = [*g1_reprs, *g2_reprs]

    Tks = []

    c = 0
    for i in tqdm.tqdm(range(perms)):
        random.shuffle(combined)
        Tk = fit_across_group(combined[:comps], combined[comps:])
        
        Tks.append(Tk)
        # print(Tk, Tobs)
        c += (Tk <= Tobs)

    return c / perms, (Tobs, Tks)


def ttest(algo: int, comps1=100, comps2=1):
    """
    Args:
        algo: index of the algorithm being group 1
        comps1: num models to sample from algorithm belonging in group 1
        comps2: num models to construct comparison group 2. 
    """
    prim_reprs = pickle.load(open(f"algo-compressed-{algo+1}.pkl", "rb"))

    prim_reprs = collate_reprs(prim_reprs)
    r1 = np.random.choice(np.arange(len(prim_reprs)), size=comps1)
    g1_reprs = [prim_reprs[v] for v in r1]

    stats = dict()
    stats[(algo, algo)] = fit_across_group(g1_reprs, g1_reprs)

    print("Self-Calibration:", np.mean(stats[(algo, algo)]))

    for i in range(6):
        if i == algo:
            continue

        other_reprs = pickle.load(open(f"algo-compressed-{algo+1}.pkl", "rb"))
        other_reprs = collate_reprs(other_reprs)
        r2 = np.random.choice(np.arange(len(other_reprs)), size=comps2)
        g2_reprs = [other_reprs[v] for v in r2]

        # stats[(algo, i)] = permutation_test(prim_reprs, other_reprs, 100)

        stats[(algo, i)] = fit_across_group(g1_reprs, g2_reprs)

        print(f"Comparing Algo {algo+1} with Algo {i+1}:", np.mean(stats[(algo, i)]))


    pickle.dump(stats, open(f"algo-{algo+1}-ttest-vsingle.pkl", "wb"))


def point_test(g1_reprs, g2_reprs, crit=1.96, n=20, n1=None,n2=None):
    p = 0

    if n1 is not None:
        idxs = np.random.choice(np.arange(len(g1_reprs)), size=n1, replace=False)
        g1_reprs = [g1_reprs[v] for v in idxs]

    if n2 is not None:
        idxs = np.random.choice(np.arange(len(g2_reprs)), size=n2, replace=False)
        g2_reprs = [g2_reprs[v] for v in idxs]

    for i in range(n):
        g1_idx1 = random.randint(0, len(g1_reprs) - 1)
        g1_idx2 = random.randint(0, len(g1_reprs) - 1)

        g2_idx = random.randint(0, len(g2_reprs) - 1)

        g11, g12 = g1_reprs[g1_idx1], g1_reprs[g1_idx2]
        g21 = g2_reprs[g2_idx]

        null_ = encoder_fit(g11, g12)
        alt_ = encoder_fit(g11, g21)

        if null_ > alt_:
            p += 1
    p = p / n
    return (p - crit * np.sqrt(p * (1-p)) / np.sqrt(n)) > 0.5



def main(algo:int=0, straps:int=40, n=20, n1=None, n2=None):

    reprs = [pickle.load(open(f"algo-compressed-{algo+1}.pkl", "rb")) for algo in range(6)]
    reprs = [collate_reprs(v) for v in reprs]

    grid = np.zeros((6,6))

    for alg1 in range(6):
        for alg2 in range(6):
            for i in tqdm.tqdm(range(int(straps))):
                grid[alg1, alg2] += point_test(reprs[alg1], reprs[alg2], n=n)

            print(grid[alg1, alg2] / straps)
            grid[alg1, alg2] = grid[alg1, alg2] / straps

    pickle.dump(grid, open(f"selectivity-bootstrap-{n}-{n1}-{n2}-score.pkl", "wb"))




if __name__ == "__main__":
    fire.Fire(main)

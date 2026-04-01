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
    return np.sqrt(max(lin.score(X_test, Y_test), 0))


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


def main(algo:int=0, n_comps:int=1):

    reprs = [pickle.load(open(f"algo-compressed-{alg+1}-bf-inter-1.pkl", "rb")) for alg in range(6)]
    reprs_inter = pickle.load(open(f"algo-compressed-{algo+1}-bf-inter-{n_comps+1}.pkl", "rb")) 
    reprs = [collate_reprs(v) for v in reprs]
    reprs_inter = collate_reprs(reprs_inter)

    print(len(reprs_inter))

    grid = np.zeros(6,)

    for alg1 in range(6):
        for i in tqdm.tqdm(range(min(len(reprs[alg1]), 100))):
            # pick a random starting point
            rnd_idx = random.randint(0, len(reprs_inter) - 1)
            base, inter = (
                reprs[algo][rnd_idx],
                reprs_inter[rnd_idx]
            )
            target = reprs[alg1][i]

            first = encoder_fit(base, target)
            secon = encoder_fit(inter, target)

            if first > secon:
                grid[alg1] += 1

        tot = min(len(reprs[alg1]), 100)
        print(grid[alg1] / tot)
        grid[alg1] = grid[alg1] / tot

    pickle.dump(grid, open(f"selectivity-bf-{algo}-{n_comps}.pkl", "wb"))

# do a one-tailed z-test, if grid[alg1] is very large then we reject the null that the two algos are different


if __name__ == "__main__":
    fire.Fire(main)

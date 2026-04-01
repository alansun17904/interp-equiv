import os
from pathlib import Path
import pickle
import random
import re
import sys

import fire
import torch
import tqdm
from functools import partial
import numpy as np
import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer
from circuits_benchmark.utils.get_cases import get_cases


### BENCHMARK ALGORITHMS
ALGO_1 = [59, 112]
ALGO_2 = [59, 48, 124]
ALGO_3 = [59, 74, 99, 124]
ALGO_4 = [59, 125]
ALGO_5 = [108, 99, 124]
ALGO_6 = [63, 108]
ALL_ALGOS = [ALGO_1, ALGO_2, ALGO_3, ALGO_4, ALGO_5, ALGO_6]


## MODEL_PATHS
CFG_PATH = "ll_model_cfg_510.pkl"
CASE_PATH = "admissible_tasks.pkl"
WEIGHT_PATH = "ll_model_510.pth"
CORR_PATH = "hl_ll_corr.pkl"


## TRAINING_PATHS
A = "arch-ll-models"
B = "corr-arch-ll-models"
C = "corr-ll-models"
D = "constant-ll-models"

MODELS = {
	48: [B, D],
	59: [B, D],
	63: [B, D],
    97: [B, D],
	74: [B, D],
	99: [B, D],
	108: [A, B, C, D],
	112: [A, B, C, D],
	124: [A, B, D],
	125: [A, B, D]
}

IDX = {A: 0, B: 1, C: 2, D: 3}


def patch_head(dst, hook, src, index):
    """Patching the output of an attention head before the final OV
    computation. `dst` has dimension (batch, sq_len, nhead, d_head)
    `src` has the same dimension. And hook is just a hook point. 
    """
    dst[:,:,index,:] = src[:,:,index,:]
    return dst


def setup_data(batch_size):
    """Sets up and returns the common benchmarking dataset.
    """

    # check if benchmark data already exists
    if os.path.exists("benchmark_ds.pkl"):
        print("Existing benchmark data already exists. Loading that...")
        clean_ds, corrupt_ds = pickle.load(open("benchmark_ds.pkl", "rb"))
    else:
        admissible = pickle.load(open(CASE_PATH, "rb"))
        case = list(admissible.values())[1]
        clean_ds = case.get_clean_data(min_samples=1000, max_samples=1000)
        corrupt_ds = case.get_corrupted_data(min_samples=1000, max_samples=1000)
        pickle.dump((clean_ds, corrupt_ds), open("benchmark_ds.pkl", "wb"))

    return (clean_ds.make_loader(batch_size=batch_size), corrupt_ds.make_loader(batch_size=batch_size))


def setup_data(batch_size):
    """Sets up and returns the common benchmarking dataset.
    """

    # check if benchmark data already exists
    if os.path.exists("benchmark_ds.pkl"):
        print("Existing benchmark data already exists. Loading that...")
        clean_ds, corrupt_ds = pickle.load(open("benchmark_ds.pkl", "rb"))
    else:
        admissible = pickle.load(open(CASE_PATH, "rb"))
        case = list(admissible.values())[1]
        clean_ds = case.get_clean_data(min_samples=1000, max_samples=1000)
        corrupt_ds = case.get_corrupted_data(min_samples=1000, max_samples=1000)
        pickle.dump((clean_ds, corrupt_ds), open("benchmark_ds.pkl", "wb"))

    return (clean_ds.make_loader(batch_size=batch_size), corrupt_ds.make_loader(batch_size=batch_size))


def load_model(mdir):
    """Given a directory for a low-level model, returns the model parameterized
    by the weights in the directory and the hl-ll correspondence.
    """
    mdir = Path(mdir)
    if not os.path.exists(mdir):
        return
    tf = HookedTransformer(pickle.load(open(mdir / CFG_PATH, "rb")))
    tf.load_state_dict(torch.load(mdir / WEIGHT_PATH))
    corr = pickle.load(open(mdir / CORR_PATH, "rb"))
    return tf, corr


def get_avaliable_mdirs(pdir):
    """For a given experiment setting, gets all of the models that is related to that setting.

    Args:
        pdir: directory of a given experimental setting. For example, /src/data/arch-ll-models
    """
    mdirs = os.listdir(pdir)
    mdir_ptrn = re.compile("c(\d+)-s(\d+)")
    dirs = dict()
    for d in mdirs:
        name_matched = re.match(mdir_ptrn, d)
        if not name_matched:
            continue

        case, seed = int(name_matched.group(1)), int(name_matched.group(2))
        if case not in dirs:
            dirs[case] = [seed]
        else:
            dirs[case].append(seed)

    return dirs


@torch.inference_mode()
def get_reprs(model, loader):
    resids = None
    next_toks = None


    logits, cache = model.run_with_cache(loader)
    resid = cache.accumulated_resid(apply_ln=True)
    resid = resid.mean(dim=2)

    return logits[:,:,:12].argmax(2), resid.permute(1, 0, 2).flatten(start_dim=1).to("cpu")

    # for x in loader:
        # logits, cache = model.run_with_cache(loader)
        # resid = cache.accumulated_resid(apply_ln=True)

    #     resid = resid.mean(dim=2)

    #     # print(resid.shape)

    #     if resids is None:
    #         resids = resid
    #         next_toks = logits[:,:,:12].argmax(2)
    #     else:
    #         resids = torch.cat([resids, resid], dim=1)
    #         next_toks = torch.cat([next_toks, logits[:,:,:12].argmax(2)], dim=0)
    #     del cache

    # return next_toks, resids.permute(1, 0, 2).flatten(start_dim=1).to("cpu")



@torch.inference_mode()
def get_reprs_patch(model, clean, corr, comps):
    """
    Comps is a list of tuples where the first entry is the 

    name of the component ; the second entry is the index of the attention head
    """

    resids = None
    next_toks = None


    if len(comps) == 0:
        return get_reprs(model, clean)


    _, corr_cache = model.run_with_cache(corr)
    
    hooks = [
        (v[0], partial(patch_head, src=corr_cache[v[0]], index=v[1]))
        for v in comps
    ]

    with model.hooks(fwd_hooks=hooks):
        logits, cache = model.run_with_cache(clean)

    resid = cache.accumulated_resid(apply_ln=True)
    resid = resid.mean(dim=2)

    # print(resid.shape)

    # if resids is None:
    #     resids = resid
    #     next_toks = logits[:,:,:12].argmax(2)
    # else:
    #     resids = torch.cat([resids, resid], dim=1)
    #     next_toks = torch.cat([next_toks, logits[:,:,:12].argmax(2)], dim=0)
    # del cache

    return logits[:,:,:12].argmax(2), resid.permute(1, 0, 2).flatten(start_dim=1).to("cpu")

    # for i in range(len(clean)):
    #     _, corr_cache = model.run_with_cache(corr[i])
        
    #     hooks = [
    #         (v[0], partial(patch_head, src=corr_cache[v[0]], index=v[1]))
    #         for v in comps
    #     ]

    #     with model.hooks(fwd_hooks=hooks):
    #         logits, cache = model.run_with_cache(clean[i])

    #     resid = cache.accumulated_resid(apply_ln=True)


    #     resid = resid.mean(dim=2)

    #     # print(resid.shape)

    #     if resids is None:
    #         resids = resid
    #         next_toks = logits[:,:,:12].argmax(2)
    #     else:
    #         resids = torch.cat([resids, resid], dim=1)
    #         next_toks = torch.cat([next_toks, logits[:,:,:12].argmax(2)], dim=0)
    #     del cache

    # return next_toks, resids.permute(1, 0, 2).flatten(start_dim=1).to("cpu")


def get_avaliable_mstrs(exps, case, mdirs):
    mstrs = []
    for exp, mdir in zip(exps, [mdirs[IDX[v]] for v in exps]):
        for seed in mdir[case]:
            mstrs.append((exp, case, seed))
    return mstrs


def reverse_last_columns(x: torch.Tensor) -> torch.Tensor:
    """
    Reverses the last (cols - 1) columns of a 2D tensor x of shape [n, cols].

    Parameters:
    x (torch.Tensor): A 2D tensor of shape [n, cols].

    Returns:
    torch.Tensor: A tensor with the last (cols - 1) columns reversed.
    """
    if x.ndim != 2:
        raise ValueError("Input tensor must be 2-dimensional")
    
    n, cols = x.shape
    if cols < 2:
        return x.clone()  # Nothing to reverse if only 1 column

    # Keep the first column, reverse the rest
    return torch.cat([x[:, :1], x[:, 1:].flip(dims=[1])], dim=1)


def process_ll_node(ll_node):
    if "attn" in ll_node.name:
        return ll_node.name, ll_node.index.as_index[2]


def main(algo: int, comps: int):
    np.random.seed(42)
	# first check the eval accuracies 
	# evals = [pickle.load(open(f"src/data/reprs/{lt}/eval-accuracy.pkl", "rb")) for lt in [A, B, C, D]]

	# def case_acc(case, eval_):
	# 	k = filter(lambda x: case == x[0], eval_.keys())
	# 	s, l = 0, 0
	# 	for ky in k:
	# 		s += eval_[ky][0]
	# 		l += 1
	# 	return s / l

	# for i, algo in enumerate(ALL_ALGOS):
	# 	total_acc = 1
	# 	for routine in algo:
	# 		avg_acc = 0
	# 		for mode in MODELS[routine]:
	# 			# print(routine, IDX[mode])
	# 			avg_acc += case_acc(routine, evals[IDX[mode]])
	# 		avg_acc /= len(MODELS[routine])
	# 	total_acc *= avg_acc
	# 	print(f"Algorithm {i+1}:", total_acc)

	# get the representations for each algorithm
    clean_ds, corr_ds = setup_data(1000)
    routines = ALL_ALGOS[algo]
    mdirs = [get_avaliable_mdirs(f"src/data/{lt}") for lt in [A, B, C, D]]
    
    for x, _ in clean_ds:
        start_toks_clean = x

    for x, _ in corr_ds:
        start_toks_corr = x

    prev_toks_clean = [start_toks_clean]
    prev_toks_corr = [start_toks_corr]
    reprs = []

    models, correspondences = [], [] 

    for i, routine in enumerate(routines):
        exps = MODELS[routine]  # all experimental settings available
        mstrs = get_avaliable_mstrs(exps, routine, mdirs)
        idxs = np.random.choice(np.arange(len(mstrs)), size=5, replace=False)
        
        model_names = [mstrs[v] for v in idxs]

        rmodels, rcorrs = [], []

        for mname in model_names:
            correspondence = pickle.load(open(Path(f"src/data/{mname[0]}/c{mname[1]}-s{mname[2]}") / CORR_PATH, "rb"))
            ll_nodes = [list(v) for v in correspondence.values()]
            nodes = []
            for l in ll_nodes:
                for node in l:
                    pnode = process_ll_node(node)
                    if pnode is not None:
                        nodes.append(pnode)

            rmodels.append(f"src/data/{mname[0]}/c{mname[1]}-s{mname[2]}")
            rcorrs.append(nodes)

        models.append(rmodels)
        correspondences.append(rcorrs)

    prev_patched = [comps for _ in range(5)]
    for i in range(len(correspondences)-1, -1, -1):
        rcorrs = correspondences[i]
        for j in range(len(rcorrs)):
            if prev_patched[j] == 0:
                correspondences[i][j] = []
            elif len(rcorrs[j]) > prev_patched[j]:
                correspondences[i][j] = correspondences[i][j][-prev_patched[j]:]
                prev_patched[j] = 0
            else:
                prev_patched[j] -= len(rcorrs[j])

    print(correspondences)

    for i, routine in tqdm.tqdm(enumerate(routines)):
        rmodels = models[i]
        rcorrs = correspondences[i]
        curr_nt, curr_reprs, curr_patched = [], [], []
        curr_corr_nt = []

        for j in range(len(rmodels)):
            model, _ = load_model(rmodels[j])
            this_patch = rcorrs[j]

            for k in range(len(prev_toks_clean)):
                pt, ct = prev_toks_clean[k], prev_toks_corr[k]


                # if len(nodes) < prev_patched[i]:
                #     this_patch = nodes
                #     curr_patched.append(prev_patched[i] - len(nodes))
                # else:
                #     this_patch = nodes[:prev_patched[i]]
                #     curr_patched.append(0)

                clean_nt, _ = get_reprs(model, pt)
                if (algo == 1 or algo == 0) and routine == 59:
                    clean_nt = reverse_last_columns(clean_nt)

                corr_nt, rep = get_reprs_patch(model, pt, ct, this_patch)
                if (algo == 1 or algo == 0) and routine == 59:
                    corr_nt = reverse_last_columns(corr_nt)

                curr_corr_nt.append(corr_nt)
                curr_nt.append(clean_nt)
                curr_reprs.append(rep)

        prev_patched = curr_patched
        prev_toks_clean = curr_nt
        prev_toks_corr = curr_corr_nt
        reprs.append(curr_reprs)


    pickle.dump(reprs, open(f"algo-compressed-{algo+1}-bf-inter-{comps+1}.pkl", "wb"))


if __name__ == "__main__":
	fire.Fire(main)
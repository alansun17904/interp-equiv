import re
import os
import sys
import fire
import tqdm
import torch
import pickle
from pathlib import Path
from itertools import product
import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer
from circuits_benchmark.utils.get_cases import get_cases


# load model
CFG_PATH = "ll_model_cfg_510.pkl"
CASE_PATH = "admissible_tasks.pkl"
WEIGHT_PATH = "ll_model_510.pth"
CORR_PATH = "hl_ll_corr.pkl"



def patch_head(dst, hook, src, index):
    """Patching the output of an attention head before the final OV
    computation. `dst` has dimension (batch, sq_len, nhead, d_head)
    `src` has the same dimension. And hook is just a hook point.
    """
    dst[index] = src[index]
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
    for x, _ in loader:
        _, cache = model.run_with_cache(x)
        resid = cache.accumulated_resid(incl_mid=False)

        if resids is None:
            resids = resid
        else:
            resids = torch.cat([resids, resid], dim=1)
        del cache

    return resids.permute(1, 0, 2, 3).flatten(start_dim=1)


@torch.inference_mode()
def eval_model(model, hl_model, loader):
    model.eval()

    def accuracy(pred, labe):
        return (pred.argmax(dim=-1) == labe.argmax(dim=-1)).float().mean()

    correct = 0
    batches = 0

    for x, _ in loader:
        logit = model(x)
        label = hl_model(x)

        correct += (logit.argmax(dim=-1) == label.argmax(dim=-1)).float().mean()
        batches += 1

    return (correct / batches).item()



def main(exp_dir, out_dir, batch_size=64, eval=False, intervene=False, n_nodes=1, in_out=True):
    """Generates and stores representations under various types of interventions.
    This acts as the first step to computing the alignment between any two models.

    Params:
        out_name (str): filename to store the representations
        intervene (bool): whether to retrieve representations under
            interventions of attention heads. If this is true, then
            additional files will be stored that denote which attention
            heads were intervened on during the course of representation
            retrieval.
        n_nodes (int): the number of nodes to intervene on.
        in_out (bool): determines which nodes are being intervened on, those
            within the circuit, or those outside of the circuit. If this is true,
            then we are sampling n_nodes from within the circuit to intervene on.
    """
    exp_dir, out_dir = Path(exp_dir), Path(out_dir)
    mdirs = get_avaliable_mdirs(exp_dir)
    clean_ds, corrupt_ds = setup_data(batch_size)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if eval:
        print(f"Running evaluation from models in experimental setting `{exp_dir}`")
        cases = pickle.load(open(CASE_PATH, "rb"))
        acc = dict()

        for case, seeds in mdirs.items():
            if case not in cases:
                continue
            hl_model = cases[case].get_hl_model()

            if hl_model.cfg.n_ctx != 10:
                continue

            for seed in seeds:
                model, _ = load_model(exp_dir / f"c{case}-s{seed}")
                clean_acc = eval_model(model, hl_model, clean_ds)
                corr_acc = eval_model(model, hl_model, corrupt_ds)

                acc[(case, seed)] = (clean_acc, corr_acc)

        pickle.dump(acc, open(out_dir / f"eval-accuracy.pkl", "wb"))
        sys.exit(0)

    if not intervene:
        print(f"Getting representations from models in experimental setting `{exp_dir}` without intervention")
        ls = sum(len(v) for v in mdirs.items())
        print(f"Total {ls} models")
        pbar = tqdm.tqdm(total=ls)

        for case, seeds in mdirs.items():
            for seed in seeds:
                pbar.set_description(f"Case: {case}; Seed: {seed}")

                model, _ = load_model(exp_dir / f"c{case}-s{seed}")
                model.eval()
                if model.cfg.n_ctx != 10:
                    pbar.update(1)
                    continue

                reprs = get_reprs(model, clean_ds)
                pickle.dump(reprs, open(out_dir / f"c{case}-s{seed}-reprs.pkl", "wb"))

                pbar.update(1)
        pbar.close()
        sys.exit(0)




    # # get all possible attn_head hooks that are a part of the model
    # act_names = list(filter(lambda x: "attn.hook_result" in x, tf.hook_dict.keys()))



    # correspondence = case.get_correspondence()
    # n_heads = tf.cfg.n_heads
    # head_idx = Index
    # head_idx = list(product(range(n_layers), range(n_heads)))
    


if __name__ == "__main__":
    fire.Fire(main)

import argparse
import random
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer

from ioi_dataset import IOIDataset


DEFAULT_MODELS = [
    "gpt2-small",
    "gpt2-medium",
    "pythia-160m",
    "pythia-410m",
    "pythia-1.4b",
    "pythia-2.8b",
]


class ObjectData(Dataset):
    def __init__(self, data, labels, subjects):
        self.data = data
        self.labels = labels
        self.subjects = subjects

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.subjects[idx]

    def __len__(self):
        return len(self.data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_label(ht, sentence):
    toks = ht.to_str_tokens(sentence, prepend_bos=False)
    label = toks[-1]
    return "".join(toks[:-1]), label


def get_loaders(ht, n_samples: int, batch_size: int, seed: int):
    """Create matched corrupted/clean IOI dataloaders."""
    set_seed(seed)
    ioi = IOIDataset(
        prompt_type="mixed",
        N=n_samples,
        prepend_bos=False,
        tokenizer=ht.tokenizer,
    )
    abc = ioi.gen_flipped_prompts(("S2", "IO"))

    clean_data, clean_labels, clean_subject_words = [], [], []
    corrupt_data, corrupt_labels = [], []

    for i, sentence in enumerate(ioi.sentences):
        subject = ioi.ioi_prompts[i]["S"]
        clean_subject_words.append(" " + subject)
        prompt, label = extract_label(ht, sentence)
        clean_data.append(prompt)
        clean_labels.append(label)

    for sentence in abc.sentences:
        prompt, label = extract_label(ht, sentence)
        corrupt_data.append(prompt)
        corrupt_labels.append(label)

    corr_loader = DataLoader(
        ObjectData(corrupt_data, corrupt_labels, clean_subject_words),
        batch_size=batch_size,
        shuffle=False,
    )
    clean_loader = DataLoader(
        ObjectData(clean_data, clean_labels, clean_subject_words),
        batch_size=batch_size,
        shuffle=False,
    )
    return corr_loader, clean_loader


def build_model(model_name: str) -> HookedTransformer:
    return HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
    )


def get_reprs(model: str, n_samples: int = 1000, batch_size: int = 2, seed: int = 24):
    """Extract final-token accumulated residual stream representations."""
    set_seed(seed)
    ht = build_model(model)
    _, clean_loader = get_loaders(ht, n_samples, batch_size, seed)

    run_reprs = []
    for prompts, _, _ in clean_loader:
        toks = ht.to_tokens(prompts)
        _, cache = ht.run_with_cache(toks)
        resid = cache.accumulated_resid(incl_mid=False)[:, :, -1, :]
        run_reprs.append(resid)

    catted = torch.cat([v.permute(1, 0, 2) for v in run_reprs], dim=0)
    catted = catted.reshape(catted.size(0), -1)
    torch.save(catted, f"{model}-reprs.pth")
    return catted


def hook_fn(act, ct, hook, head_idx):
    """Patch one attention head activation with clean activations."""
    act[:, :, head_idx, :] = ct[:, :, head_idx, :]


def kl_from_logits(logits, clean_logits):
    log_probs = F.log_softmax(logits + 1e-5, dim=-1)
    clean_probs = F.softmax(clean_logits + 1e-5, dim=-1)
    return F.kl_div(log_probs, clean_probs, reduction="batchmean", log_target=False)


def patch_heads(model: str, n_samples: int, batch_size: int, seed: int):
    """Compute per-head patching KL scores against the clean run."""
    set_seed(seed)
    ht = build_model(model)
    corrupted_loader, clean_loader = get_loaders(ht, n_samples, batch_size, seed)
    grid = np.zeros((ht.cfg.n_layers, ht.cfg.n_heads), dtype=np.float32)

    for layer_idx in tqdm.tqdm(range(ht.cfg.n_layers), desc=f"{model} layers"):
        for head_idx in tqdm.tqdm(range(ht.cfg.n_heads), desc="heads", leave=False):
            total_kl = 0.0
            total_examples = 0

            for (corr_prompts, _, _), (clean_prompts, _, _) in zip(corrupted_loader, clean_loader):
                corr_toks = ht.to_tokens(corr_prompts)
                clean_toks = ht.to_tokens(clean_prompts)
                clean_logits, clean_cache = ht.run_with_cache(clean_toks)

                clean_act = clean_cache[f"blocks.{layer_idx}.attn.hook_z"]
                hook = partial(hook_fn, ct=clean_act, head_idx=head_idx)
                patched_logits = ht.run_with_hooks(
                    corr_toks,
                    fwd_hooks=[(f"blocks.{layer_idx}.attn.hook_z", hook)],
                )

                batch_kl = kl_from_logits(patched_logits[:, -1, :], clean_logits[:, -1, :])
                batch_size_now = clean_logits.size(0)
                total_kl += float(batch_kl.item()) * batch_size_now
                total_examples += batch_size_now

            grid[layer_idx, head_idx] = total_kl / max(total_examples, 1)
    return grid


def encoder_fit(X, Y):
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    lin = RidgeCV(alpha_per_target=False, fit_intercept=True)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=0
    )
    lin.fit(X_train, Y_train)
    return np.sqrt(max(lin.score(X_test, Y_test), 0))


def compute_transfer_matrix(models: Sequence[str]):
    """Compute sqrt(R^2) transfer matrix between saved representation files."""
    ent = []
    for i in range(len(models)):
        row = []
        for j in range(len(models)):
            repr_i = torch.load(f"{models[i]}-reprs.pth", map_location="cpu")
            repr_j = torch.load(f"{models[j]}-reprs.pth", map_location="cpu")
            row.append(encoder_fit(repr_i, repr_j))
        ent.append(row)
    return ent


def parse_args():
    parser = argparse.ArgumentParser(description="IOI representation extraction utilities.")
    parser.add_argument("--mode", choices=["reprs", "patch-heads", "transfer"], required=False)
    parser.add_argument("--model", default=DEFAULT_MODELS[0])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode is None:
        return

    if args.mode == "reprs":
        out = Path(args.output) if args.output else Path(f"{args.model}-reprs.pth")
        reps = get_reprs(args.model, args.n_samples, args.batch_size, args.seed)
        torch.save(reps, out)
        print(f"Saved representations to {out}")
        return

    if args.mode == "patch-heads":
        out = Path(args.output) if args.output else Path(f"{args.model}-heads.pth")
        grid = patch_heads(args.model, args.n_samples, args.batch_size, args.seed)
        torch.save(grid, out)
        print(f"Saved patching grid to {out}")
        return

    if args.mode == "transfer":
        matrix = compute_transfer_matrix(DEFAULT_MODELS)
        print(matrix)


if __name__ == "__main__":
    main()
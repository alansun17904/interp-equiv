import os
import fire
import torch
import seaborn as sns

from pathlib import Path


torch.set_grad_enabled(False)


def corr_(w1, w2):
    """
    Compute the correlation between two weight tensors.
    """
    w1_flat = w1.flatten()
    w2_flat = w2.flatten()

    w1_centered = w1_flat - torch.mean(w1_flat)
    w2_centered = w2_flat - torch.mean(w2_flat)

    n = len(w1_flat)
    covariance = torch.sum(w1_centered * w2_centered) / n

    std1 = torch.sqrt(torch.sum(w1_centered ** 2) / n)
    std2 = torch.sqrt(torch.sum(w2_centered ** 2) / n)

    return covariance / (std1 * std2)


def global_corr(p1, p2):
    """
    Load two transformer models, flatten all parameters, and compute their covariance.
    Returns a single number representing the covariance.
    """
    # Load model parameters
    params1 = torch.load(p1)
    params2 = torch.load(p2)

    corrs = []

    for k, v in params1.items():
         if "mask" in k or "IGNORE" in k:
             continue
         corrs.append(corr_(v, params2[k]))

    return sum(corrs) / len(corrs)


def main(case_path, case_id, min_id=1, max_id=5):
    cases = list(filter(lambda x: x.startswith(str(case_id) + "-"), os.listdir(case_path)))

    d = dict()

    base = Path(case_path)

    for i in range(min_id, max_id + 1):
        for j in range(i+ 1, max_id + 1):
            d[(i,j)] = global_corr(base / cases[i] / "ll_model_510.pth", base / cases[j] / "ll_model_510.pth")

    print(d)


if __name__ == "__main__":
    fire.Fire(main)

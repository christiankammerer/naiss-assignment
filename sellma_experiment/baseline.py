"""
Step 3: non-LLM baseline. Can simple statistics of the weights separate the corpora?

Features per model (all from the middle blocks only, no forward pass):
  - effective rank of each block's write matrices W_o.l and W_proj.l
    (effective rank = exp(entropy of the normalized singular values), Roy & Vetterli 2007)
  - effective rank of everything the LAST block writes ([W_o.3 | W_proj.3]) -- the cue
    Astra used to spot romaji in the paper (fewer letters -> fewer write directions)
  - Frobenius norms of each matrix

Two analyses:
  1. The table of raw values, so you can see whether any single number separates corpora.
  2. Leave-one-out nearest-neighbour classification on standardized features.
     NOTE: this uses the labels of the OTHER models, which the LLM never sees, so it is
     a generous upper bound on what simple statistics can do, not a like-for-like rival.

Usage:  python baseline.py            (reads blinded/, writes results/baseline.json)
"""
import argparse
import os

import numpy as np

from common import CORPORA, N_LAYER, load_json, save_json


def effective_rank(m):
    s = np.linalg.svd(m, compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def features(W):
    f = {}
    for l in range(N_LAYER):
        f[f"erank_W_o.{l}"] = effective_rank(W[f"W_o.{l}"])
        f[f"erank_W_proj.{l}"] = effective_rank(W[f"W_proj.{l}"])
        for n in ("W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"):
            f[f"norm_{n}.{l}"] = float(np.linalg.norm(W[f"{n}.{l}"]))
    last = N_LAYER - 1
    f["erank_last_block_writes"] = effective_rank(
        np.concatenate([W[f"W_o.{last}"], W[f"W_proj.{last}"]], axis=1))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    a = ap.parse_args()
    key = load_json(os.path.join(a.dir, "blinded", "KEY_DO_NOT_SHARE.json"))
    names = sorted(key)
    feats = {n: features(dict(np.load(os.path.join(a.dir, "blinded", f"{n}.npz"))))
             for n in names}

    # 1. Table of the headline statistic, grouped by corpus.
    print("Effective rank of everything the last block writes (higher = more directions):")
    order = sorted(names, key=lambda n: (str(key[n]["corpus"]), n))
    for n in order:
        print(f"  {n:16s} {str(key[n]['corpus']):10s} "
              f"{feats[n]['erank_last_block_writes']:8.1f}   "
              f"W_o.3 {feats[n]['erank_W_o.3']:6.1f}   W_proj.3 {feats[n]['erank_W_proj.3']:6.1f}")

    # 2. Leave-one-out 1-nearest-neighbour on the trained models.
    trained = [n for n in names if key[n]["corpus"] in CORPORA]
    cols = sorted(feats[trained[0]])
    X = np.array([[feats[n][c] for c in cols] for n in trained])
    y = [key[n]["corpus"] for n in trained]
    preds = []
    for i in range(len(trained)):
        mask = np.arange(len(trained)) != i
        mu, sd = X[mask].mean(0), X[mask].std(0) + 1e-9
        Z = (X - mu) / sd
        d = np.linalg.norm(Z[mask] - Z[i], axis=1)
        preds.append(np.array(y)[mask][d.argmin()])
    acc = float(np.mean([p == t for p, t in zip(preds, y)]))
    scandi = [i for i, t in enumerate(y) if t != "python"]
    acc_scandi = float(np.mean([preds[i] == y[i] for i in scandi])) if scandi else None
    print(f"\nLeave-one-out 1-NN accuracy, all corpora: {acc:.2f} (chance {1/len(set(y)):.2f})")
    if acc_scandi is not None:
        print(f"  on Scandinavian models only: {acc_scandi:.2f}")
    for n, t, p in zip(trained, y, preds):
        print(f"  {n:10s} true {t:10s} predicted {p}")

    save_json({"features": feats, "loo_1nn": {"accuracy": acc, "accuracy_scandinavian": acc_scandi,
               "predictions": dict(zip(trained, preds)), "truth": dict(zip(trained, y))}},
              os.path.join(a.dir, "results", "baseline.json"))
    print("\nWrote results/baseline.json")


if __name__ == "__main__":
    main()

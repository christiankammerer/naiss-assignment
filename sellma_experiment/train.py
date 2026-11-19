"""
Step 2: train the models, measure ground-truth head behaviour, and export blinded weights.

Architecture matches the paper's Experiment 3 exactly: 4 blocks, 6 heads, d = 384,
context 128, ReLU MLP (hidden 4d), no biases in linear layers, untied output layer,
init std 0.02 (0.007071 for W_o and W_proj), AdamW with weight decay 0.01 on matrices.
~7.2M parameters.

Usage (Colab, T4 GPU):
    !python train.py --iters 50 --quick    # smoke test (~1 min)
    !python train.py                       # 4 corpora x 2 seeds + dev model + random control
    !python train.py --out /content/drive/MyDrive/sellma   # save straight to Google Drive
Finished runs are skipped on restart, so a disconnect only loses the current model.

Outputs (in --out, default "."):
    runs/<corpus>_s<seed>/ckpt.pt         full model (ground truth; keep private)
    blinded/model_<X>.npz                 middle blocks only, neutral names -> LLM
    blinded/control_random.npz            untrained model (negative control) -> LLM
    blinded/KEY_DO_NOT_SHARE.json         answer key incl. measured head behaviour
    dev/model_dev.npz (+ KEY)             extra model ONLY for testing the LLM harness
    results/train_summary.json            settings, losses, cross-corpus loss matrix
"""
import argparse
import math
import os
import platform
import random
import string
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CONTEXT, CORPORA, D_MODEL, N_HEAD, N_LAYER, VOCAB, load_json, save_json

# --------------------------------------------------------------------------- model


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.W_fc = nn.Linear(d, 4 * d, bias=False)
        self.W_proj = nn.Linear(4 * d, d, bias=False)

    def _heads(self, t):
        B, T, C = t.shape
        return t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    def forward(self, x, return_att=False):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self._heads(self.W_q(h)), self._heads(self.W_k(h)), self._heads(self.W_v(h))
        if return_att:  # explicit attention, used only for ground-truth measurements
            att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool,
                                                        device=x.device), 1), float("-inf"))
            att = att.softmax(-1)
            y = att @ v
        else:
            att = None
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.W_o(y.transpose(1, 2).contiguous().view(B, T, C))
        x = x + self.W_proj(F.relu(self.W_fc(self.ln2(x))))
        return x, att


class CharGPT(nn.Module):
    def __init__(self, vocab_size=len(VOCAB), context=CONTEXT, n_layer=N_LAYER,
                 n_head=N_HEAD, d=D_MODEL):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)   # input layer (stripped)
        self.pos_emb = nn.Embedding(context, d)      # input layer (stripped)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)                   # output layer (stripped)
        self.W_out = nn.Linear(d, vocab_size)        # output layer (stripped), untied
        small = 0.02 / math.sqrt(2 * n_layer)        # = 0.007071 for 4 layers
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, 0.0, small if ("W_o" in name or "W_proj" in name) else 0.02)
        nn.init.zeros_(self.W_out.bias)

    def forward(self, idx, targets=None, return_att=False):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        atts = []
        for b in self.blocks:
            x, a = b(x, return_att)
            atts.append(a)
        logits = self.W_out(self.lnf(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, atts


def export_blocks(model):
    """Middle blocks only, with exactly the tensor names used in the LLM prompt."""
    out = {}
    for l, b in enumerate(model.blocks):
        out[f"ln1_w.{l}"], out[f"ln1_b.{l}"] = b.ln1.weight, b.ln1.bias
        out[f"ln2_w.{l}"], out[f"ln2_b.{l}"] = b.ln2.weight, b.ln2.bias
        for n in ("W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"):
            out[f"{n}.{l}"] = getattr(b, n).weight
    return {k: v.detach().float().cpu().numpy() for k, v in out.items()}

# --------------------------------------------------------------------------- data


def load_split(data_dir, corpus, split):
    return np.memmap(os.path.join(data_dir, corpus, f"{split}.bin"), dtype=np.uint8, mode="r")


def get_batch(data, batch_size, gen, device, context=CONTEXT):
    ix = torch.randint(len(data) - context - 1, (batch_size,), generator=gen).numpy()
    x = np.stack([data[i:i + context] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + context] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def amp(device):
    return torch.autocast("cuda", dtype=torch.float16) if device == "cuda" \
        else torch.autocast("cpu", enabled=False)


@torch.no_grad()
def eval_loss(model, data, device, n_batches, batch_size=64):
    model.eval()
    gen = torch.Generator().manual_seed(1234)  # identical eval batches for every model
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, gen, device)
        with amp(device):
            losses.append(model(x, y)[1].item())
    model.train()
    return float(np.mean(losses))

# --------------------------------------------------------------------------- ground truth heads


@torch.no_grad()
def measure_heads(model, val_data, device, induction_threshold=0.2, prev_threshold=0.5):
    """Measure what each head does by RUNNING the model (the LLM is not allowed to).
    - previous score : mean attention to position t-1 on real text
    - induction score: on a random character sequence repeated twice, mean attention
                       from position t (2nd copy) to the token after t's earlier occurrence
    - mean distance  : average |t - s| of attention on real text
    """
    model.eval()
    gen = torch.Generator().manual_seed(0)
    x, _ = get_batch(val_data, 32, gen, device)
    _, _, att_text = model(x, return_att=True)

    L = 60  # repeated random sequence: length 2L <= 128
    letters = torch.randint(2, len(VOCAB), (32, L), generator=gen)
    rep = torch.cat([letters, letters], 1).to(device)
    _, _, att_rep = model(rep, return_att=True)

    T = x.shape[1]
    pos = torch.arange(T, device=device)
    dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()
    heads = {}
    for l in range(len(model.blocks)):
        for h in range(model.blocks[l].n_head):
            a = att_text[l][:, h]                                   # (B, T, T)
            prev = a[:, 1:, :].diagonal(offset=0, dim1=1, dim2=2).mean().item()
            mean_dist = (a * dist).sum(-1)[:, 1:].mean().item()
            r = att_rep[l][:, h]
            t = torch.arange(L, 2 * L, device=device)
            ind = r[:, t, t - L + 1].mean().item()
            if ind > induction_threshold:
                label = "copy"
            elif prev > prev_threshold:
                label = "previous"
            elif mean_dist < 5:
                label = "local"
            else:
                label = "broad"
            heads[f"L{l}H{h}"] = {"label": label, "induction": round(ind, 3),
                                  "previous": round(prev, 3), "mean_distance": round(mean_dist, 2)}
    model.train()
    return heads

# --------------------------------------------------------------------------- training


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(it, a):
    if it < a.warmup:
        return a.lr * (it + 1) / a.warmup
    p = (it - a.warmup) / max(1, a.iters - a.warmup)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def train_one(corpus, seed, val_sets, a, device):
    set_seed(seed)
    model = CharGPT().to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    train = load_split(a.data_dir, corpus, "train")
    gen = torch.Generator().manual_seed(seed)
    t0, log = time.time(), []
    for it in range(a.iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, a)
        x, y = get_batch(train, a.batch_size, gen, device)
        with amp(device):
            _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if it % a.log_every == 0 or it == a.iters - 1:
            vl = eval_loss(model, val_sets[corpus], device, n_batches=4)
            log.append({"iter": it, "train_loss": round(loss.item(), 4), "val_loss": round(vl, 4)})
            print(f"  [{corpus} s{seed}] it {it:5d}  train {loss.item():.3f}  val {vl:.3f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    return model, {"seconds": round(time.time() - t0, 1), "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default=".")
    ap.add_argument("--corpora", nargs="+", default=CORPORA)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument("--blind_seed", type=int, default=42)
    ap.add_argument("--no_dev", action="store_true", help="skip the harness-test model")
    ap.add_argument("--quick", action="store_true", help="smoke test: one seed, few evals")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.log_every, a.eval_batches = a.seeds[:1], 25, 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cpu" and not a.quick:
        print("WARNING: no GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")
    n_params = sum(p.numel() for p in CharGPT().parameters())
    print(f"Parameters per model: {n_params:,}  |  characters seen per model: "
          f"{a.iters * a.batch_size * CONTEXT:,}")
    val_sets = {c: load_split(a.data_dir, c, "val") for c in a.corpora}
    P = lambda *p: os.path.join(a.out, *p)

    jobs = [(c, s) for c in a.corpora for s in a.seeds]
    if not a.no_dev:
        jobs.append(("swedish", 100))  # harness-test model, never evaluated
    trained = {}
    for corpus, seed in jobs:
        ckpt = P("runs", f"{corpus}_s{seed}", "ckpt.pt")
        if os.path.exists(ckpt):
            print(f"Found {ckpt}, skipping training")
            saved = torch.load(ckpt, map_location=device)
            model = CharGPT().to(device)
            model.load_state_dict(saved["state_dict"])
            info = saved["info"]
        else:
            print(f"Training {corpus}, seed {seed}", flush=True)
            model, info = train_one(corpus, seed, val_sets, a, device)
            info["val_loss_by_corpus"] = {c: round(eval_loss(model, val_sets[c], device,
                                                             a.eval_batches), 4)
                                          for c in a.corpora}
            info["heads"] = measure_heads(model, val_sets[corpus], device)
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "corpus": corpus, "seed": seed,
                        "info": info, "args": vars(a)}, ckpt)
        trained[(corpus, seed)] = (export_blocks(model), info)

    # Blinded evaluation set: shuffled, neutral letters, key kept separately.
    eval_jobs = [j for j in jobs if j[1] != 100]
    rng = random.Random(a.blind_seed)
    rng.shuffle(eval_jobs)
    os.makedirs(P("blinded"), exist_ok=True)
    key = {}
    for letter, (corpus, seed) in zip(string.ascii_uppercase, eval_jobs):
        blocks, info = trained[(corpus, seed)]
        np.savez(P("blinded", f"model_{letter}.npz"), **blocks)
        key[f"model_{letter}"] = {"corpus": corpus, "seed": seed, "heads": info["heads"]}
    set_seed(999)
    np.savez(P("blinded", "control_random.npz"), **export_blocks(CharGPT()))
    key["control_random"] = {"corpus": None, "seed": 999, "note": "untrained"}
    save_json(key, P("blinded", "KEY_DO_NOT_SHARE.json"))

    if not a.no_dev:
        os.makedirs(P("dev"), exist_ok=True)
        blocks, info = trained[("swedish", 100)]
        np.savez(P("dev", "model_dev.npz"), **blocks)
        save_json({"model_dev": {"corpus": "swedish", "seed": 100, "heads": info["heads"]}},
                  P("dev", "KEY_dev.json"))

    runs = [{"corpus": c, "seed": s, **trained[(c, s)][1]} for c, s in jobs]
    save_json({"args": vars(a), "params_per_model": n_params, "device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "torch": torch.__version__, "python": platform.python_version(),
               "manifest": load_json(os.path.join(a.data_dir, "manifest.json"))
               if os.path.exists(os.path.join(a.data_dir, "manifest.json")) else None,
               "runs": runs}, P("results", "train_summary.json"))

    print("\nCross-corpus validation loss (rows: model, cols: eval corpus; lower = better)")
    print(f"{'model':14s}" + "".join(f"{c[:9]:>11s}" for c in a.corpora))
    for r in runs:
        print(f"{r['corpus'][:9] + '_s' + str(r['seed']):14s}" +
              "".join(f"{r['val_loss_by_corpus'][c]:11.3f}" for c in a.corpora))
    print("\nCopy (induction) heads per model:")
    for r in runs:
        print(f"  {r['corpus']}_s{r['seed']}: " +
              (", ".join(h for h, v in r["heads"].items() if v["label"] == "copy") or "none"))
    print("\nDone. Next: python baseline.py  and  python make_prompts.py")


if __name__ == "__main__":
    main()

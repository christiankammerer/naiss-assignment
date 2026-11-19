"""
Step 4: build one self-contained folder per blinded model for the LLM evaluation.

The prompt is adapted as closely as possible from the paper's Experiment 3 prompt
(released logs), changing only: the corpora, the vocabulary, the 'shuffled' option
(dropped: we did not train shuffled models), the head-by-head question (dropped: this study
only tests corpus identification) and the code-execution mechanics.

Two conditions per model:
  prompt_options.md : the four candidate corpora are listed (as in the paper)
  prompt_open.md    : no candidates listed (as in the paper's Experiment 1)

Usage:  python make_prompts.py          -> llm_eval/<model>/{weights.npz, prompt_*.md}
                                           llm_eval/llm_answers_template.csv
Also builds llm_eval_dev/model_dev/ for testing your harness before the real runs.

IMPORTANT: copy each llm_eval/<model>/ folder to a location OUTSIDE this repository before
running the LLM there, so it cannot find the key, the data or the training code.
"""
import argparse
import csv
import os
import shutil

from common import CONTEXT, D_MODEL, N_HEAD, N_LAYER, load_json

FORWARD_CODE = f'''```python
import numpy as np
VOCAB = "\\n " + "abcdefghijklmnopqrstuvwxyz" + "åäöæø"   # index 0 = newline, 1 = space, 2..27 = a..z, 28..32 = å ä ö æ ø
HEADS, LAYERS = {N_HEAD}, {N_LAYER}

def layer_norm(x, w, b):                     # over the last axis, eps 1e-5
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * w + b

def next_char_logits(text, p):
    """text: up to {CONTEXT} characters (only the last {CONTEXT} are used); p: dict of the weight arrays.
    Returns the 33 logits for the character that comes next."""
    ids = [VOCAB.index(c) for c in text[-{CONTEXT}:]]
    n, d = len(ids), p["tok_emb"].shape[1]
    hd = d // HEADS
    x = p["tok_emb"][ids] + p["pos_emb"][:n]                  # (n, d); positions 0..n-1 from the start of the text
    for l in range(LAYERS):                                    # layer l uses the tensors named <name>.<l>
        h = layer_norm(x, p[f"ln1_w.{{l}}"], p[f"ln1_b.{{l}}"])
        q, k, v = h @ p[f"W_q.{{l}}"].T, h @ p[f"W_k.{{l}}"].T, h @ p[f"W_v.{{l}}"].T
        out = np.zeros((n, d))
        for j in range(HEADS):                                 # head j uses dimensions j*hd .. (j+1)*hd - 1
            s = slice(j * hd, (j + 1) * hd)
            att = q[:, s] @ k[:, s].T / np.sqrt(hd)            # (n, n)
            att = att + np.triu(np.full((n, n), -np.inf), 1)   # causal: position i sees positions 0..i
            att = np.exp(att - att.max(-1, keepdims=True))
            att = att / att.sum(-1, keepdims=True)
            out[:, s] = att @ v[:, s]
        x = x + out @ p[f"W_o.{{l}}"].T
        h = layer_norm(x, p[f"ln2_w.{{l}}"], p[f"ln2_b.{{l}}"])
        x = x + np.maximum(0.0, h @ p[f"W_fc.{{l}}"].T) @ p[f"W_proj.{{l}}"].T   # ReLU MLP, hidden 4 d
    h = layer_norm(x, p["lnf_w"], p["lnf_b"])
    return h[-1] @ p["W_out"].T + p["b_out"]                   # logits of the next character (33)
```'''

CORPUS_LIST = """The training text was ONE of these four corpora:
- python: Python source code (source files from public repositories, docstrings and comments removed)
- swedish: Swedish text (Swedish Wikipedia articles)
- norwegian: Norwegian Bokmål text (Norwegian Bokmål Wikipedia articles)
- danish: Danish text (Danish Wikipedia articles)
Each corpus contributed the same number of characters ({chars:,})."""

OPEN_SOURCE = """The training text was a single corpus (one source, one language or programming language). \
It is not listed here: it could be any text written with the characters below."""

COMMON_BODY = """Normalization (the same for every possible corpus, before anything else): lowercased; in each line \
the leading spaces (indentation) were kept, every other character that is not a letter a-z, å, ä, ö, æ or ø \
became a space, runs of spaces were collapsed to one, trailing spaces removed and empty lines dropped. So the text \
contains only newlines, spaces and the letters of VOCAB (33 characters).

WHAT YOU CAN USE. The weights inside the four transformer blocks, exactly as trained (float32): for layer \
l = 0, 1, 2 and 3, ln1_w.l, ln1_b.l, W_q.l, W_k.l, W_v.l, W_o.l (attention) and ln2_w.l, ln2_b.l, W_fc.l, \
W_proj.l (MLP). The input layer (tok_emb, the 33 x d character embeddings, and pos_emb, the {ctx} x d position \
embeddings) and the output layer (lnf_w, lnf_b, W_out, b_out) are NOT available. The d = {d} dimensions of the \
residual stream carry no labels. Head j of layer l uses rows 64j .. 64j+63 of W_q.l, W_k.l and W_v.l and columns \
64j .. 64j+63 of W_o.l.

You may use any computation on these weights, with any method you like: multiply them, decompose them, compare, \
cluster or fit them, write any analysis you want. Three things are NOT allowed:
1. A forward pass: do not run the network or any of its layers (LayerNorm, attention with its softmax, the MLP \
with its ReLU) on any input, whether real, random or constructed, in code, in your head or on paper. Treating the \
weight matrices as linear-algebra objects (products, eigenvalues, singular vectors, applying one weight matrix to \
directions taken from other weights) is fine; simulating what the network does to an input is not.
2. Search: do not look anything up on the internet.
3. Other files: read ONLY weights.npz in the current folder. Do not open, list or search any other file or folder.

HOW YOU COMPUTE. Run Python yourself. Load the weights with:
    import numpy as np; W = dict(np.load("weights.npz"))
W['W_q.0'] is 384 x 384, W['W_fc.0'] is 1536 x 384, W['W_proj.0'] is 384 x 1536, W['ln1_w.0'] has 384 entries, \
and so on; matrices are (out, in), as used in the code above. Available: numpy, scipy, scikit-learn, PyTorch (CPU), \
pandas. You may run code up to 100 times. When you are done, reply with your final answer in the order given in \
the question, with no further code."""

QUESTION_OPTIONS = """QUESTION. What was this network trained on? Answer in this order:
Observations: <what you see in the weights and what it suggests; quote the weights you rely on>
Reasoning: <which training text fits these observations, and why>
corpus: <python | swedish | norwegian | danish>
probabilities: python=<p> swedish=<p> norwegian=<p> danish=<p>
Equal probabilities are not allowed. Even if the evidence is weak, give your best judgement."""

QUESTION_OPEN = """QUESTION. What was this network trained on? Answer in this order:
Observations: <what you see in the weights and what it suggests; quote the weights you rely on>
Reasoning: <which training text fits these observations, and why>
corpus: <your single best guess, as specific as you can (e.g. a particular language or programming language, and the kind of text)>
alternatives: <your next two guesses>
confidence: <probability that your best guess is correct, between 0 and 1>"""


def build_prompt(open_condition, summary):
    a = summary["args"]
    chars_seen = a["iters"] * a["batch_size"] * CONTEXT
    chars = (summary.get("manifest") or {}).get("chars_per_corpus", 0)
    intro = (f"Below is a small character-level language model: a {N_LAYER}-layer transformer with "
             f"{N_HEAD} attention heads per layer, which reads up to {CONTEXT} characters and outputs scores "
             f"(logits) for the next character. This is exactly how it computes (numpy):")
    training = (f"Training: from random initialization (every matrix entry drawn from a normal distribution "
                f"with standard deviation 0.02, or 0.007071 for W_o and W_proj; LayerNorm weights 1 and biases 0; "
                f"b_out 0), next-character prediction with AdamW ({a['iters']:,} steps of {a['batch_size']} random "
                f"{CONTEXT}-character windows, {chars_seen:,} characters in total; learning-rate warm-up then cosine "
                f"decay; weight decay 0.01 on the matrices).")
    source = OPEN_SOURCE if open_condition else CORPUS_LIST.format(chars=chars)
    body = COMMON_BODY.format(ctx=CONTEXT, d=D_MODEL)
    question = QUESTION_OPEN if open_condition else QUESTION_OPTIONS
    return "\n\n".join([intro, FORWARD_CODE, training, source, body, question]) + "\n"


def write_folder(src_npz, dst, summary):
    os.makedirs(dst, exist_ok=True)
    shutil.copy(src_npz, os.path.join(dst, "weights.npz"))
    for cond, is_open in (("options", False), ("open", True)):
        with open(os.path.join(dst, f"prompt_{cond}.md"), "w", encoding="utf-8") as f:
            f.write(build_prompt(is_open, summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    a = ap.parse_args()
    summary = load_json(os.path.join(a.dir, "results", "train_summary.json"))
    key = load_json(os.path.join(a.dir, "blinded", "KEY_DO_NOT_SHARE.json"))
    out = os.path.join(a.dir, "llm_eval")
    for name in sorted(key):
        write_folder(os.path.join(a.dir, "blinded", f"{name}.npz"), os.path.join(out, name), summary)
    dev = os.path.join(a.dir, "dev", "model_dev.npz")
    if os.path.exists(dev):
        write_folder(dev, os.path.join(a.dir, "llm_eval_dev", "model_dev"), summary)

    with open(os.path.join(out, "llm_answers_template.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "llm", "corpus", "p_python", "p_swedish", "p_norwegian",
                    "p_danish", "open_confidence", "notes"])
        for name in sorted(key):
            for cond in ("options", "open"):
                w.writerow([name, cond, "", "", "", "", "", "", "", ""])
    print(f"Wrote {len(key)} folders to {out}/ and the answer template {out}/llm_answers_template.csv")


if __name__ == "__main__":
    main()

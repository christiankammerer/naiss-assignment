"""
Step 1: download and normalize the four corpora.

  python    : Python source files (codeparrot/codeparrot-clean-valid), comments and docstrings removed
  swedish   : Swedish Wikipedia      (wikimedia/wikipedia, 20231101.sv)
  norwegian : Norwegian Bokmal Wiki  (wikimedia/wikipedia, 20231101.no)
  danish    : Danish Wikipedia       (wikimedia/wikipedia, 20231101.da)

All corpora get the same character budget and the same normalization (see common.normalize).

Usage (Colab):  !python prepare_data.py            # 30M characters per corpus
                !python prepare_data.py --chars 5000000   # faster, smaller
Output: data/<corpus>/train.bin, val.bin (uint8 ids), data/manifest.json
"""
import argparse
import os
import re

from datasets import load_dataset
from huggingface_hub import HfApi

from common import CORPORA, VOCAB, encode, normalize, save_json

SOURCES = {
    "python": ("codeparrot/codeparrot-clean-valid", None, "content"),
    "swedish": ("wikimedia/wikipedia", "20231101.sv", "text"),
    "norwegian": ("wikimedia/wikipedia", "20231101.no", "text"),
    "danish": ("wikimedia/wikipedia", "20231101.da", "text"),
}
MIN_DOC_CHARS = 300


def strip_python(src: str) -> str:
    """Approximate removal of docstrings / triple-quoted strings and comments."""
    src = re.sub(r'(?s)(""".*?"""|\'\'\'.*?\'\'\')', "", src)
    lines = []
    for line in src.split("\n"):
        if line.lstrip().startswith("#"):
            continue
        lines.append(re.sub(r"\s+#.*$", "", line))  # may rarely cut a '#' inside a string
    return "\n".join(lines)


def collect(corpus, budget, revisions):
    name, config, field = SOURCES[corpus]
    ds = load_dataset(name, config, split="train", streaming=True, revision=revisions[name])
    parts, total, n_docs = [], 0, 0
    for row in ds:  # streaming order is fixed by the dataset, so this is reproducible
        raw = row[field]
        if corpus == "python":
            raw = strip_python(raw)
        t = normalize(raw)
        if len(t) < MIN_DOC_CHARS:
            continue
        parts.append(t)
        total += len(t) + 1
        n_docs += 1
        if total >= budget:
            break
    return "\n".join(parts)[:budget], n_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=30_000_000, help="characters per corpus")
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    api = HfApi()
    revisions = {n: api.dataset_info(n).sha for n in {s[0] for s in SOURCES.values()}}
    manifest = {"chars_per_corpus": args.chars, "vocab": VOCAB, "revisions": revisions,
                "corpora": {}}
    for corpus in CORPORA:
        print(f"Collecting {corpus} ...", flush=True)
        text, n_docs = collect(corpus, args.chars, revisions)
        ids = encode(text)
        split = int(len(ids) * (1 - args.val_fraction))
        d = os.path.join(args.out, corpus)
        os.makedirs(d, exist_ok=True)
        ids[:split].tofile(os.path.join(d, "train.bin"))
        ids[split:].tofile(os.path.join(d, "val.bin"))
        used = sorted(set(text) - {"\n", " "})
        manifest["corpora"][corpus] = {"source": SOURCES[corpus][:2], "documents": n_docs,
                                       "chars": len(text), "letters_used": "".join(used)}
        print(f"  {n_docs} documents, {len(text):,} chars, letters used: {''.join(used)}")
    save_json(manifest, os.path.join(args.out, "manifest.json"))
    print("Done. Wrote", os.path.join(args.out, "manifest.json"))


if __name__ == "__main__":
    main()

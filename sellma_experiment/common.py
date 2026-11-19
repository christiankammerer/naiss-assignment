"""Shared constants and helpers used by every script."""
import json
import os
import re
import unicodedata

import numpy as np

# Shared character vocabulary for ALL corpora, so every model has identical shapes.
# index 0 = newline, 1 = space, then a-z, then the Scandinavian letters.
LETTERS = "abcdefghijklmnopqrstuvwxyz" + "åäöæø"
VOCAB = "\n " + LETTERS
STOI = {c: i for i, c in enumerate(VOCAB)}

CORPORA = ["python", "swedish", "norwegian", "danish"]
SHORT = {"python": "py", "swedish": "sv", "norwegian": "no", "danish": "da"}

# Paper-matched architecture (Experiment 3): 4 blocks, 6 heads, d = 384, context 128.
N_LAYER, N_HEAD, D_MODEL, CONTEXT = 4, 6, 384, 128


def normalize(text: str) -> str:
    """Paper-style normalization: lowercase; keep each line's leading spaces
    (indentation); every other character that is not a letter becomes a space;
    collapse runs of spaces; strip trailing spaces; drop lines that end up empty."""
    text = unicodedata.normalize("NFC", text).lower().replace("\t", "    ")
    out = []
    for line in text.split("\n"):
        body = line.lstrip(" ")
        indent = line[: len(line) - len(body)]
        body = "".join(c if c in LETTERS else " " for c in body)
        body = re.sub(" +", " ", body).strip()
        if body:
            out.append(indent + body)
    return "\n".join(out)


def encode(text: str) -> np.ndarray:
    return np.frombuffer(bytes(STOI[c] for c in text), dtype=np.uint8)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
